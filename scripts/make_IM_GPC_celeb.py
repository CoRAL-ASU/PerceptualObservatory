import argparse
import re
import hashlib
import base64, io
from pathlib import Path
from functools import lru_cache
import numpy as np
import ujson as json
from tqdm import tqdm
from datasets import load_from_disk
from joblib import Parallel, delayed
from PIL import Image


# -------------------------------
# Helpers
# -------------------------------

def stable_seed_from_str(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=20000)
def decode_b64(b64_str: str):
    """Cached base64 → PIL.Image"""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def render_grid_b64(grid_b64s):
    """Return stitched 2×2 grid as base64 string"""

    imgs = [decode_b64(b64) for b64 in grid_b64s]
    w, h = imgs[0].size

    # Make 2×2 collage
    canvas = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
    canvas.paste(imgs[0], (0, 0))
    canvas.paste(imgs[1], (w, 0))
    canvas.paste(imgs[2], (0, h))
    canvas.paste(imgs[3], (w, h))

    # Resize back to original size
    canvas = canvas.resize((w, h), Image.Resampling.LANCZOS)

    # Encode to base64 (no disk I/O)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -------------------------------
# Label & Gender
# -------------------------------

def parse_label(raw_label: str) -> str:
    """Strip numeric prefix (e.g. '1234_keanu reeves' → 'keanu reeves')."""
    if not raw_label:
        return ""
    m = re.match(r"^\d+_(.+)$", raw_label)
    if m:
        return m.group(1).strip()
    return raw_label.strip()

def lookup_gender(label: str, gender_map: dict) -> str:
    clean = label.strip()
    if clean in gender_map:
        return gender_map[clean]
    for key in gender_map:
        if clean.lower() == key.lower():
            return gender_map[key]
    return "unknown"


# -------------------------------
# Embeddings Loader
# -------------------------------

def load_embeddings_from_json(path: Path):
    emb_map = {}
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            for rec in data:
                if "img_embs" in rec and "gt" in rec["img_embs"]:
                    emb_map[rec["id"]] = np.array(rec["img_embs"]["gt"], dtype=np.float32)
        else:
            for line in f:
                rec = json.loads(line)
                if "img_embs" in rec and "gt" in rec["img_embs"]:
                    emb_map[rec["id"]] = np.array(rec["img_embs"]["gt"], dtype=np.float32)
    return emb_map


# -------------------------------
# Precompute Similarity Matrix
# -------------------------------

def build_similarity_cache(celebs, emb_map):
    all_ids = [cid for cid in celebs if cid in emb_map]
    emb_matrix = np.stack([emb_map[cid] for cid in all_ids])
    normed = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    sim_matrix = np.dot(normed, normed.T)
    genders = [celebs[cid]["gender"] for cid in all_ids]
    return all_ids, genders, sim_matrix

def pick_nn_ids_fast(target_id, all_ids, genders, sim_matrix):
    idx = all_ids.index(target_id)
    sims = sim_matrix[idx]

    male_best, female_best = None, None
    male_sim, female_sim = -1, -1
    for j, cid in enumerate(all_ids):
        if j == idx:
            continue
        if genders[j] == "male" and sims[j] > male_sim:
            male_best, male_sim = cid, sims[j]
        if genders[j] == "female" and sims[j] > female_sim:
            female_best, female_sim = cid, sims[j]

    return male_best, female_best


# -------------------------------
# Per-celeb processing
# -------------------------------

def process_celeb(cid, c, words, celeb_gt_b64, all_ids, genders, sim_matrix, master_seed):
    rng = np.random.default_rng(master_seed ^ stable_seed_from_str(cid))
    male_nn_id, female_nn_id = pick_nn_ids_fast(cid, all_ids, genders, sim_matrix)
    male_b64 = celeb_gt_b64.get(male_nn_id)
    female_b64 = celeb_gt_b64.get(female_nn_id)
    if male_b64 is None or female_b64 is None:
        return [], []

    male_nn = {"id": f"{male_nn_id}_gt", "image_b64": male_b64, "tag": "male_nn"}
    female_nn = {"id": f"{female_nn_id}_gt", "image_b64": female_b64, "tag": "female_nn"}

    word_item = rng.choice(words)

    task1_records = []
    task2_records = []

    # ---- Task 1: Image Matching ----
    variants = [("gt", c["gt"])]
    variants += [(f"aug{i}", img) for i, img in enumerate(c["augs"])]
    variants += [(f"illus{i}", img) for i, img in enumerate(c["ills"])]

    for tag, img_b64 in variants:
        options = [
            {"id": f"{cid}_{tag}", "image_b64": img_b64, "tag": "correct"},
            word_item,
            male_nn,
            female_nn,
        ]
        rng.shuffle(options)
        answer_idx = next(i for i, o in enumerate(options) if o["tag"] == "correct")

        rec1 = {
            "task_id": f"im:{cid}:{tag}",
            "support": {"id": f"{cid}_gt", "image_b64": c["gt"], "tag": "GT"},
            "options": options,
            "answer_idx": answer_idx,
        }
        task1_records.append(rec1)

        # ---- Task 2: Grid Pointing ----
        grid_items = [word_item, male_nn, female_nn]
        positions = [(0,0),(0,1),(1,0),(1,1)]
        for row, col in positions:
            distractors = grid_items.copy()
            rng.shuffle(distractors)
            ordered = distractors[:3]
            ordered.insert((row*2+col), {"id": f"{cid}_gt", "image_b64": c["gt"], "tag": "gt"})

            rec2 = {
                "task_id": f"gpg:{cid}:{tag}:{row}{col}",
                "support": {"id": f"{cid}_{tag}", "image_b64": img_b64, "tag": tag},
                "grid_b64": render_grid_b64([g["image_b64"] for g in ordered]),
                "answer_idx": [row, col],
            }
            task2_records.append(rec2)

    return task1_records, task2_records


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celeb_hf", required=True)
    ap.add_argument("--word_hf", required=True)
    ap.add_argument("--celeb_gender", required=True)
    ap.add_argument("--embeddings_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=8)
    args = ap.parse_args()

    with open(args.celeb_gender, "r") as f:
        gender_map = json.load(f)

    celeb_ds = load_from_disk(args.celeb_hf)
    word_ds = load_from_disk(args.word_hf)

    subset_ids = {f"test{i}" for i in range(0, 100)}
    subset_ds = celeb_ds.filter(lambda rec: rec["id"] in subset_ids)

    celebs = {}
    for rec in subset_ds:
        cid = rec["id"]
        parsed_label = parse_label(rec.get("label", ""))
        gender = lookup_gender(parsed_label, gender_map)
        celebs[cid] = {
            "id": cid,
            "gt": rec["gt"],
            "augs": list(rec.get("augmentation").values()),
            "ills": list(rec.get("illusion").values()),
            "gender": gender
        }

    celeb_gt_b64 = {cid: celebs[cid]["gt"] for cid in celebs}

    words = [{"id": f"{rec['id']}_gt", "image_b64": rec["gt"], "tag": "word"} for rec in word_ds]
    emb_map = load_embeddings_from_json(Path(args.embeddings_json))

    all_ids, genders, sim_matrix = build_similarity_cache(celebs, emb_map)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    t1_path = out_dir / "task1_image_matching.jsonl"
    t2_path = out_dir / "task2_gpg.jsonl"

    with open(t1_path, "w") as f1, open(t2_path, "w") as f2:
        results = Parallel(n_jobs=args.n_jobs, backend="threading")(
            delayed(process_celeb)(cid, c, words, celeb_gt_b64, all_ids, genders, sim_matrix, args.seed)
            for cid, c in tqdm(celebs.items(), desc="Celebs")
        )

        for t1, t2 in results:
            if t1: f1.write("\n".join(json.dumps(r) for r in t1) + "\n")
            if t2: f2.write("\n".join(json.dumps(r) for r in t2) + "\n")

    print(f"[✓] Task1 → {t1_path}")
    print(f"[✓] Task2 → {t2_path}")


if __name__=="__main__":
    main()
