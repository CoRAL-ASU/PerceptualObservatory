import argparse
import json
import base64, io
import hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from joblib import Parallel, delayed
from svgwrite import Drawing
from PIL import Image
import cairosvg

# -------------------------------
# Helpers
# -------------------------------

def stable_seed_from_str(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")

def b64_to_image(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def render_grid_b64(grid_b64s):
    imgs = [b64_to_image(b64) for b64 in grid_b64s]
    w, h = imgs[0].size
    canvas = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
    canvas.paste(imgs[0], (0, 0))
    canvas.paste(imgs[1], (w, 0))
    canvas.paste(imgs[2], (0, h))
    canvas.paste(imgs[3], (w, h))
    canvas = canvas.resize((w, h), Image.Resampling.LANCZOS)
    return image_to_b64(canvas)

@lru_cache(maxsize=20000)
def render_word_to_b64(text, font="Verdana", case="lower", position="center"):
    if case == "upper":
        text = text.upper()
    elif case == "camel":
        text = "".join(word.capitalize() for word in text.split())
    else:
        text = text.lower()

    dwg = Drawing(size=("1024px", "1024px"))
    dwg.attribs["style"] = "background-color:white"
    dwg.add(dwg.rect(insert=(0, 0), size=("1024px", "1024px"),
                     fill="white", stroke="black", stroke_width=4))

    pos_map = {"top": ("50%", "10%"),
               "center": ("50%", "50%"),
               "bottom": ("50%", "90%")}
    x, y = pos_map.get(position, ("50%", "50%"))

    dwg.add(dwg.text(
        text,
        insert=(x, y),
        text_anchor="middle",
        alignment_baseline="middle",
        font_size=250,
        font_family=font,
        fill="black",
        stroke="black",
        stroke_width=3
    ))

    png_bytes = cairosvg.svg2png(bytestring=dwg.tostring().encode("utf-8"))
    return base64.b64encode(png_bytes).decode("utf-8")


# -------------------------------
# Per-word worker
# -------------------------------

def process_word(wid, w, celebs, rng):
    celeb_id, celeb = rng.choice(list(celebs.items()))
    celeb_item = {"id": f"{celeb_id}_gt", "image_b64": celeb["gt"], "tag": "ood_celeb"}

    # render NN1/NN2 with cache
    nn1_b64 = render_word_to_b64(w["label"]["nn1"], w["label"]["font"],
                                 w["label"]["case"], w["label"]["position"])
    nn2_b64 = render_word_to_b64(w["label"]["nn2"], w["label"]["font"],
                                 w["label"]["case"], w["label"]["position"])

    nn1_item = {"id": f"{wid}_nn1", "image_b64": nn1_b64, "tag": "nn1"}
    nn2_item = {"id": f"{wid}_nn2", "image_b64": nn2_b64, "tag": "nn2"}

    records_t1, records_t2 = [], []


    # ---- Task 1 ----
    variants = [("gt", w["gt"])]
    variants += [(f"aug{i}", img) for i, img in enumerate(w["augs"])]
    variants += [(f"illus{i}", img) for i, img in enumerate(w["ills"])]

    for tag, img_b64 in variants:
        support = {"id": f"{wid}_gt", "image_b64": w["gt"], "tag": "GT"}

        options = [
            {"id": f"{wid}_{tag}", "image_b64": img_b64, "tag": "correct"},
            nn1_item,
            nn2_item,
            celeb_item
        ]
        rng.shuffle(options)
        answer_idx = next(i for i, o in enumerate(options) if o["tag"] == "correct")

        rec1 = {
            "task_id": f"im:{wid}:{tag}",
            "support": support,
            "options": options,
            "answer_idx": answer_idx
        }
        records_t1.append(json.dumps(rec1))


    # ---- Task 2 ----
    variants = [("gt", w["gt"])]
    variants += [(f"aug{i}", img) for i, img in enumerate(w["augs"])]
    variants += [(f"illus{i}", img) for i, img in enumerate(w["ills"])]

    for tag, img_b64 in variants:
        positions = [(0,0),(0,1),(1,0),(1,1)]
        for row, col in positions:
            distractors = [nn1_item, nn2_item, celeb_item]
            rng.shuffle(distractors)
            ordered = distractors[:3]
            ordered.insert(row*2+col, {"id": f"{wid}_gt", "image_b64": w["gt"], "tag": "gt"})

            rec2 = {
                "task_id": f"gpg:{wid}:{tag}:{row}{col}",
                "support": {"id": f"{wid}_{tag}", "image_b64": img_b64, "tag": tag},
                "grid_b64": render_grid_b64([g["image_b64"] for g in ordered]),
                "answer_idx": [row, col]
            }
            records_t2.append(json.dumps(rec2))

    return records_t1, records_t2


# -------------------------------
# Task builder
# -------------------------------

def build_tasks(words, celebs, out_dir, seed=42, n_jobs=16):
    out_dir.mkdir(parents=True, exist_ok=True)
    t1_path = out_dir / "task1_image_matching.jsonl"
    t2_path = out_dir / "task2_gpg.jsonl"

    rng = np.random.default_rng(seed)

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_word)(wid, w, celebs, rng) for wid, w in tqdm(words.items(), desc="Words")
    )

    with open(t1_path, "w") as f1, open(t2_path, "w") as f2:
        for rec1s, rec2s in results:
            if rec1s: f1.write("\n".join(rec1s) + "\n")
            if rec2s: f2.write("\n".join(rec2s) + "\n")

    print(f"[✓] Task1 → {t1_path}")
    print(f"[✓] Task2 → {t2_path}")


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word_hf", required=True)
    ap.add_argument("--celeb_hf", required=True)
    ap.add_argument("--word_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=16)
    args = ap.parse_args()

    word_ds = load_from_disk(args.word_hf)
    celeb_ds = load_from_disk(args.celeb_hf)

    with open(args.word_json, "r") as f:
        label_map = json.load(f)

    words = {}
    for rec in word_ds:
        wid = rec["id"]
        if wid not in label_map:
            continue
        words[wid] = {
            "id": wid,
            "gt": rec["gt"],
            "augs": list(rec.get("augmentation", {}).values()),
            "ills": list(rec.get("illusion", {}).values()),
            "label": label_map[wid]["label"]
        }

    celebs = {rec["id"]: {"gt": rec["gt"]} for rec in celeb_ds}

    build_tasks(words, celebs, Path(args.out_dir), seed=args.seed, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
