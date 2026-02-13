import json
import re
import os
import unicodedata
import numpy as np
from datasets import load_from_disk
from typing import Dict, Any, List, Set
from scipy.optimize import linear_sum_assignment


# ---------------------- LOAD CELEB GENDER MAP ----------------------
with open("celeb_gender.json", "r", encoding="utf-8") as f:
    CELEB_GENDER = json.load(f)


def normalize_name(name: str) -> str:
    """Normalize a celeb name for consistent dictionary lookup."""
    if "_" in name:  # remove numeric IDs like "1344_"
        name = name.split("_", 1)[-1]
    name = unicodedata.normalize("NFKC", name)
    return name.strip()


def lookup_gender(label: str) -> str:
    """Look up gender from celeb_gender.json (default 'unknown')."""
    clean = normalize_name(label)
    if clean in CELEB_GENDER:
        return CELEB_GENDER[clean]
    for key in CELEB_GENDER:
        if clean.lower() == key.lower():
            return CELEB_GENDER[key]
    return "unknown"


# ---------------------- IO HELPERS ----------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_id(entry_id: str) -> Dict[str, str]:
    parts = entry_id.split("__")
    out = {}
    for i, p in enumerate(parts):
        if p == "support" and i + 1 < len(parts):
            out["support"] = parts[i + 1]
        elif p == "query" and i + 1 < len(parts):
            out["query"] = parts[i + 1]
    return out


# ---------------------- BBOX UTILS ----------------------
def iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)


def normalize_bbox_dict(bbox_dict: Dict[str, Dict[str, int]]) -> Dict[str, List[int]]:
    return {k: [v["x1"], v["y1"], v["x2"], v["y2"]] for k, v in bbox_dict.items()}


def ensure_response_dict(resp):
    """
    Extract {feature: [x1, y1, x2, y2]} from model response.
    Robust to extra text, reasoning, and markdown fences.
    """
    if isinstance(resp, dict):
        work = resp
    elif isinstance(resp, str):
        txt = resp.strip()
        match = re.search(r"(\{.*\}|\[.*\])", txt, flags=re.DOTALL)
        if not match:
            return {}
        candidate = match.group(1)
        try:
            work = json.loads(candidate)
        except Exception:
            return {}
    else:
        return {}

    if isinstance(work, list):
        merged = {}
        for entry in work:
            if isinstance(entry, dict):
                merged.update(entry)
        work = merged

    if not isinstance(work, dict):
        return {}

    cleaned = {}
    for feat, val in work.items():
        if isinstance(val, dict):
            try:
                coords = [float(val[k]) for k in ["x1", "y1", "x2", "y2"]]
                cleaned[feat] = coords
            except Exception:
                continue
    return cleaned


# ---------------------- THINK LENGTH ----------------------
def extract_think_length(resp: str) -> int:
    """Extract length of <think> block (in tokens)."""
    if not isinstance(resp, str):
        return 0
    match = re.search(r"<think>(.*?)</think>", resp, flags=re.DOTALL)
    if not match:
        return 0
    think_text = match.group(1).strip()
    return len(think_text.split())  # simple whitespace tokenization


# ---------------------- EVAL HELPERS ----------------------
def evaluate(pred_data, gt_map, skip_gender=False):
    results = []
    for item in pred_data:
        entry_id = item["id"]
        mapping = parse_id(entry_id)
        support_id = mapping.get("support")
        if support_id and support_id.endswith("_gt"):
            support_id = support_id[:-3]
        if not support_id or support_id not in gt_map:
            continue

        gt_item = gt_map[support_id]
        pred_resp = ensure_response_dict(item["response"])
        think_len = extract_think_length(item["response"])

        feat_logs = {}
        iou_only = {}
        all_ious = []

        # Nose & mouth
        for feat in ["nose", "mouth"]:
            if feat in pred_resp and feat in gt_item["response"]:
                this_iou = iou(pred_resp[feat], gt_item["response"][feat])
                feat_logs[feat] = {
                    "pred_coords": pred_resp[feat],
                    "gt_coords": gt_item["response"][feat],
                    "iou": this_iou,
                }
                iou_only[feat] = this_iou
                all_ious.append(this_iou)

        # Eyes with optimal assignment
        if all(f in pred_resp for f in ["left_eye", "right_eye"]) and all(
            f in gt_item["response"] for f in ["left_eye", "right_eye"]
        ):
            iou_matrix = np.array([
                [
                    iou(pred_resp["left_eye"], gt_item["response"]["left_eye"]),
                    iou(pred_resp["left_eye"], gt_item["response"]["right_eye"]),
                ],
                [
                    iou(pred_resp["right_eye"], gt_item["response"]["left_eye"]),
                    iou(pred_resp["right_eye"], gt_item["response"]["right_eye"]),
                ]
            ])
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                pred_key = "left_eye" if r == 0 else "right_eye"
                gt_key = "left_eye" if c == 0 else "right_eye"
                this_iou = iou_matrix[r, c]
                feat_logs[pred_key] = {
                    "pred_coords": pred_resp[pred_key],
                    "gt_coords": gt_item["response"][gt_key],
                    "iou": this_iou,
                }
                iou_only[pred_key] = this_iou
                all_ious.append(this_iou)

        avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

        record = {
            "id": entry_id,
            "support": support_id,
            "label": gt_item["label"],
            "features": feat_logs,
            "ious": iou_only,
            "avg_iou": avg_iou,
            "think_len": think_len,
        }
        if not skip_gender:
            record["gender"] = gt_item["gender"]

        results.append(record)
    return results


def collect_needed_ids(*pred_sets) -> Set[str]:
    needed_ids = set()
    for preds in pred_sets:
        for item in preds:
            mapping = parse_id(item["id"])
            support_id = mapping.get("support")
            if support_id and support_id.endswith("_gt"):
                support_id = support_id[:-3]
            if support_id:
                needed_ids.add(support_id)
    return needed_ids


def build_gt_map(hf_ds, needed_ids: Set[str], skip_gender=False) -> Dict[str, Any]:
    sub_ds = hf_ds.filter(lambda x: x["id"] in needed_ids)
    gt_map = {}
    for rec in sub_ds:
        entry = {
            "response": normalize_bbox_dict(rec["bbox"]),
            "label": rec["label"],
        }
        if not skip_gender:
            entry["gender"] = lookup_gender(rec["label"])
        gt_map[rec["id"]] = entry
    return gt_map


# ---------------------- MAIN ----------------------
def main():
    Dataset = "celeb"
    folder = "./results_attribution_P/attribution-partial/" + Dataset
    output_folder = "./results"
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(folder) if f.endswith(".jsonl")]
    hf_ds = load_from_disk(f"/gscratch/tkishore/hf_{Dataset}_dataset")

    parsed = {}

    for fname in files:
        base = os.path.splitext(fname)[0]

        if "_q3_" in base:
            model_and_size, rest = base.split("_q3_", 1)
            if "-" in rest:
                dataset, split = rest.split("-", 1)
            else:
                dataset, split = rest, "unknown"
        else:
            model_and_size, dataset, split = base, "unknown", "unknown"

        skip_gender = (dataset == "word")

        model, param = "unknown", "unknown"
        parts = model_and_size.split("-")

        if model_and_size.startswith("qwen2_5vl-"):
            model = "qwen2_5vl"
            param = parts[1] if len(parts) > 1 else "unknown"
        elif model_and_size.startswith("gemma3-"):
            if len(parts) >= 3:
                model = f"{parts[0]}-{parts[2]}"
                param = parts[1]
        elif model_and_size.startswith("intervl3_5-"):
            base_model = "intervl3_5"
            if "thinking" in parts:
                model = f"{base_model}-thinking"
                param = next((p for p in parts if p.endswith("b")), "unknown")
            else:
                model = base_model
                param = parts[1] if len(parts) > 1 else "unknown"
        else:
            if len(parts) >= 2:
                param = parts[1]
                model = "-".join([parts[0]] + parts[2:]) if len(parts) > 2 else parts[0]

        preds = load_jsonl(os.path.join(folder, fname))
        needed_ids = collect_needed_ids(preds)
        gt_map = build_gt_map(hf_ds, needed_ids, skip_gender=skip_gender)
        results = evaluate(preds, gt_map, skip_gender=skip_gender)

        if model not in parsed:
            parsed[model] = {}
        if param not in parsed[model]:
            parsed[model][param] = {}
        if dataset not in parsed[model][param]:
            parsed[model][param][dataset] = {}

        for r in results:
            test_key = f"test_{r['support']}"
            if test_key not in parsed[model][param][dataset]:
                parsed[model][param][dataset][test_key] = {
                    "label": r["label"]
                }
                if not skip_gender:
                    parsed[model][param][dataset][test_key]["gender"] = r["gender"]

            if split == "gt":
                parsed[model][param][dataset][test_key]["gt"] = {
                    **r["ious"],
                    "avg": r["avg_iou"],
                    "features": r["features"],
                    "think_len": r["think_len"]
                }
            else:
                existing = [k for k in parsed[model][param][dataset][test_key] if k.startswith(split)]
                split_key = f"{split}{len(existing)}"
                parsed[model][param][dataset][test_key][split_key] = {
                    **r["ious"],
                    "avg": r["avg_iou"],
                    "features": r["features"],
                    "think_len": r["think_len"]
                }

    out_path = os.path.join(output_folder, "all_parsed_celeb_p.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    print(f"[INFO] Saved parsed results to {out_path}")


if __name__ == "__main__":
    main()
