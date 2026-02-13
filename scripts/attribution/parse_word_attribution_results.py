import json
import re
import os
import numpy as np
from datasets import load_from_disk
from typing import Dict, Any, List, Set


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


def normalize_bbox_dict(bbox_dict: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Normalize GT bboxes into { 'sequence': [x1, y1, x2, y2] } (word dataset).
    Accepts top-level coords or {feat: coords} or {feat: [x1,y1,x2,y2]}.
    """
    # Case 1: directly contains coords
    if isinstance(bbox_dict, dict) and all(k in bbox_dict for k in ["x1", "y1", "x2", "y2"]):
        return {"sequence": [float(bbox_dict["x1"]), float(bbox_dict["y1"]),
                             float(bbox_dict["x2"]), float(bbox_dict["y2"])]}

    # Case 2: dict of dicts/lists
    out = {}
    if isinstance(bbox_dict, dict):
        for k, v in bbox_dict.items():
            if isinstance(v, dict) and all(coord in v for coord in ["x1", "y1", "x2", "y2"]):
                out["sequence"] = [float(v["x1"]), float(v["y1"]), float(v["x2"]), float(v["y2"])]
                break
            elif isinstance(v, (list, tuple)) and len(v) == 4:
                out["sequence"] = [float(x) for x in v]
                break
    return out


# ---------------------- ROBUST PRED PARSER ----------------------
def ensure_response_dict(resp: Any) -> Dict[str, List[float]]:
    """
    Extract a single canonical feature for word dataset:
      returns {'sequence': [x1, y1, x2, y2]}
    Robust to:
      - markdown fences (``` / ```json)
      - extra reasoning text
      - JSON array or dict
      - coords at top-level or nested
    Always uses the key 'sequence' to match GT.
    """
    candidate = None

    # 1) If already a dict/list, use directly
    if isinstance(resp, (dict, list)):
        candidate = resp
    elif isinstance(resp, str):
        txt = resp.strip()

        # Strip opening/closing fences anywhere in the string
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\s*```$", "", txt).strip()

        # Try full parse first
        try:
            candidate = json.loads(txt)
        except Exception:
            # Prefer arrays first, then dicts
            m = re.search(r"\[[\s\S]*\]", txt) or re.search(r"\{[\s\S]*\}", txt)
            if not m:
                return {}
            try:
                candidate = json.loads(m.group(0))
            except Exception:
                return {}
    else:
        return {}

    # 2) If list, take the first entry that yields coords
    if isinstance(candidate, list):
        for entry in candidate:
            if not isinstance(entry, dict):
                continue
            # top-level coords in an entry
            if all(k in entry for k in ["x1", "y1", "x2", "y2"]):
                return {"sequence": [float(entry["x1"]), float(entry["y1"]),
                                     float(entry["x2"]), float(entry["y2"])]}
            # nested coords inside entry
            for _, v in entry.items():
                if isinstance(v, dict) and all(k in v for k in ["x1", "y1", "x2", "y2"]):
                    return {"sequence": [float(v["x1"]), float(v["y1"]),
                                         float(v["x2"]), float(v["y2"])]}
        return {}

    # 3) If dict
    if isinstance(candidate, dict):
        # top-level coords
        if all(k in candidate for k in ["x1", "y1", "x2", "y2"]):
            return {"sequence": [float(candidate["x1"]), float(candidate["y1"]),
                                 float(candidate["x2"]), float(candidate["y2"])]}
        # nested coords
        for _, v in candidate.items():
            if isinstance(v, dict) and all(k in v for k in ["x1", "y1", "x2", "y2"]):
                return {"sequence": [float(v["x1"]), float(v["y1"]),
                                     float(v["x2"]), float(v["y2"])]}
    return {}


# ---------------------- EVAL HELPERS ----------------------
def evaluate(pred_data, gt_map):
    results = []
    for item in pred_data:
        entry_id = item["id"]
        mapping = parse_id(entry_id)
        support_id = mapping.get("support")
        if support_id and support_id.endswith("_gt"):
            support_id = support_id[:-3]
        if not support_id or support_id not in gt_map:
            continue

        gt_item = gt_map[support_id]                 # {'response': {'sequence': [...]}, 'label': ...}
        pred_resp = ensure_response_dict(item["response"])  # {'sequence': [...]}

        feat_logs = {}
        iou_only = {}
        all_ious = []

        # For word dataset we only care about 'sequence'
        if "sequence" in pred_resp and "sequence" in gt_item["response"]:
            this_iou = iou(pred_resp["sequence"], gt_item["response"]["sequence"])
            feat_logs["sequence"] = {
                "pred_coords": pred_resp["sequence"],
                "gt_coords": gt_item["response"]["sequence"],
                "iou": this_iou,
            }
            iou_only["sequence"] = this_iou
            all_ious.append(this_iou)
        else:
            # No parsed coords -> count as failure (IoU = 0)
            iou_only["sequence"] = 0.0
            feat_logs["sequence"] = {
                "pred_coords": pred_resp.get("sequence", None),
                "gt_coords": gt_item["response"].get("sequence", None),
                "iou": 0.0,
            }

        avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

        results.append({
            "id": entry_id,
            "support": support_id,
            "label": gt_item["label"],
            "features": feat_logs,
            "ious": iou_only,
            "avg_iou": avg_iou
        })
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


def build_gt_map(hf_ds, needed_ids: Set[str]) -> Dict[str, Any]:
    sub_ds = hf_ds.filter(lambda x: x["id"] in needed_ids)
    return {
        rec["id"]: {
            "response": normalize_bbox_dict(rec["bbox"]),  # -> {'sequence': [x1,y1,x2,y2]}
            "label": rec["label"],
        }
        for rec in sub_ds
    }


# ---------------------- MAIN ----------------------
def main():
    Dataset = "word"
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

        # --- Generalized model/param parsing ---
        model, param = "unknown", "unknown"
        parts = model_and_size.split("-")

        if model_and_size.startswith("qwen2_5vl-"):
            model = "qwen2_5vl"
            param = parts[1] if len(parts) > 1 else "unknown"
        elif model_and_size.startswith("gemma3-"):
            if len(parts) >= 3:
                model = f"{parts[0]}-{parts[2]}"   # e.g., "gemma3-it"
                param = parts[1]                   # e.g., "12b"
        elif model_and_size.startswith("intervl3_5-"):
            # Handle intervl3_5 specifically
            base_model = "intervl3_5"
            # check if "thinking" appears
            if "thinking" in parts:
                model = f"{base_model}-thinking"
                # param will be the part that ends with 'b' (e.g., "8b" or "14b")
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
        gt_map = build_gt_map(hf_ds, needed_ids)
        results = evaluate(preds, gt_map)

        # insert into parsed structure
        parsed.setdefault(model, {}).setdefault(param, {}).setdefault(dataset, {})

        for r in results:
            test_key = f"test_{r['support']}"
            if test_key not in parsed[model][param][dataset]:
                parsed[model][param][dataset][test_key] = {"label": r["label"]}

            if split == "gt":
                parsed[model][param][dataset][test_key]["gt"] = {
                    **r["ious"],
                    "avg": r["avg_iou"],
                    "features": r["features"]
                }
            else:
                existing = [k for k in parsed[model][param][dataset][test_key] if k.startswith(split)]
                split_key = f"{split}{len(existing)}"
                parsed[model][param][dataset][test_key][split_key] = {
                    **r["ious"],
                    "avg": r["avg_iou"],
                    "features": r["features"]
                }

    out_path = os.path.join(output_folder, "all_parsed_word_p.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    print(f"[INFO] Saved parsed results to {out_path}")


if __name__ == "__main__":
    main()
