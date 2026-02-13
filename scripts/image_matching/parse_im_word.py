import os
import re
import json
import unicodedata
from typing import Dict, Any

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}

# ---------------------- HELPERS ----------------------
def normalize_name(name: str) -> str:
    """Normalize a name/word for consistent dictionary lookup."""
    if "_" in name:  # remove numeric IDs like "1344_"
        name = name.split("_", 1)[-1]
    name = unicodedata.normalize("NFKC", name)
    return name.strip()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_str(s: str) -> str:
    """Remove invalid control characters from a string."""
    return "".join(ch for ch in s if ord(ch) >= 32 or ch in ("\n", "\t"))


def extract_pred_idx(response_str: str, options):
    """Extract final_answer (A/B/C/D) from response and map to option index."""
    match = re.search(r'"final_answer"\s*:\s*"([A-D])"', response_str)
    if match:
        letter = match.group(1)
        pred_idx = LETTER_TO_IDX[letter]
        pred_answer = options[pred_idx] if pred_idx < len(options) else None
        return pred_idx, pred_answer
    return None, None


def parse_support_id(task_id: str) -> str:
    """Extract test* ID from task_id string like 'im:test62:aug1'."""
    if not task_id:
        return None
    parts = task_id.split(":")
    for p in parts:
        if p.startswith("test"):
            return p
    return None


def parse_split(task_id: str) -> str:
    """
    Extract split key exactly as in task_id.
    Returns 'gt', 'aug0'...'aug14', 'illus0'...'illus14', or 'pred'.
    """
    if not task_id:
        return "pred"
    parts = task_id.split(":")
    for p in parts:
        if p == "gt":
            return "gt"
        if p.startswith("aug"):
            return p
        if p.startswith("illus"):
            return p
    return "pred"


# ---------------------- MAIN PARSER ----------------------
def parse_folder(folder: str, dataset="word"):
    parsed = {}

    files = [f for f in os.listdir(folder) if f.endswith(".jsonl")]
    print(f"[INFO] Found {len(files)} JSONL files in {folder}")

    for fname in files:
        base = os.path.splitext(fname)[0]

        # --- Model & Param parsing ---
        parts = base.split("-")

        if base.startswith("qwen2.5-vl-") or base.startswith("qwen2_5-vl-"):
            model = "qwen2.5-vl"
            param = parts[2] if len(parts) > 1 else "unknown"
            if "_im" in param:
                param = param.split("_im")[0]
        elif base.startswith("gemma3-"):
            if len(parts) >= 3:
                model = f"{parts[0]}"   # e.g., gemma3-it
                param = parts[1]        # e.g., 12b
                if "_im" in param:
                    param = param.split("_im")[0]
        elif base.startswith("internvl3_5-") or base.startswith("internvl3.5-"):
            model = "internvl3.5"
            if "thinking_im" in parts:
                model = f"{model}-thinking"
                param = next((p for p in parts if p.endswith("b")), "unknown")
            else:
                param = parts[1] if len(parts) > 1 else "unknown"
                if "_im" in param:
                    param = param.split("_im")[0]

        # --- Load predictions ---
        data = load_jsonl(os.path.join(folder, fname))

        if model not in parsed:
            parsed[model] = {}
        if param not in parsed[model]:
            parsed[model][param] = {}
        if dataset not in parsed[model][param]:
            parsed[model][param][dataset] = {}

        # --- Iterate through tests ---
        for ex in data:
            task_id = ex.get("task_id", "")
            option_ids = ex.get("option_ids", [])
            answer_idx = ex.get("answer_idx", -1)

            # --- Extract test key and split ---
            test_key = parse_support_id(task_id)
            if not test_key:
                continue
            split = parse_split(task_id)

            # --- Extract predicted index from response ---
            pred_idx, pred_answer = extract_pred_idx(
                ex.get("response", ""), option_ids
            )

            # --- Ensure entry exists ---
            entry = parsed[model][param][dataset].setdefault(test_key, {})

            # --- Store under correct split key ---
            if split == "gt":
                entry["gt"] = {
                    "task_id": task_id,
                    "option_ids": option_ids,
                    "answer_idx": answer_idx,
                    "pred_idx": pred_idx,
                    "pred_answer": pred_answer,
                    "model": ex.get("model"),
                }
            else:
                entry[split] = {
                    "task_id": task_id,
                    "option_ids": option_ids,
                    "answer_idx": answer_idx,
                    "pred_idx": pred_idx,
                    "pred_answer": pred_answer,
                    "model": ex.get("model"),
                }

    return parsed


def main():
    dataset = "word"
    folder = "./results_im/" + dataset
    output_folder = "./results"
    os.makedirs(output_folder, exist_ok=True)

    parsed = parse_folder(folder, dataset=dataset)

    out_path = os.path.join(output_folder, "all_parsed_word_im.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    print(f"[INFO] Saved parsed MCQ results to {out_path}")


if __name__ == "__main__":
    main()
