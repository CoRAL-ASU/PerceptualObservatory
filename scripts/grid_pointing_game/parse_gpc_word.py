import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

CELL_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


# ---------------------- IO ----------------------

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------- ANSWER PARSING ----------------------

def decode_answer_idx(ans: Any) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(ans, (list, tuple)) and len(ans) == 2:
        return int(ans[0]), int(ans[1])
    if isinstance(ans, int) and 0 <= ans <= 3:
        return ans // 2, ans % 2
    return None, None


def parse_final_answer(response: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extracts the final [row, col] answer from model response text.
    Expected pattern: "final_answer": "[r, c]".
    Returns (None, None) on failure.
    """
    if response is None:
        return None, None
    m = re.search(r'"final_answer"\s*:\s*"\[\s*(\d)\s*,\s*(\d)\s*\]"', str(response))
    if m is None:
        return None, None
    return int(m.group(1)), int(m.group(2))


# ---------------------- TASK-ID HELPERS ----------------------

def parse_split(task_id: str) -> str:
    parts = (task_id or "").split(":")
    for p in parts:
        if p == "gt":
            return "gt"
        if p.startswith("aug"):
            return p
        if p.startswith("illus"):
            return p
    return "pred"


def parse_support(task_id: str) -> Optional[str]:
    parts = (task_id or "").split(":")
    for p in parts:
        if p.startswith("test"):
            return p
    return None


# ---------------------- MODEL NAME ----------------------

def parse_model_name(source_file: str) -> str:
    base = os.path.basename(source_file)
    name = base.replace("_gpg-word.jsonl", "").replace("_gpg-word.json", "")
    return name


# ---------------------- RECORD PARSING ----------------------

def parse_record(
    rec: Dict[str, Any],
    source_file: str,
) -> Dict[str, Any]:

    task_id = rec.get("task_id", "")
    split = parse_split(task_id)
    support = parse_support(task_id)

    # ---- grid answers ----
    gt_row, gt_col = decode_answer_idx(rec.get("answer_idx"))
    pred_row, pred_col = parse_final_answer(rec.get("response"))

    # ---- model ----
    model_name = rec.get("model")
    if "thinking" in parse_model_name(source_file) and model_name and "-thinking" not in model_name:
        model_name += "-thinking"

    return {
        "task_id": task_id,
        "support": support,
        "split": split,
        "gt_row": gt_row,
        "gt_col": gt_col,
        "pred_row": pred_row,
        "pred_col": pred_col,
        "model": model_name,
        "source_file": os.path.basename(source_file),
    }


def parse_folder(folder: str) -> List[Dict[str, Any]]:
    files = glob.glob(os.path.join(folder, "*.jsonl"))
    parsed: List[Dict[str, Any]] = []
    for f in files:
        for rec in load_jsonl(f):
            parsed.append(parse_record(rec, f))
    return parsed


# ---------------------- MAIN ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="./results_gpc/word/", help="Folder containing *.jsonl result files")
    ap.add_argument("--out-json", default="./results/all_parsed_word_gpc.json")
    args = ap.parse_args()

    parsed = parse_folder(args.folder)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    print(f"✅ Parsed {len(parsed)} records from {args.folder}")
    print(f"📄 JSON saved to {args.out_json}")


if __name__ == "__main__":
    main()
