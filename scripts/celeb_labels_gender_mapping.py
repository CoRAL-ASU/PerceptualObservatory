import os
import json
import re
import unicodedata
from datasets import load_from_disk

# ---------------------- LOAD CELEB GENDER MAP ----------------------
with open("celeb_gender.json", "r", encoding="utf-8") as f:
    CELEB_GENDER = json.load(f)


def normalize_name(name: str) -> str:
    """Normalize a celeb name for consistent dictionary lookup (removes ID prefix)."""
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


def build_cache(hf_ds, out_path, max_tests=100):
    """
    Build cache mapping test0...testN → celeb raw label, parsed label, and gender.
    """
    cache = {}
    for rec in hf_ds:
        rec_id = rec["id"]
        if not rec_id.startswith("test"):
            continue

        match = re.match(r"(test\d+)", rec_id)
        if not match:
            continue

        test_key = match.group(1)  # "test0"
        if int(test_key.replace("test", "")) > max_tests:
            continue

        raw_label = rec["label"]  # e.g. "8771_John Oliver"
        parsed_label = normalize_name(raw_label)  # e.g. "John Oliver"
        gender = lookup_gender(raw_label)

        cache[test_key] = {
            "label": raw_label,
            "parsed_label": parsed_label,
            "gender": gender
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    print(f"[INFO] Saved cache with {len(cache)} entries → {out_path}")


def main():
    dataset = "celeb"
    hf_ds = load_from_disk(f"/gscratch/tkishore/hf_{dataset}_dataset")

    os.makedirs("./results", exist_ok=True)
    out_path = f"./results/{dataset}_cache.json"

    build_cache(hf_ds, out_path, max_tests=100)


if __name__ == "__main__":
    main()
