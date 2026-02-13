import argparse, json, os, math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import re

# ---------------------- HELPERS ----------------------

def load_parsed(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cell_id(r, c):
    if r is None or c is None: return None
    if r < 0 or c < 0: return None
    return r*2 + c

def accuracy(y_true, y_pred):
    return (sum(int(a==b) for a,b in zip(y_true,y_pred)) / len(y_true)) if y_true else 0.0

def manhattan_cell_dist(a, b):
    if a is None or b is None: return None
    ar, ac = divmod(a, 2)
    br, bc = divmod(b, 2)
    return abs(ar - br) + abs(ac - bc)

def wilson_ci(k, n, alpha=0.05):
    if n == 0: return (0.0, 0.0)
    z = 1.96
    phat = k / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    margin = (z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

# ---------------------- CORE STATS ----------------------

def per_support_stats(records):
    gt_cells=[cell_id(e["gt_row"],e["gt_col"]) for e in records]
    pr_cells=[cell_id(e["pred_row"],e["pred_col"]) for e in records]

    # safe "no answer"
    no_answer_mask = [
        (e["pred_row"] is None or e["pred_col"] is None or e["pred_row"] < 0 or e["pred_col"] < 0)
        for e in records
    ]
    no_answer_rate = float(sum(no_answer_mask)) / len(records) if records else 0.0

    pr_valid = [p for p in pr_cells if p is not None]
    fixation = int(len(set(pr_valid)) == 1) if pr_valid else 0

    acc_by_pos = {}
    for pos in [0,1,2,3]:
        idx_pos = [i for i,g in enumerate(gt_cells) if g == pos]
        if not idx_pos:
            acc_by_pos[pos] = None
            continue
        valid_idx = [i for i in idx_pos if pr_cells[i] is not None]
        acc_by_pos[pos] = accuracy([gt_cells[i] for i in valid_idx],
                                   [pr_cells[i] for i in valid_idx]) if valid_idx else 0.0
    return fixation, acc_by_pos, no_answer_rate

def compute_distance_errors(records):
    dists = []
    for ex in records:
        gt = cell_id(ex["gt_row"], ex["gt_col"])
        pr = cell_id(ex["pred_row"], ex["pred_col"])
        if gt is None or pr is None: continue
        if pr != gt:
            d = manhattan_cell_dist(pr, gt)
            if d is not None:
                dists.append(d)
    mean_dist = float(np.mean(dists)) if dists else 0.0
    return mean_dist, dists

def compute_model_stats(model_records):
    buckets=defaultdict(list)
    for ex in model_records:
        key=(ex["support"], ex["split"])
        buckets[key].append(ex)

    split_results=defaultdict(list)
    fixation_counts=defaultdict(list)
    split_example_acc = defaultdict(list)
    split_error_dists = defaultdict(list)

    for (support,split),recs in buckets.items():
        if len(recs) < 4: continue
        fixation, acc_by_pos, no_ans = per_support_stats(recs)
        fixation_counts[split].append(fixation)
        split_results[split].append(acc_by_pos)

        for e in recs:
            gt = cell_id(e["gt_row"], e["gt_col"])
            pr = cell_id(e["pred_row"], e["pred_col"])
            if gt is None: continue
            correct = int(pr == gt) if pr is not None else 0
            split_example_acc[split].append(correct)
        _, dists = compute_distance_errors(recs)
        split_error_dists[split].extend(dists)

    stats={}
    for split,lst in split_results.items():
        all_accs={pos:[d[pos] for d in lst if d[pos] is not None] for pos in [0,1,2,3]}
        mean_accs={pos: float(np.mean(v)) if v else 0.0 for pos,v in all_accs.items()}
        sensitivity = max(mean_accs.values()) - min(mean_accs.values()) if mean_accs else 0.0

        ex_accs = split_example_acc[split]
        acc = float(np.mean(ex_accs)) if ex_accs else 0.0
        acc_ci = wilson_ci(int(acc*len(ex_accs)), len(ex_accs)) if ex_accs else (0.0, 0.0)

        dists = split_error_dists[split]
        mean_dist = float(np.mean(dists)) if dists else 0.0

        stats[split]={
            "per_pos_acc": mean_accs,
            "pos_gap": sensitivity,
            "fixation_rate": float(np.mean(fixation_counts[split])) if fixation_counts[split] else 0.0,
            "example_accuracy": acc,
            "example_accuracy_ci": [acc_ci[0], acc_ci[1]],
            "mean_manhattan_error": mean_dist,
            "n_examples": len(split_example_acc[split])
        }

    return stats

def group_by_model(data):
    models=defaultdict(list)
    for ex in data:
        models[ex["model"]].append(ex)
    return models

# ---------------------- SUMMARY ----------------------

def build_rich_summary(results):
    summary = {}
    for model, model_stats in results.items():
        model_key = model.split("/")[-1] if "/" in model else model
        entry = {}
        all_stats = model_stats["all"]

        # --- GT ---
        if "gt" in all_stats:
            gt_stats = all_stats["gt"]
            entry["gt"] = {
                "accuracy": gt_stats.get("example_accuracy", 0.0),
                "per_pos_acc": gt_stats.get("per_pos_acc", {}),
                "pos_gap": gt_stats.get("pos_gap", 0.0),
                "mean_manhattan_error": gt_stats.get("mean_manhattan_error", 0.0),
                "fixation_rate": gt_stats.get("fixation_rate", 0.0),
            }

        # --- AUG avg ---
        aug_splits = {k:v for k,v in all_stats.items() if k.startswith("aug")}
        if aug_splits:
            mean_acc = np.mean([v.get("example_accuracy", 0.0) for v in aug_splits.values()])
            mean_gap  = np.mean([v.get("pos_gap", 0.0) for v in aug_splits.values()])
            per_pos = {pos: np.mean([v["per_pos_acc"].get(pos,0.0) for v in aug_splits.values()]) for pos in [0,1,2,3]}
            entry["aug"] = {
                "accuracy": float(mean_acc),
                "per_pos_acc": per_pos,
                "pos_gap": float(mean_gap),
            }

        # --- ILLUS avg ---
        illus_splits = {k:v for k,v in all_stats.items() if k.startswith("illus")}
        if illus_splits:
            mean_acc = np.mean([v.get("example_accuracy", 0.0) for v in illus_splits.values()])
            mean_gap  = np.mean([v.get("pos_gap", 0.0) for v in illus_splits.values()])
            per_pos = {pos: np.mean([v["per_pos_acc"].get(pos,0.0) for v in illus_splits.values()]) for pos in [0,1,2,3]}
            entry["illus"] = {
                "accuracy": float(mean_acc),
                "per_pos_acc": per_pos,
                "pos_gap": float(mean_gap),
            }

        summary[model_key] = entry
    return summary

# ---------------------- PLOTTING ----------------------

FAMILY_COLORS = {
    "InternVL": "tab:blue",
    "Qwen": "tab:orange",
    "Gemma": "tab:purple",
    "LLaVA": "tab:green",
    "MiniGPT": "tab:red",
    "Other": "tab:brown",
}

def get_family(model_name: str) -> str:
    low = model_name.lower()
    if "internvl" in low: return "InternVL"
    elif "qwen" in low: return "Qwen"
    elif "gemma" in low: return "Gemma"
    elif "llava" in low: return "LLaVA"
    elif "minigpt" in low: return "MiniGPT"
    else: return "Other"

def shorten_model_name(name: str) -> str:
    low = name.lower()

    if "gemma" in low:
        name = re.sub(r"gemma[-_]?3[-_]?", "G-", name, flags=re.IGNORECASE)
    if "qwen2.5-vl" in low:
        name = re.sub(r"qwen2\.5[-_]?vl[-_]?", "Q-", name, flags=re.IGNORECASE)
    if "internvl3_5" in low:
        name = re.sub(r"internvl3_5[-_]?", "I-", name, flags=re.IGNORECASE)

    if re.search(r"thinking$", name, flags=re.IGNORECASE):
        name = re.sub(r"-thinking$", "-T", name, flags=re.IGNORECASE)
        return name

    name = re.sub(r"-(instruct|it|hf)$", "", name, flags=re.IGNORECASE)
    return name

def plot_accuracy_vs_gap(points, title, filename):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=160)

    x = np.linspace(0, 1, 400); y = np.linspace(0, 1, 400)
    X, Y = np.meshgrid(x, y); Z = (1 - X) * Y
    ax.contourf(X, Y, Z, levels=100, cmap="Blues", alpha=0.5)
    ax.axhline(0.5, color="white", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1)

    kwargs = dict(ha="center", va="center", fontsize=11, alpha=0.9, color="black")
    
    for model, (gap, acc) in points.items():
        family = get_family(model)
        base_color = FAMILY_COLORS.get(family, "gray")
        ax.scatter(gap, acc, color=base_color, marker="o", s=70,
                   edgecolor="black", linewidth=0.6)

    ax.set_title(title, fontsize=15)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------------------- MAIN ----------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parsed_json", default="./results/gpg/word/all_parsed_word_gpc.json")
    ap.add_argument("--out", default="./results/gpg/word/avg_performance_word_gpc.json")
    args=ap.parse_args()

    data=load_parsed(args.parsed_json)
    models=group_by_model(data)

    results={}
    for model,recs in models.items():
        results[model]={}
        results[model]["all"]=compute_model_stats(recs)


    rich_summary = build_rich_summary(results)
    rich_summary_out = args.out.replace(".json", "_rich_summary.json")

    # --- PLOTTING (single marker per model) ---
    for split in ["gt", "aug", "illus"]:
        pts = {}
        for model, vals in rich_summary.items():
            if split in vals:
                pts[model] = (vals[split]["pos_gap"], vals[split]["accuracy"])
        if pts:
            fig_path = args.out.replace(".json", f"_{split}.png")
            plot_accuracy_vs_gap(pts, f"{split.upper()}: Accuracy vs Gap", fig_path)
            print(f"📊 Saved plot {fig_path}")

if __name__=="__main__":
    main()
