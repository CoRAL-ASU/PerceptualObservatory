import json
import numpy as np
import matplotlib.pyplot as plt
import re
import colorsys

# ---------------------- LOAD DATA ----------------------
with open("./results/all_parsed_word_im.json", "r") as f:
    data = json.load(f)


def get_mean(val):
    return val["mean"] if isinstance(val, dict) else val


# ---------------------- METRIC COMPUTATION ---------------------- #
def compute_accuracy_metrics(d):
    """
    Compute accuracy mean/std for gt, aug, illus.
    Groups by original GT sample (1 gt + 15 aug + 15 illus).
    """
    results = {}
    for model, sizes in d.items():
        for size, domains in sizes.items():

            # Store per-group metrics first
            group_metrics = {"gt": [], "aug": [], "illus": []}

            for domain, tests in domains.items():
                for _, test_data in tests.items():

                    # --- GT ---
                    if "gt" in test_data:
                        ans_idx = test_data["gt"].get("answer_idx", -1)
                        pred_idx = test_data["gt"].get("pred_idx", -1)
                        if ans_idx != -1 and pred_idx != -1:
                            group_metrics["gt"].append(int(pred_idx == ans_idx))

                    # --- AUG variants ---
                    aug_scores = []
                    for split_key, split_data in test_data.items():
                        if split_key.startswith("aug"):
                            ans_idx = split_data.get("answer_idx", -1)
                            pred_idx = split_data.get("pred_idx", -1)
                            if ans_idx != -1 and pred_idx != -1:
                                aug_scores.append(int(pred_idx == ans_idx))
                    if aug_scores:
                        group_metrics["aug"].append({
                            "mean": float(np.mean(aug_scores)),
                            "std": float(np.std(aug_scores))
                        })

                    # --- ILLUS variants ---
                    illus_scores = []
                    for split_key, split_data in test_data.items():
                        if split_key.startswith("illus"):
                            ans_idx = split_data.get("answer_idx", -1)
                            pred_idx = split_data.get("pred_idx", -1)
                            if ans_idx != -1 and pred_idx != -1:
                                illus_scores.append(int(pred_idx == ans_idx))
                    if illus_scores:
                        group_metrics["illus"].append({
                            "mean": float(np.mean(illus_scores)),
                            "std": float(np.std(illus_scores))
                        })

            # --- Aggregate across all GT groups for this model ---
            res = {}
            # GT: single value per group → can just average
            if group_metrics["gt"]:
                res["gt"] = {
                    "mean": float(np.mean(group_metrics["gt"])),
                    "std": float(np.std(group_metrics["gt"]))
                }
            else:
                res["gt"] = {"mean": 0.0, "std": 0.0}

            # AUG/ILLUS: each entry already has mean/std → aggregate across groups
            for split in ["aug", "illus"]:
                if group_metrics[split]:
                    means = [g["mean"] for g in group_metrics[split]]
                    stds  = [g["std"] for g in group_metrics[split]]
                    res[split] = {
                        "mean": float(np.mean(means)),
                        "std": float(np.mean(stds))  # avg within-group stds
                    }
                else:
                    res[split] = {"mean": 0.0, "std": 0.0}

            results[f"{model}_{size}"] = res

    return results


# ---------------------- COLOR / LABEL HELPERS ----------------------
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
    if "qwen" in low: return "Qwen"
    if "gemma" in low: return "Gemma"
    if "llava" in low: return "LLaVA"
    if "minigpt" in low: return "MiniGPT"
    return "Other"


def adjust_color(color, factor=1.0):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


def shorten_model_name(name: str) -> str:
    m = re.search(r"(\d+)[bB]", name)
    if m:
        size = m.group(1) + "B"
        if name.lower().startswith("internvl"): return f"I-{size}"
        if name.lower().startswith("qwen"): return f"Q-{size}"
        if name.lower().startswith("gemma"): return f"G-{size}"
        if name.lower().startswith("llava"): return f"L-{size}"
        if name.lower().startswith("minigpt"): return f"M-{size}"
    return name


# ---------------------- PLOTTING ----------------------
def plot_models(metric_name, metrics, filename):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=160)

    x = np.linspace(0, 1, 400); y = np.linspace(0, 1, 400)
    X, Y = np.meshgrid(x, y); Z = (1 - X) * Y
    ax.contourf(X, Y, Z, levels=100, cmap="RdBu", alpha=0.65)
    ax.axhline(0.5, color="white", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1)

    kwargs = dict(ha="center", va="center", fontsize=11, alpha=0.9, color="black")
    ax.text(0.25, 0.75, "High Acc, Low Gap\n(ideal)", fontweight="bold", **kwargs)
    ax.text(0.75, 0.75, "High Acc, High Gap", **kwargs)
    ax.text(0.25, 0.25, "Low Acc, Low Gap", **kwargs)
    ax.text(0.75, 0.25, "Low Acc, High Gap\n(worst)", **kwargs)

    seen_families = {}
    for model, vals in metrics.items():
        mx = get_mean(vals["aug"])
        my = get_mean(vals["illus"])

        family = get_family(model)
        base_color = FAMILY_COLORS[family]
        color = adjust_color(base_color, 1.0)

        ax.scatter(mx, my, s=70, color=color, marker="o", edgecolor="black", linewidth=0.6)
        ax.text(mx + 0.01, my + 0.01, shorten_model_name(model), fontsize=8.5, color="black")

        if family not in seen_families:
            seen_families[family] = base_color

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Augmentation Robustness", fontsize=12)
    ax.set_ylabel("Illusion Robustness", fontsize=12)
    ax.set_title(f"{metric_name} (Word Dataset)", fontsize=16)

    handles = []
    for fam, col in seen_families.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                                  markeredgecolor="black", markersize=7, label=f"{fam}"))

    ax.legend(handles=handles, title="Model Families", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------- RUN ----------------------
metrics = compute_accuracy_metrics(data)

# Save plots (Accuracy metrics for axes)
plot_models("Accuracy", metrics, "./results/acc_word_im.png")

# Save overall avg accuracy
with open("./results/avg_performance_word_im.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Saved word dataset plots and avg accuracy performance")


plot_models("Accuracy", metrics, "./results/acc_word_im.png")