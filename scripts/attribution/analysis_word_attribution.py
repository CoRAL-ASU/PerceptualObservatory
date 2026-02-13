import json
import numpy as np
import matplotlib.pyplot as plt


# ---------------------- LOAD DATA ----------------------
with open("./results/all_parsed_word_p.json", "r") as f:
    data = json.load(f)


# ---------------------- METRIC COMPUTATION ----------------------
def compute_iou_metrics(d):
    """
    Compute IoU_aug and IoU_illus per attribute for each model (word dataset).
    If features are missing/empty, count them as IoU = 0.
    """
    results = {}
    for model, sizes in d.items():
        for size, domains in sizes.items():
            attr_vals = {}
            for domain, tests in domains.items():
                for test_id, test_data in tests.items():

                    # --- Ground truth (define attributes)
                    gt = test_data.get("gt") or test_data.get("gt0") or {}
                    if isinstance(gt, dict):
                        if "features" in gt:
                            for attr in gt["features"].keys():
                                attr_vals.setdefault(attr, {"aug": [], "illus": []})
                        else:
                            # no features at all
                            attr_vals.setdefault("sequence", {"aug": [], "illus": []})

                    # --- Aug & Illus splits
                    for k, v in test_data.items():
                        if not isinstance(v, dict):
                            continue

                        if k.startswith("aug"):
                            if "features" in v:
                                if v["features"]:
                                    for attr, info in v["features"].items():
                                        attr_vals.setdefault(attr, {"aug": [], "illus": []})
                                        attr_vals[attr]["aug"].append(info.get("iou", 0.0))
                                else:
                                    # empty features → count as IoU = 0
                                    attr_vals.setdefault("sequence", {"aug": [], "illus": []})
                                    attr_vals["sequence"]["aug"].append(0.0)
                            else:
                                attr_vals.setdefault("sequence", {"aug": [], "illus": []})
                                attr_vals["sequence"]["aug"].append(v.get("avg", 0.0))

                        elif k.startswith("illus"):
                            if "features" in v:
                                if v["features"]:
                                    for attr, info in v["features"].items():
                                        attr_vals.setdefault(attr, {"aug": [], "illus": []})
                                        attr_vals[attr]["illus"].append(info.get("iou", 0.0))
                                else:
                                    attr_vals.setdefault("sequence", {"aug": [], "illus": []})
                                    attr_vals["sequence"]["illus"].append(0.0)
                            else:
                                attr_vals.setdefault("sequence", {"aug": [], "illus": []})
                                attr_vals["sequence"]["illus"].append(v.get("avg", 0.0))

            # Compute averages
            for attr, vals in attr_vals.items():
                IoU_aug = float(np.mean(vals["aug"])) if vals["aug"] else 0.0
                IoU_illus = float(np.mean(vals["illus"])) if vals["illus"] else 0.0

                results.setdefault(attr, {})
                results[attr][f"{model}_{size}"] = (IoU_aug, IoU_illus)
    return results


def compute_overall_avgs(d):
    """
    Compute overall averages per model across GT, aug, illus.
    If features are empty, count avg as 0.
    """
    results = {}
    for model, sizes in d.items():
        for size, domains in sizes.items():
            gt_vals, aug_vals, illus_vals = [], [], []
            for domain, tests in domains.items():
                for test_id, test_data in tests.items():
                    # GT avg
                    if "gt" in test_data and "avg" in test_data["gt"]:
                        gt_vals.append(test_data["gt"]["avg"] or 0.0)

                    # Augs
                    for k, v in test_data.items():
                        if k.startswith("aug") and isinstance(v, dict) and "avg" in v:
                            aug_vals.append(v["avg"] or 0.0)

                    # Illus
                    for k, v in test_data.items():
                        if k.startswith("illus") and isinstance(v, dict) and "avg" in v:
                            illus_vals.append(v["avg"] or 0.0)

            key = f"{model}_{size}"
            results[key] = {
                "gt": float(np.mean(gt_vals)) if gt_vals else 0.0,
                "aug": float(np.mean(aug_vals)) if aug_vals else 0.0,
                "illus": float(np.mean(illus_vals)) if illus_vals else 0.0,
            }
    return results


# ---------------------- PLOT ----------------------
def plot_word_dataset(attr, metrics, filename):
    plt.figure(figsize=(8.5, 8.5), dpi=160)

    # Background red-blue gradient
    x = np.linspace(0, 1, 400)
    y = np.linspace(0, 1, 400)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    bg = plt.contourf(X, Y, Z, levels=100, cmap="RdBu", alpha=0.65)

    # Quadrant lines
    plt.axhline(0.5, color="white", linestyle="--", linewidth=1)
    plt.axvline(0.5, color="white", linestyle="--", linewidth=1)

    # Quadrant labels
    kwargs = dict(ha="center", va="center", fontsize=11, alpha=0.9, color="black")
    plt.text(0.25, 0.75, "High Illus Robust,\nLow Aug Robust", **kwargs)
    plt.text(0.75, 0.75, "Robust to Both\n(ideal)", fontweight="bold", **kwargs)
    plt.text(0.25, 0.25, "Fragile to Both", **kwargs)
    plt.text(0.75, 0.25, "High Aug Robust,\nLow Illus Robust", **kwargs)

    # Scatter points for each model
    for model, (mx, my) in metrics.get(attr, {}).items():
        plt.scatter(mx, my, s=80, edgecolor="black", linewidth=0.6)
        plt.text(mx + 0.012, my + 0.012, model, fontsize=8.5, color="black")

    plt.title(f"{attr.capitalize()} Robustness (Word Dataset)", fontsize=16, pad=12)
    plt.xlabel("IoU_aug (Augmentation Robustness)", fontsize=12)
    plt.ylabel("IoU_illus (Illusion Robustness)", fontsize=12)
    plt.xlim(0, 1.25); plt.ylim(0, 1)
    plt.grid(True, linewidth=0.4, linestyle=":", alpha=0.6)

    cbar = plt.colorbar(bg, fraction=0.046, pad=0.04)
    cbar.set_label("Joint robustness (IoU_aug × IoU_illus)\n(low=red, high=blue)")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ---------------------- RUN ----------------------
attr_metrics = compute_iou_metrics(data)

# Word dataset is usually just "sequence"
plot_word_dataset("sequence", attr_metrics, "sequence_word_p.png")
print("Chart saved: sequence_word_p.png")

# Save averages
avg_performance = compute_overall_avgs(data)
with open("./results/avg_performance_word_p.json", "w") as f:
    json.dump(avg_performance, f, indent=2)
print("Saved avg performance to ./results/avg_performance_word_p.json")
