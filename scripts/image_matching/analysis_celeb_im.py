# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score

# # ---------------------- LOAD DATA ----------------------
# with open("./results/all_parsed_celeb_im.json", "r") as f:
#     data = json.load(f)


# # ---------------------- HELPERS ----------------------
# def compute_f1_metrics(d, gender_filter=None):
#     """
#     Compute F1_gt, F1_aug, F1_illus for each model,
#     optionally filtered by gender.
#     """
#     results = {}
#     for model, sizes in d.items():
#         for size, domains in sizes.items():
#             split_vals = {"gt": [], "aug": [], "illus": []}
#             for domain, tests in domains.items():
#                 for test_id, test_data in tests.items():
#                     if gender_filter and test_data.get("gender") != gender_filter:
#                         continue

#                     # loop over split entries
#                     for split_key, split_data in test_data.items():
#                         if split_key in ("label", "parsed_label", "gender"):
#                             continue

#                         ans_idx = split_data.get("answer_idx", -1)
#                         pred_idx = split_data.get("pred_idx", -1)
#                         if ans_idx == -1 or pred_idx == -1:
#                             continue

#                         correct = int(pred_idx == ans_idx)
#                         y_true, y_pred = 1, 1 if correct else 0

#                         if split_key == "gt":
#                             split_vals["gt"].append((y_true, y_pred))
#                         elif split_key.startswith("aug"):
#                             split_vals["aug"].append((y_true, y_pred))
#                         elif split_key.startswith("illus"):
#                             split_vals["illus"].append((y_true, y_pred))

#             # Aggregate F1
#             res = {}
#             for split in ["gt", "aug", "illus"]:
#                 if split_vals[split]:
#                     y_true = [yt for yt, _ in split_vals[split]]
#                     y_pred = [yp for _, yp in split_vals[split]]
#                     res[split] = f1_score(y_true, y_pred)
#                 else:
#                     res[split] = 0.0

#             results[f"{model}_{size}"] = res
#     return results


# def compute_accuracy_metrics(d, gender_filter=None):
#     """
#     Compute simple accuracy (%) for gt, aug, illus.
#     """
#     results = {}
#     for model, sizes in d.items():
#         for size, domains in sizes.items():
#             split_vals = {"gt": [], "aug": [], "illus": []}
#             for domain, tests in domains.items():
#                 for test_id, test_data in tests.items():
#                     if gender_filter and test_data.get("gender") != gender_filter:
#                         continue
#                     for split_key, split_data in test_data.items():
#                         if split_key in ("label", "parsed_label", "gender"):
#                             continue
#                         ans_idx = split_data.get("answer_idx", -1)
#                         pred_idx = split_data.get("pred_idx", -1)
#                         if ans_idx == -1 or pred_idx == -1:
#                             continue
#                         split_type = (
#                             "gt" if split_key == "gt"
#                             else "aug" if split_key.startswith("aug")
#                             else "illus"
#                         )
#                         split_vals[split_type].append(int(pred_idx == ans_idx))

#             # Aggregate accuracy
#             results[f"{model}_{size}"] = {
#                 split: float(np.mean(vals)) if vals else 0.0
#                 for split, vals in split_vals.items()
#             }
#     return results


# # ---------------------- RUN ----------------------
# male_f1 = compute_f1_metrics(data, gender_filter="male")
# female_f1 = compute_f1_metrics(data, gender_filter="female")

# male_acc = compute_accuracy_metrics(data, gender_filter="male")
# female_acc = compute_accuracy_metrics(data, gender_filter="female")

# # Save avg performance per model/param (accuracy version)
# avg_performance = compute_accuracy_metrics(data)
# with open("./results/avg_performance_celeb_im.json", "w") as f:
#     json.dump(avg_performance, f, indent=2)

# print("✅ Saved avg accuracy performance to ./results/avg_performance_celeb_im.json")





import json
import numpy as np
import matplotlib.pyplot as plt
import re
import colorsys
from sklearn.metrics import f1_score

# ---------------------- LOAD DATA ----------------------
with open("./results/im/celeb/all_parsed_celeb_im.json", "r") as f:
    data = json.load(f)



def get_mean(val):
    return val["mean"] if isinstance(val, dict) else val


# ---------------------- METRIC COMPUTATION ----------------------
# def compute_f1_metrics(d, gender_filter=None):
#     results = {}
#     for model, sizes in d.items():
#         for size, domains in sizes.items():
#             split_vals = {"gt": [], "aug": [], "illus": []}
#             for domain, tests in domains.items():
#                 for _, test_data in tests.items():
#                     if gender_filter and test_data.get("gender") != gender_filter:
#                         continue
#                     for split_key, split_data in test_data.items():
#                         if split_key in ("label", "parsed_label", "gender"):
#                             continue
#                         ans_idx = split_data.get("answer_idx", -1)
#                         pred_idx = split_data.get("pred_idx", -1)
#                         if ans_idx == -1 or pred_idx == -1:
#                             continue
#                         correct = int(pred_idx == ans_idx)
#                         y_true, y_pred = 1, 1 if correct else 0
#                         if split_key == "gt":
#                             split_vals["gt"].append((y_true, y_pred))
#                         elif split_key.startswith("aug"):
#                             split_vals["aug"].append((y_true, y_pred))
#                         elif split_key.startswith("illus"):
#                             split_vals["illus"].append((y_true, y_pred))
#             res = {}
#             for split in ["gt", "aug", "illus"]:
#                 if split_vals[split]:
#                     y_true = [yt for yt, _ in split_vals[split]]
#                     y_pred = [yp for _, yp in split_vals[split]]
#                     res[split] = f1_score(y_true, y_pred)
#                 else:
#                     res[split] = 0.0
#             results[f"{model}_{size}"] = res
#     return results



# def compute_f1_metrics(d, gender_filter=None):
#     """
#     Compute F1 mean/std for gt, aug, illus.
#     Groups by original GT sample (1 gt + 15 aug + 15 illus).
#     """
#     results = {}
#     for model, sizes in d.items():
#         for size, domains in sizes.items():

#             # Store per-group metrics
#             group_metrics = {"gt": [], "aug": [], "illus": []}

#             for domain, tests in domains.items():
#                 for _, test_data in tests.items():
#                     if gender_filter and test_data.get("gender") != gender_filter:
#                         continue

#                     # --- GT ---
#                     if "gt" in test_data:
#                         ans_idx = test_data["gt"].get("answer_idx", -1)
#                         pred_idx = test_data["gt"].get("pred_idx", -1)
#                         if ans_idx != -1 and pred_idx != -1:
#                             y_true = [1]
#                             y_pred = [1 if pred_idx == ans_idx else 0]
#                             f1 = f1_score(y_true, y_pred)
#                             group_metrics["gt"].append(f1)

#                     # --- AUG variants ---
#                     aug_y_true, aug_y_pred = [], []
#                     for split_key, split_data in test_data.items():
#                         if split_key.startswith("aug"):
#                             ans_idx = split_data.get("answer_idx", -1)
#                             pred_idx = split_data.get("pred_idx", -1)
#                             if ans_idx != -1 and pred_idx != -1:
#                                 aug_y_true.append(1)
#                                 aug_y_pred.append(1 if pred_idx == ans_idx else 0)
#                     if aug_y_true:
#                         # F1 per aug variant = just binary → same as accuracy
#                         aug_scores = [f1_score([1], [yp]) for yp in aug_y_pred]
#                         group_metrics["aug"].append({
#                             "mean": float(np.mean(aug_scores)),
#                             "std": float(np.std(aug_scores))
#                         })

#                     # --- ILLUS variants ---
#                     illus_y_true, illus_y_pred = [], []
#                     for split_key, split_data in test_data.items():
#                         if split_key.startswith("illus"):
#                             ans_idx = split_data.get("answer_idx", -1)
#                             pred_idx = split_data.get("pred_idx", -1)
#                             if ans_idx != -1 and pred_idx != -1:
#                                 illus_y_true.append(1)
#                                 illus_y_pred.append(1 if pred_idx == ans_idx else 0)
#                     if illus_y_true:
#                         illus_scores = [f1_score([1], [yp]) for yp in illus_y_pred]
#                         group_metrics["illus"].append({
#                             "mean": float(np.mean(illus_scores)),
#                             "std": float(np.std(illus_scores))
#                         })

#             # --- Aggregate across all GT groups for this model ---
#             res = {}
#             # GT: single F1 per group
#             if group_metrics["gt"]:
#                 res["gt"] = {
#                     "mean": float(np.mean(group_metrics["gt"])),
#                     "std": float(np.std(group_metrics["gt"]))
#                 }
#             else:
#                 res["gt"] = {"mean": 0.0, "std": 0.0}

#             # AUG / ILLUS: aggregate per-group means/stds
#             for split in ["aug", "illus"]:
#                 if group_metrics[split]:
#                     means = [g["mean"] for g in group_metrics[split]]
#                     stds  = [g["std"] for g in group_metrics[split]]
#                     res[split] = {
#                         "mean": float(np.mean(means)),
#                         "std": float(np.mean(stds))   # avg within-group stds
#                     }
#                 else:
#                     res[split] = {"mean": 0.0, "std": 0.0}

#             results[f"{model}_{size}"] = res

#     return results


# def compute_accuracy_metrics(d, gender_filter=None):
#     results = {}
#     for model, sizes in d.items():
#         for size, domains in sizes.items():
#             split_vals = {"gt": [], "aug": [], "illus": []}
#             for domain, tests in domains.items():
#                 for _, test_data in tests.items():
#                     if gender_filter and test_data.get("gender") != gender_filter:
#                         continue
#                     for split_key, split_data in test_data.items():
#                         if split_key in ("label", "parsed_label", "gender"):
#                             continue
#                         ans_idx = split_data.get("answer_idx", -1)
#                         pred_idx = split_data.get("pred_idx", -1)
#                         if ans_idx == -1 or pred_idx == -1:
#                             continue
#                         split_type = (
#                             "gt" if split_key == "gt"
#                             else "aug" if split_key.startswith("aug")
#                             else "illus"
#                         )
#                         split_vals[split_type].append(int(pred_idx == ans_idx))
#             results[f"{model}_{size}"] = {
#                 split: float(np.mean(vals)) if vals else 0.0
#                 for split, vals in split_vals.items()
#             }
#     return results


def compute_accuracy_metrics(d, gender_filter=None):
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
                    if gender_filter and test_data.get("gender") != gender_filter:
                        continue

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
def plot_gender_segments(metric_name, male_metrics, female_metrics, filename):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=160)

    # background grid
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    ax.contourf(X, Y, Z, levels=100, cmap="Blues", alpha=0.5)

    ax.axhline(0.5, color="white", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1)

    # Quadrant text
    # kwargs = dict(ha="center", va="center", fontsize=11, alpha=0.9, color="black")
    # ax.text(0.25, 0.75, "High Illus Robust,\nLow Aug Robust", **kwargs)
    # ax.text(0.75, 0.75, "Robust to Both\n(ideal)", fontweight="bold", **kwargs)
    # ax.text(0.25, 0.25, "Fragile to Both", **kwargs)
    # ax.text(0.75, 0.25, "High Aug Robust,\nLow Illus Robust", **kwargs)

    seen_families = {}
    for model in male_metrics:
        if model in female_metrics:
            mx, my = get_mean(male_metrics[model]["aug"]), get_mean(male_metrics[model]["illus"])
            fx, fy = get_mean(female_metrics[model]["aug"]), get_mean(female_metrics[model]["illus"])

            family = get_family(model)
            base_color = FAMILY_COLORS[family]
            male_color = adjust_color(base_color, 1.3)
            female_color = adjust_color(base_color, 0.7)

            # connecting line
            ax.plot([mx, fx], [my, fy], color="gray", linewidth=1, alpha=0.7)

            ax.scatter(mx, my, s=70, color=male_color, marker="o", edgecolor="black", linewidth=0.6)
            ax.scatter(fx, fy, s=70, color=female_color, marker="s", edgecolor="black", linewidth=0.6)

            midx, midy = (mx + fx) / 2, (my + fy) / 2
            ax.text(midx + 0.01, midy + 0.01, shorten_model_name(model), fontsize=8.5, color="black")

            if family not in seen_families:
                seen_families[family] = base_color

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Augmentation Robustness", fontsize=12)
    ax.set_ylabel("Illusion Robustness", fontsize=12)
    ax.set_title(f"{metric_name} (Male vs Female)", fontsize=16)

    handles = []
    for fam, col in seen_families.items():
        male_col = adjust_color(col, 1.3)
        female_col = adjust_color(col, 0.7)
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=male_col,
                                  markeredgecolor="black", markersize=7, label=f"{fam} (Male)"))
        handles.append(plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=female_col,
                                  markeredgecolor="black", markersize=7, label=f"{fam} (Female)"))

    # ax.legend(handles=handles, title="Model Families", bbox_to_anchor=(1.05, 1), loc="upper left")
    # --- Legend below figure ---
    fig.legend(
        handles=handles,
        title="Model Families",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),   # move below the figure
        ncol=3,                        # number of columns (adjust as needed)
        frameon=False
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------- RUN ----------------------
# male_f1 = compute_f1_metrics(data, gender_filter="male")
# female_f1 = compute_f1_metrics(data, gender_filter="female")

male_acc = compute_accuracy_metrics(data, gender_filter="male")
female_acc = compute_accuracy_metrics(data, gender_filter="female")

# Save plots (using Accuracy metrics for axes)
plot_gender_segments("Accuracy", male_acc, female_acc, "./results/im/celeb/acc_male_female_im.png")
# plot_gender_segments("F1 Score", male_f1, female_f1, "./results/f1_male_female.png")

# Save overall avg accuracy
# avg_performance = compute_accuracy_metrics(data)
# with open("./results/avg_performance_celeb_im.json", "w") as f:
#     json.dump(avg_performance, f, indent=2)


# # Save overall avg accuracy
# avg_performance_f1 = compute_f1_metrics(data)
# with open("./results/avg_performance_celeb_im_f1.json", "w") as f:
#     json.dump(avg_performance_f1, f, indent=2)

print("✅ Saved plots and avg accuracy performance")
