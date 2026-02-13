import json
import numpy as np
import matplotlib.pyplot as plt
import re
import colorsys
import matplotlib.transforms as mtransforms
from matplotlib.scale import ScaleBase
import matplotlib.ticker as mticker
from matplotlib import scale as mscale
from scipy.stats import ttest_ind, mannwhitneyu

# ---------------- CONFIG ----------------
PIECEWISE_CUTOFF = 0.9
PIECEWISE_STRETCH = 5.0
LABEL_MIN_DY = 0.03
MALE_MARKER = "o"
FEMALE_MARKER = "s"
delta = 0.0

# ---------------- LOAD ----------------
with open("./results/attribution-partial/celeb/all_parsed_celeb_p.json", "r") as f:
    data = json.load(f)

# ---------------- IOU METRICS ----------------
def compute_iou_metrics(d, gender_filter=None):
    results = {}
    for model, sizes in d.items():
        for size, domains in sizes.items():
            attr_vals = {}
            for domain, tests in domains.items():
                for test_id, test_data in tests.items():
                    if gender_filter and test_data.get("gender") != gender_filter:
                        continue
                    gt = test_data.get("gt") or test_data.get("gt0") or {}
                    if isinstance(gt, dict):
                        for attr in gt.get("features", {}).keys():
                            attr_vals.setdefault(attr, {"aug": [], "illus": []})
                    for k, v in test_data.items():
                        if not isinstance(v, dict):
                            continue
                        if k.startswith("aug"):
                            if "features" in v:
                                for attr, info in v["features"].items():
                                    attr_vals.setdefault(attr, {"aug": [], "illus": []})
                                    attr_vals[attr]["aug"].append(info["iou"])
                        elif k.startswith("illus"):
                            if "features" in v:
                                for attr, info in v["features"].items():
                                    attr_vals.setdefault(attr, {"aug": [], "illus": []})
                                    attr_vals[attr]["illus"].append(info["iou"])
            avg_aug, avg_illus = [], []
            for attr, vals in attr_vals.items():
                IoU_aug = float(np.mean(vals["aug"]) - delta) if vals["aug"] else 0.0
                IoU_illus = float(np.mean(vals["illus"]) - delta) if vals["illus"] else 0.0
                results.setdefault(attr, {})
                results[attr][f"{model}_{size}"] = (IoU_aug, IoU_illus)
                if attr != "avg":
                    avg_aug.append(IoU_aug)
                    avg_illus.append(IoU_illus)
            if avg_aug or avg_illus:
                results.setdefault("avg", {})
                results["avg"][f"{model}_{size}"] = (
                    float(np.mean(avg_aug)) if avg_aug else 0.0,
                    float(np.mean(avg_illus)) if avg_illus else 0.0,
                )
    return results

# ---------------- HELPERS ----------------
def shorten_model_name(name: str) -> str:
    low = name.lower()
    m = re.search(r"(\d+)[bB]", name)
    if m:
        size = m.group(1) + "B"
        if low.startswith("internvl"): return f"I-{size}"
        elif low.startswith("qwen"): return f"Q-{size}"
        elif low.startswith("gemma"): return f"G-{size}"
        elif low.startswith("llava"): return f"L-{size}"
        elif low.startswith("minigpt"): return f"M-{size}"
    return name

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

def adjust_color(color, factor=1.0):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)

# ---------------- LABEL REPEL ----------------
def repel_labels(ax, labels, min_dy=LABEL_MIN_DY, ylim=(0.0, 1.0)):
    if not labels: return
    ys = np.array([lab['y'] for lab in labels], dtype=float)
    ys.sort()
    for _ in range(200):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i-1] < min_dy:
                delta = (min_dy - (ys[i] - ys[i-1])) / 2.0
                ys[i] += delta
                ys[i-1] -= delta
                moved = True
        ys = np.clip(ys, ylim[0] + 0.01, ylim[1] - 0.01)
        if not moved: break
    labels_sorted = sorted(labels, key=lambda d: d['y'])
    for lab, y_adj in zip(labels_sorted, ys):
        ax.annotate(
            lab['text'],
            xy=(lab['x_anchor'], lab['y_anchor']),
            xytext=(lab['x'], y_adj),
            textcoords='data',
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.6, alpha=0.85),
            fontsize=8.5, color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75),
        )

# ---------------- SCALE ----------------
class PiecewiseScale(mtransforms.Transform):
    input_dims = output_dims = 1
    is_separable = True
    def __init__(self, cutoff=PIECEWISE_CUTOFF, stretch=PIECEWISE_STRETCH):
        super().__init__(); self.cutoff = cutoff; self.stretch = stretch
    def transform_non_affine(self, x):
        x = np.array(x); y = np.empty_like(x)
        mask = x <= self.cutoff
        y[mask] = x[mask]
        y[~mask] = self.cutoff + (x[~mask] - self.cutoff) * self.stretch
        return y
    def inverted(self): return InvertedPiecewiseScale(self.cutoff, self.stretch)

class InvertedPiecewiseScale(PiecewiseScale):
    def transform_non_affine(self, y):
        y = np.array(y); x = np.empty_like(y)
        mask = y <= self.cutoff
        x[mask] = y[mask]
        x[~mask] = self.cutoff + (y[~mask] - self.cutoff) / self.stretch
        return x

class PiecewiseAxis(ScaleBase):
    name = "piecewise"
    def __init__(self, axis, **kwargs):
        self.cutoff = kwargs.get("cutoff", PIECEWISE_CUTOFF)
        self.stretch = kwargs.get("stretch", PIECEWISE_STRETCH)
    def get_transform(self): return PiecewiseScale(self.cutoff, self.stretch)
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mticker.MaxNLocator(6))
        axis.set_major_formatter(mticker.ScalarFormatter())

mscale.register_scale(PiecewiseAxis)

# ---------------- PLOT GENDER ----------------
def plot_gender_segments(attr, male_metrics, female_metrics, filename):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=160)
    x = np.linspace(0, 1, 400); y = np.linspace(0, 1, 400)
    X, Y = np.meshgrid(x, y); Z = X * Y
    ax.contourf(X, Y, Z, levels=100, cmap="Blues", alpha=0.75)
    ax.axhline(0.5, color="white", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1)
    kwargs = dict(ha="center", va="center", fontsize=11, alpha=0.9, color="black")
    ax.text(0.25, 0.75, "High Illus Robust,\nLow Aug Robust", **kwargs)
    ax.text(0.75, 0.75, "Robust to Both\n(ideal)", fontweight="bold", **kwargs)
    ax.text(0.25, 0.25, "Fragile to Both", **kwargs)
    ax.text(0.75, 0.25, "High Aug Robust,\nLow Illus Robust", **kwargs)

    seen_families = {}; all_aug_vals = []; label_requests = []
    for model in male_metrics.get(attr, {}):
        if model in female_metrics.get(attr, {}):
            mx, my = male_metrics[attr][model]; fx, fy = female_metrics[attr][model]
            all_aug_vals.extend([mx, fx])
            family = get_family(model); base_color = FAMILY_COLORS[family]
            male_color = adjust_color(base_color, 1.3)
            female_color = adjust_color(base_color, 0.7)
            ax.plot([mx, fx], [my, fy], color="gray", linewidth=1, alpha=0.8)
            ax.scatter(mx, my, color=male_color, s=70, edgecolor="black", linewidth=0.6, marker=MALE_MARKER)
            ax.scatter(fx, fy, color=female_color, s=70, edgecolor="black", linewidth=0.6, marker=FEMALE_MARKER)
            if fx >= mx: x_anchor, y_anchor = fx, fy
            else: x_anchor, y_anchor = mx, my
            if x_anchor >= PIECEWISE_CUTOFF: x_lab = max(0.0, x_anchor - 0.03)
            else: x_lab = min(1.0, x_anchor + 0.02)
            label_requests.append({
                "text": shorten_model_name(model),
                "x_anchor": x_anchor, "y_anchor": y_anchor,
                "x": x_lab, "y": y_anchor
            })
            if family not in seen_families: seen_families[family] = base_color
    if all_aug_vals and np.mean(np.array(all_aug_vals) >= PIECEWISE_CUTOFF) > 0.7:
        ax.set_xscale("piecewise", cutoff=PIECEWISE_CUTOFF, stretch=PIECEWISE_STRETCH)
    ax.set_title(f"{attr.capitalize()} Robustness (Male vs Female)", fontsize=16, pad=12)
    ax.set_xlabel("IoU_aug (Augmentation Robustness)", fontsize=12)
    ax.set_ylabel("IoU_illus (Illusion Robustness)", fontsize=12)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0); ax.grid(True, linewidth=0.4, linestyle=":", alpha=0.6)
    repel_labels(ax, label_requests, min_dy=LABEL_MIN_DY, ylim=(0.0, 1.0))
    handles = []
    for fam, col in seen_families.items():
        male_col = adjust_color(col, 1.3); female_col = adjust_color(col, 0.7)
        handles.append(plt.Line2D([0], [0], marker=MALE_MARKER, color="w",
                                  markerfacecolor=male_col, markeredgecolor="black",
                                  markersize=7, label=f"{fam} (Male)"))
        handles.append(plt.Line2D([0], [0], marker=FEMALE_MARKER, color="w",
                                  markerfacecolor=female_col, markeredgecolor="black",
                                  markersize=7, label=f"{fam} (Female)"))
    ax.legend(handles=handles, title="Model Families",
              bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(filename, dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- AVG PERF ----------------
def compute_overall_avgs(d):
    results = {}
    for model, sizes in d.items():
        for size, domains in sizes.items():
            gt_vals, aug_vals, illus_vals = [], [], []
            for domain, tests in domains.items():
                for test_id, test_data in tests.items():
                    if "gt" in test_data and "avg" in test_data["gt"]:
                        gt_vals.append(test_data["gt"]["avg"])
                    for k, v in test_data.items():
                        if k.startswith("aug") and isinstance(v, dict) and "avg" in v:
                            aug_vals.append(v["avg"])
                        if k.startswith("illus") and isinstance(v, dict) and "avg" in v:
                            illus_vals.append(v["avg"])
            key = f"{model}_{size}"
            results[key] = {
                "gt": float(np.mean(gt_vals)) if gt_vals else 0.0,
                "aug": float(np.mean(aug_vals)) if aug_vals else 0.0,
                "illus": float(np.mean(illus_vals)) if illus_vals else 0.0,
            }
    return results

# ---------------- INTERNVL ANALYSIS ----------------
def analyze_internvl_corrections(d):
    def is_internvl35(name: str) -> bool:
        return bool(re.match(r".*internvl[\._-]?3[\._-]?5", name.lower()))
    results = {"gt": {"fix": [], "fail": []},
               "aug": {"fix": [], "fail": []},
               "illus": {"fix": [], "fail": []}}
    for model, sizes in d.items():
        if not is_internvl35(model): continue
        for size, domains in sizes.items():
            for domain, tests in domains.items():
                for test_id, test_data in tests.items():
                    gt_avg = test_data.get("gt", {}).get("avg", None)
                    aug_avgs = [v["avg"] for k, v in test_data.items()
                                if k.startswith("aug") and isinstance(v, dict) and "avg" in v]
                    aug_baseline = np.mean(aug_avgs) if aug_avgs else None
                    illus_avgs = [v["avg"] for k, v in test_data.items()
                                  if k.startswith("illus") and isinstance(v, dict) and "avg" in v]
                    illus_baseline = np.mean(illus_avgs) if illus_avgs else None
                    for split_key, split_data in test_data.items():
                        if not isinstance(split_data, dict): continue
                        if "avg" not in split_data or "think_len" not in split_data: continue
                        avg_iou = split_data["avg"]; think_len = split_data["think_len"]
                        if gt_avg is not None:
                            if gt_avg <= 0.5 and avg_iou > 0.5: results["gt"]["fix"].append(think_len)
                            elif gt_avg > 0.5 and avg_iou <= 0.5: results["gt"]["fail"].append(think_len)
                        if aug_baseline is not None:
                            if aug_baseline <= 0.5 and avg_iou > 0.5: results["aug"]["fix"].append(think_len)
                            elif aug_baseline > 0.5 and avg_iou <= 0.5: results["aug"]["fail"].append(think_len)
                        if illus_baseline is not None:
                            if illus_baseline <= 0.5 and avg_iou > 0.5: results["illus"]["fix"].append(think_len)
                            elif illus_baseline > 0.5 and avg_iou <= 0.5: results["illus"]["fail"].append(think_len)
    for ctx in ["gt", "aug", "illus"]:
        fix = results[ctx]["fix"]; fail = results[ctx]["fail"]
        print(f"\n[InternVL3.5][{ctx}] Fixes={len(fix)}, Fails={len(fail)}")
        if fix: print(f"  Mean think_len (fix): {np.mean(fix):.2f}")
        if fail: print(f"  Mean think_len (fail): {np.mean(fail):.2f}")
        # inside analyze_internvl_corrections, replace the histogram section:
        if fix or fail:
            plt.figure(figsize=(6,4))

            # choose bins based on combined data
            bins = np.linspace(0, max(fix+fail+[1]), 40)

            # plot side-by-side histograms with slight offset
            width = (bins[1] - bins[0]) * 0.4
            if fix:
                plt.hist(fix, bins=bins, alpha=0.8, label="Fixes",
                        color="tab:blue", width=width, align="left")
            if fail:
                plt.hist(fail, bins=bins, alpha=0.8, label="Fails",
                        color="tab:orange", width=width, align="right")

            plt.title(f"InternVL3.5 [{ctx}] think_len", fontsize=14)
            plt.xlabel("think_len", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.yscale("log")  # helpful since you have a heavy head + long tail
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(frameon=False)
            plt.tight_layout()

            outname = f"./results/attribution-partial/internvl35_{ctx}_thinklen_hist.png"
            plt.savefig(outname, dpi=300)
            plt.close()
            print(f"  Saved histogram → {outname}")

    return results

# ---------------- RUN ----------------
if __name__ == "__main__":
    male_attr_metrics = compute_iou_metrics(data, gender_filter="male")
    female_attr_metrics = compute_iou_metrics(data, gender_filter="female")
    attributes = ["avg", "left_eye", "right_eye", "nose", "mouth"]
    for attr in attributes:
        if attr in male_attr_metrics and attr in female_attr_metrics:
            filename = f"./results/attribution-partial/celeb/{attr}_male_female_attribution.png"
            plot_gender_segments(attr, male_attr_metrics, female_attr_metrics, filename)
    print("Charts saved: avg, left_eye, right_eye, nose, mouth (Male vs Female)")

    analyze_internvl_corrections(data)
