"""Slide figure: homogeneity, completeness and V-measure on four toy partitions (notebook 3.3, section 3).

Twelve samples with three known labels (earthquake, explosion, noise; four
each), grouped four ways, as y_toy and toy_clusters in the notebook. Each row
draws the clusters as boxes holding the samples, colored by label, and prints
the three scores from scikit-learn's homogeneity_completeness_v_measure.

Regenerate: pixi run python book/slides/2026/figscripts/cluster_scores_toy.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import homogeneity_completeness_v_measure  # noqa: E402

plt.rcParams.update({"font.size": 20})
MUTED, INK = "#6e675c", "#26241f"
LABEL_COLORS = {"earthquake": "#1f77b4", "explosion": "#ff7f0e", "noise": "#d62728"}
OUT = Path(__file__).resolve().parent.parent / "figs" / "3.3_clustering"
OUT.mkdir(parents=True, exist_ok=True)

# ---- as in notebook 3.3 -----------------------------------------------------
y_toy = np.repeat(["earthquake", "explosion", "noise"], 4)   # the known label of each of 12 samples
toy_clusters = {                                                # the cluster of each sample, four cases
    "perfect, numbers swapped": np.repeat([2, 0, 1], 4),
    "all in one cluster": np.zeros(12, dtype=int),
    "each sample its own cluster": np.arange(12),
    "mixed": np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0]),
}
scores = {name: homogeneity_completeness_v_measure(y_toy, lab) for name, lab in toy_clusters.items()}
for name, (h, c, v) in scores.items():
    print(f"{name:30s} homogeneity {h:.3f}  completeness {c:.3f}  V-measure {v:.3f}")

fig, ax = plt.subplots(figsize=(22, 8.0))
ax.set_xlim(-0.2, 29.0); ax.set_ylim(-0.75, len(toy_clusters) - 0.45)
ax.axis("off")
x_start = 7.6
for r, (name, lab) in enumerate(toy_clusters.items()):
    y = len(toy_clusters) - 1 - r
    ax.text(0, y, name, va="center", fontsize=21, fontweight="bold", color=INK)
    x = x_start
    for j in np.unique(lab):
        members = np.where(lab == j)[0]
        w = 0.62 * len(members) + 0.25
        ax.add_patch(FancyBboxPatch((x, y - 0.3), w, 0.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#f4efe6", ec=MUTED, lw=1.5))
        for k, i in enumerate(members):
            ax.scatter(x + 0.43 + 0.62 * k, y, s=260, color=LABEL_COLORS[y_toy[i]], edgecolor="white", lw=1, zorder=3)
        ax.text(x + w / 2, y + 0.33, str(j), ha="center", va="bottom", fontsize=16, color=MUTED)
        x += w + (0.12 if len(np.unique(lab)) > 4 else 0.22)
    h, c, v = scores[name]
    ax.text(29.0, y, f"h = {h:.3f}   c = {c:.3f}   V = {v:.3f}", ha="right", va="center", fontsize=21,
            family="monospace", color=INK)
for k, (lab, col) in enumerate(LABEL_COLORS.items()):
    ax.scatter(x_start + 0.3 + 4.2 * k, -0.5, s=260, color=col, edgecolor="white")
    ax.text(x_start + 0.65 + 4.2 * k, -0.5, lab, va="center", fontsize=19)
fig.suptitle("Twelve samples, three known labels (four each), grouped four ways; boxes are clusters (number above); "
             "h homogeneity, c completeness, V V-measure (scikit-learn)",
             x=0.01, ha="left", fontsize=20, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = OUT / "cluster_scores_toy_slide.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
