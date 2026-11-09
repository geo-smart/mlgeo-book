"""Lecture 18: diagram of k-fold cross-validation, shuffled and forward in time.

Writes figs/lec18_robust_training/cv_diagram.png. No data; a schematic of
40 samples in time order, 5 folds, and a final test period held aside.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parents[1] / "figs" / "lec18_robust_training"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN, VAL, TEST, UNUSED, GAP = "#9bb7d4", "#d95f02", "#1b1b1b", "#eeeeee", "#f1d98a"
N, K, N_TEST = 40, 5, 8
rng = np.random.default_rng(0)
order = rng.permutation(N - N_TEST)
plt.rcParams.update({"font.size": 17})
fig, axes = plt.subplots(1, 2, figsize=(21, 6.2))


def draw(ax, rows, title):
    for r, colors in enumerate(rows):
        for i, c in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, -r), 0.92, 0.8, color=c))
        ax.text(-0.6, -r + 0.4, f"step {r + 1}", ha="right", va="center")
    ax.set_xlim(-4.5, N + 0.2)
    ax.set_ylim(-K + 0.5, 1.4)
    ax.annotate("", xy=(N, 1.15), xytext=(0, 1.15),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(N / 2, 1.25, "time order of the samples", ha="center", va="bottom")
    ax.set_title(title, loc="left", fontweight="bold", pad=34)
    ax.axis("off")


shuffled = []
for k in range(K):
    val = set(order[k::K])
    shuffled.append([VAL if i in val else TRAIN for i in range(N - N_TEST)] + [TEST] * N_TEST)
draw(axes[0], shuffled, "Shuffled k-fold: validation samples sit between training samples")

edges = np.linspace(8, N - N_TEST, K + 1).astype(int)
forward = []
for k in range(K):
    a, b = edges[k], edges[k + 1] if k < K - 1 else N - N_TEST
    row = [TRAIN if i < a - 2 else GAP if i < a else VAL if i < b else UNUSED for i in range(N - N_TEST)]
    forward.append(row[:-2] + [GAP, GAP] + [TEST] * N_TEST)   # a gap also separates the test window
draw(axes[1], forward, "Forecast: train on the past, leave a gap, validate later")

fig.legend(handles=[Patch(color=TRAIN, label="training: fit the model"),
                    Patch(color=VAL, label="validation: compare models and settings"),
                    Patch(color=TEST, label="test: most recent, scored once"),
                    Patch(color=GAP, label="gap: dropped"),
                    Patch(color=UNUSED, label="not used in this step")],
           loc="lower center", ncol=5, frameon=False, fontsize=16)
fig.subplots_adjust(left=0.01, right=0.995, top=0.85, bottom=0.13, wspace=0.08)
fig.savefig(OUT / "cv_diagram.png", dpi=110)
print("wrote", OUT / "cv_diagram.png")

# Leakage: the same score with and without the leak, four cases from this course.
# Values: P395 and earthquake table from lec18_scores_ensembles_automl.py
# (numbers.json); oversampling from lec16_balancing.py; search from lec18 (search_optimism).
CASES = [
    ("Neighbors in time\nP395 forecast, MAE (mm)", "shuffled folds", 0.46, "future folds", 12.4, False),
    ("Same event on both sides\nsynthetic quake table, R²", "random rows", 0.87, "whole events", 0.65, True),
    ("Resampled before the split\nexplosion vs quake, F1", "before split", 0.995, "after split", 0.590, True),
    ("Test used to choose\nseismic search, macro-F1", "best search score", 0.845, "untouched test", 0.820, True),
]
fig, axes = plt.subplots(1, 4, figsize=(21, 6.0))
for ax, (title, la, a, lb, b, higher) in zip(axes, CASES):
    ax.bar([0, 1], [a, b], color=[VAL, TRAIN], width=0.6)
    for x, v in zip([0, 1], [a, b]):
        ax.text(x, v, f"{v:g}", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks([0, 1], [f"with leak\n({la})", f"without\n({lb})"])
    ax.set_ylim(0, max(a, b) * 1.18)
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold")
    ax.text(0.98, 0.98, "higher is better" if higher else "lower is better",
            transform=ax.transAxes, ha="right", va="top", color="#555555", fontsize=14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.03, right=0.995, top=0.84, bottom=0.2, wspace=0.28)
fig.savefig(OUT / "leakage_pairs.png", dpi=110)
print("wrote", OUT / "leakage_pairs.png")
