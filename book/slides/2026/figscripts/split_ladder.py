"""Slide-scale re-plot of 3.8's three-way split visualization.

Extracted notebook figures carry print-size (~10pt) labels; this figscript
redraws the diagram at lecture scale (20pt+) for the slide deck. Figscripts
live beside the decks and regenerate with:
    pixi run python book/slides/2026/figscripts/split_ladder.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, TimeSeriesSplit

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 24, "axes.titleweight": "bold",
    "axes.labelsize": 21, "xtick.labelsize": 18, "ytick.labelsize": 18,
})

N = 712
X = np.zeros((N, 1))
splitters = [
    ("ShuffleSplit", ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)),
    ("KFold (shuffled)", KFold(n_splits=5, shuffle=True, random_state=0)),
    ("TimeSeriesSplit", TimeSeriesSplit(n_splits=5)),
]

fig, axes = plt.subplots(3, 1, figsize=(15, 8.2), sharex=True)
for ax, (name, sp) in zip(axes, splitters):
    for fold, (tr, va) in enumerate(sp.split(X)):
        ax.scatter(tr, np.full_like(tr, fold), marker="|", s=140, lw=1.6,
                   color="#1f77b4", label="train" if fold == 0 else None)
        ax.scatter(va, np.full_like(va, fold), marker="|", s=140, lw=1.6,
                   color="#ff7f0e", label="validation" if fold == 0 else None)
    ax.set_title(name, loc="left")
    ax.set_ylabel("fold")
    ax.set_yticks(range(5))
    ax.set_ylim(-0.6, 4.6)
axes[0].legend(loc="upper right", ncol=2, fontsize=18, framealpha=0.95)
axes[-1].set_xlabel("sample index (time order)")
fig.tight_layout(h_pad=0.8)

out = Path(__file__).resolve().parent.parent / "figs" / "3.8_robust_training" / "split_ladder_slide.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
