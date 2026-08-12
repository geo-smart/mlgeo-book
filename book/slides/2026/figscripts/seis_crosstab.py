"""Slide figure: k-means clusters vs true seismic source types (notebook 3.3).

Re-plots the executed notebook's cluster-vs-label crosstab at lecture scale.
The counts are hard-coded from the executed 3.3 output (KMeans k=4,
random_state=42, on 15 PCs of the standardized 61 features): edit them only
after re-running the notebook. Data: 4000 Pacific Northwest seismic events,
Zenodo DOI 10.5281/zenodo.14025693.

Regenerate: pixi run python book/slides/2026/figscripts/seis_crosstab.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

sources = ["earthquake", "explosion", "noise", "surface event"]
counts = np.array([   # rows: cluster 0..3 — executed notebook 3.3 crosstab
    [141, 732, 14, 794],
    [71, 52, 698, 125],
    [764, 117, 41, 48],
    [24, 99, 247, 33],
])

fig, ax = plt.subplots(figsize=(10.5, 6.8))
im = ax.imshow(counts, cmap="Blues")
ax.set_xticks(range(4))
ax.set_xticklabels(sources, rotation=20, ha="right")
ax.set_yticks(range(4))
ax.set_yticklabels([f"cluster {i}" for i in range(4)])
for i in range(4):
    for j in range(4):
        ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=20,
                color="white" if counts[i, j] > counts.max() / 2 else "black")
ax.set_title("Clusters found without labels vs analyst labels", loc="left")
fig.colorbar(im, ax=ax, label="number of events")
fig.suptitle("4000 Pacific Northwest seismic events — Zenodo 10.5281/zenodo.14025693",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = (Path(__file__).resolve().parent.parent / "figs" / "3.3_clustering"
       / "seis_crosstab_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
