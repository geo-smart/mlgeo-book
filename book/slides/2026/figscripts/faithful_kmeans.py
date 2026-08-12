"""Slide figure: the Old Faithful two-regime discovery, raw data next to k-means.

Left panel: 271 consecutive eruption-duration pairs exactly as recorded — the
two regimes are visible before any algorithm runs. Right panel: the same data
standardized, colored by scikit-learn KMeans with k = 2 (random_state=0, the
notebook 3.3 run), centroids marked. Reads the same cached faithful.csv the
notebook downloads with pooch.

Regenerate: pixi run python book/slides/2026/figscripts/faithful_kmeans.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

path = pooch.retrieve(
    url="https://raw.githubusercontent.com/UW-MLGEO/MLGeo-dataset/main/data/faithful.csv",
    known_hash=None,
    fname="faithful.csv",
    path=pooch.os_cache("mlgeo"),
)
faithful = pd.read_csv(path)
data = StandardScaler().fit_transform(faithful[["current", "next"]])
km = KMeans(n_clusters=2, random_state=0).fit(data)

colors = ["#1f77b4", "#ff7f0e"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.6))

ax1.scatter(faithful["current"], faithful["next"], s=36, color="#3b3b3b")
ax1.set_xlabel("eruption duration (min)")
ax1.set_ylabel("next eruption duration (min)")
ax1.set_title("The data, as recorded", loc="left")
ax1.grid(True, alpha=0.4)

for k in (0, 1):
    m = km.labels_ == k
    ax2.scatter(data[m, 0], data[m, 1], s=36, color=colors[k],
                label=f"cluster {k}")
ax2.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            marker="X", s=420, color="black", label="centroids", zorder=5)
ax2.set_xlabel("eruption duration (standardized)")
ax2.set_ylabel("next duration (standardized)")
ax2.set_title("k-means, k = 2 — no labels used", loc="left")
ax2.legend(loc="lower left", fontsize=16, framealpha=0.95)
ax2.grid(True, alpha=0.4)

fig.suptitle("Old Faithful geyser, Yellowstone — 271 consecutive eruption pairs",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = (Path(__file__).resolve().parent.parent / "figs" / "3.3_clustering"
       / "faithful_kmeans_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
