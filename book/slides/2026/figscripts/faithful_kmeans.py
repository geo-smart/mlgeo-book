"""Slide figure: the Old Faithful two-regime discovery, raw data next to k-means.

Left panel: 271 consecutive eruption-duration pairs as recorded. Middle and
right panels: the same pairs standardized and grouped by scikit-learn KMeans
with k = 2 and k = 4 (k-means++, n_init=20, random_state=42, as in notebook
3.3), centroids marked. Prints the silhouette score of each. Reads the same cached faithful.csv the
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
from sklearn.metrics import silhouette_score  # noqa: E402
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
fits = {k: KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42).fit(data) for k in (2, 4)}
for k, km in fits.items():
    print(f"k = {k}: silhouette {silhouette_score(data, km.labels_):.3f}")

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
ax1 = axes[0]
ax1.scatter(faithful["current"], faithful["next"], s=30, color="#3b3b3b")
ax1.set_xlabel("eruption duration (min)")
ax1.set_ylabel("next eruption duration (min)")
ax1.set_title("The pairs, as recorded", loc="left")
ax1.grid(True, alpha=0.4)
for ax, (k, km) in zip(axes[1:], fits.items()):
    for j in range(k):
        m = km.labels_ == j
        ax.scatter(data[m, 0], data[m, 1], s=30, color=colors[j])
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               marker="X", s=380, color="black", zorder=5)
    ax.set_xlabel("duration (standardized)")
    ax.set_ylabel("next duration (standardized)")
    ax.set_title(f"k-means, k = {k}, no labels", loc="left")
    ax.grid(True, alpha=0.4)

fig.suptitle("Old Faithful geyser, Yellowstone: 271 consecutive eruption pairs (Azzalini & Bowman 1990)",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = (Path(__file__).resolve().parent.parent / "figs" / "3.3_clustering"
       / "faithful_kmeans_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
