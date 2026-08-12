"""Slide figure: the elbow criterion working, then failing, on the same data.

Panel 1: within-cluster scatter vs k for the standardized Old Faithful data —
a clean kink at k = 2. Panel 2: the notebook 3.3 stress test — the same data
shrunk toward the origin (alpha * sign(x) * |x|^2, alpha = 0.5) so the
clusters bleed together. Panel 3: the elbow curve for the shrunken data — the
kink is gone and the choice of k is ambiguous. Mirrors notebook 3.3 with
scikit-learn KMeans in place of the from-scratch loop.

Regenerate: pixi run python book/slides/2026/figscripts/elbow_limits.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
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
shrunk = 0.5 * np.sign(data) * np.abs(data) ** 2.0   # notebook 3.3 stress test

ks = np.arange(1, 9)


def elbow(x):
    return [KMeans(n_clusters=k, random_state=0).fit(x).inertia_ / len(x)
            for k in ks]


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.6))

ax1.plot(ks, elbow(data), marker="o", color="#1f77b4", linewidth=2.5)
ax1.axvline(2, color="#b3402a", linestyle="--", linewidth=2)
ax1.set_title("Clear kink at k = 2", loc="left")
ax1.set_xlabel("number of clusters k")
ax1.set_ylabel("within-cluster scatter")
ax1.grid(True, alpha=0.4)

ax2.scatter(shrunk[:, 0], shrunk[:, 1], s=22, color="#3b3b3b")
ax2.set_title("Clusters pushed closer", loc="left")
ax2.set_xlabel("eruption duration (shrunk)")
ax2.set_ylabel("next duration (shrunk)")
ax2.grid(True, alpha=0.4)

ax3.plot(ks, elbow(shrunk), marker="o", color="#1f77b4", linewidth=2.5)
ax3.set_title("Kink gone: k is a guess", loc="left")
ax3.set_xlabel("number of clusters k")
ax3.set_ylabel("within-cluster scatter")
ax3.grid(True, alpha=0.4)

fig.suptitle("Old Faithful geyser, Yellowstone (panels 2–3: shrunken stress test, notebook 3.3)",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.4)

out = (Path(__file__).resolve().parent.parent / "figs" / "3.3_clustering"
       / "elbow_limits_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
