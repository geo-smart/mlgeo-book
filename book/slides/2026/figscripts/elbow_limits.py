"""Slide figure: the elbow criterion working, then failing, on the same data.

Panel 1: inertia (divided by its value at k = 1) vs k for the standardized
Old Faithful pairs, as recorded and blurred. Panel 2: the notebook 3.3 stress
test, the same pairs plus Gaussian scatter of 1 standardized unit
(default_rng(42)). Panel 3: silhouette score vs k for both. KMeans settings as
in notebook 3.3 (k-means++, n_init=20, random_state=42).

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
blurred = data + np.random.default_rng(42).normal(0, 1.0, data.shape)   # notebook 3.3 stress test
ks = np.arange(1, 9)
sk = np.arange(2, 7)


def diagnostics(x):
    fits = {k: KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42).fit(x) for k in ks}
    iner = np.array([fits[k].inertia_ for k in ks])
    sil = np.array([silhouette_score(x, fits[k].labels_) for k in sk])
    return iner / iner[0], sil


(i_rec, s_rec), (i_blur, s_blur) = diagnostics(data), diagnostics(blurred)
print("silhouette as recorded:", dict(zip(sk, s_rec.round(3))))
print("silhouette blurred:   ", dict(zip(sk, s_blur.round(3))))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6.0))
for ax, y, x, name in [(ax1, i_rec, ks, "as recorded"), (ax1, i_blur, ks, "blurred"),
                       (ax3, s_rec, sk, "as recorded"), (ax3, s_blur, sk, "blurred")]:
    style = dict(marker="o", color="#1f77b4") if name == "as recorded" else dict(marker="s", color="#b3402a", linestyle="--")
    ax.plot(x, y, linewidth=2.5, markersize=9, label=name, **style)
ax1.set_title("Elbow: inertia vs k", loc="left")
ax1.set_xlabel("number of clusters k")
ax1.set_ylabel("inertia / inertia at k = 1")
ax1.legend(fontsize=17)
ax1.grid(True, alpha=0.4)
ax2.scatter(blurred[:, 0], blurred[:, 1], s=26, color="#3b3b3b")
ax2.set_title("The same pairs, blurred", loc="left")
ax2.set_xlabel("duration (standardized)")
ax2.set_ylabel("next duration (standardized)")
ax2.grid(True, alpha=0.4)
ax3.set_title("Silhouette score vs k", loc="left")
ax3.set_xlabel("number of clusters k")
ax3.set_ylabel("mean silhouette")
ax3.set_ylim(0, 0.8)
ax3.grid(True, alpha=0.4)

fig.suptitle("Old Faithful geyser, Yellowstone: the stress test of notebook 3.3 adds scatter of 1 standardized unit (1.14 min)",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.4)

out = (Path(__file__).resolve().parent.parent / "figs" / "3.3_clustering"
       / "elbow_limits_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
