"""Slide figure: three ways a test set can lie outside the training domain.

Schematic built from synthetic numbers (no data download). Three panels:
  statistical      samples in a two-feature space; the shaded area is within the
                   95th percentile of training-to-training nearest-neighbor
                   distances, computed with training statistics only
  spatiotemporal   gauges by year: training and validation share gauges and years,
                   the test set is a later period and a gauge never seen in training
  physical         flood peaks: training and validation below a threshold, test
                   floods larger than any in training

Regenerate: pixi run python book/slides/2026/figscripts/lec10_partition_axes.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
})
INK, MUTED, TRAIN, VAL, TEST = "#26241f", "#6e675c", "#b9b2a6", "#1f6f6b", "#c0392b"
rng = np.random.default_rng(23)

fig, ax = plt.subplots(1, 3, figsize=(24, 7.2), gridspec_kw=dict(width_ratios=[1, 1.15, 1], wspace=0.28))

# 1. Statistical: distance to the training set in feature space
train = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 300)
val = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 40)
test_out = np.c_[rng.uniform(2.6, 4.2, 25), rng.uniform(-2.8, -0.8, 25)]
nn = NearestNeighbors(n_neighbors=2).fit(train)
d_tt = nn.kneighbors(train)[0][:, 1]                    # each training point to its nearest other one
thr = np.percentile(d_tt, 95)
gx, gy = np.meshgrid(np.linspace(-4, 4.6, 300), np.linspace(-4, 4, 300))
d_grid = nn.kneighbors(np.c_[gx.ravel(), gy.ravel()], n_neighbors=1)[0].reshape(gx.shape)
ax[0].contourf(gx, gy, d_grid <= thr * 2.5, levels=[0.5, 1.5], colors=["#e8f1ef"])
ax[0].contour(gx, gy, d_grid <= thr * 2.5, levels=[0.5], colors=[VAL], linewidths=1.5)
ax[0].scatter(*train.T, s=14, color=TRAIN, label="training")
ax[0].scatter(*val.T, s=30, color=VAL, label="validation")
ax[0].scatter(*test_out.T, s=46, color=TEST, marker="D", label="test, out of domain")
ax[0].set_xlabel("feature 1 (standardized)"); ax[0].set_ylabel("feature 2 (standardized)")
ax[0].set_title("Statistical", loc="left")
ax[0].legend(fontsize=15, loc="upper left", framealpha=0.9)
ax[0].set_xlim(-4, 4.6); ax[0].set_ylim(-4, 4)

# 2. Spatiotemporal: gauges by year
years = np.arange(2015, 2025)
gauges = ["A", "B", "C", "D", "E"]
grid = np.zeros((len(gauges), len(years)), dtype=int)   # 0 train, 1 validation, 2 test
grid[:4, :8][rng.random((4, 8)) < 0.2] = 1              # validation: random years at training gauges
grid[:, 8:] = 2                                         # test: a later period
grid[4, :] = 2                                          # test: a gauge never used in training
ax[1].imshow(grid, cmap=ListedColormap([TRAIN, VAL, TEST]), aspect="auto", vmin=0, vmax=2)
ax[1].set_xticks(range(len(years))); ax[1].set_xticklabels([str(y)[2:] for y in years])
ax[1].set_yticks(range(len(gauges))); ax[1].set_yticklabels([f"gauge {g}" for g in gauges])
ax[1].set_xlabel("year (20xx)")
ax[1].set_xticks(np.arange(-0.5, len(years)), minor=True); ax[1].set_yticks(np.arange(-0.5, len(gauges)), minor=True)
ax[1].grid(which="minor", color="white", lw=2); ax[1].tick_params(which="minor", length=0)
ax[1].set_title("Spatiotemporal", loc="left")
for lab, col, x in [("training", TRAIN, 0.0), ("validation", VAL, 0.34), ("test", TEST, 0.73)]:
    ax[1].text(x, -0.24, "■ " + lab, color=col, fontsize=16, transform=ax[1].transAxes, fontweight="bold")

# 3. Physical: flood size outside the training range
peaks = rng.lognormal(np.log(60), 0.7, 400)
cut = np.percentile(peaks, 90)
bins = np.logspace(np.log10(peaks.min()), np.log10(peaks.max()), 30)
inside = peaks[peaks <= cut]
is_val = rng.random(len(inside)) < 0.2
ax[2].hist(inside[~is_val], bins=bins, color=TRAIN, label="training")
ax[2].hist(inside[is_val], bins=bins, color=VAL, label="validation")
ax[2].hist(peaks[peaks > cut], bins=bins, color=TEST, label="test: larger floods")
ax[2].axvline(cut, color=INK, ls="--", lw=1.5)
ax[2].set_xscale("log")
ax[2].set_xlabel("flood peak (m³/s)"); ax[2].set_ylabel("number of floods")
ax[2].set_title("Physical", loc="left")
ax[2].legend(fontsize=15, loc="upper left", framealpha=0.9)

fig.text(0.125, -0.10, "Schematic from synthetic numbers. Statistical panel: shaded area within 2.5 × the 95th percentile "
         "of training nearest-neighbor distances.", fontsize=13, color=MUTED)
out = Path(__file__).resolve().parent.parent / "figs" / "lec10_ai_ready" / "partition_axes.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}; NN threshold (95th pct) = {thr:.3f}; physical cut = {cut:.0f} m3/s")
