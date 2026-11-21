"""Schematics for notebook 4.6, written to figs/mlgeo_4.6/:

 1. autoencoder.png  encoder, bottleneck, decoder around one synthetic spectrogram
                     (the dense autoencoder of 4.6: 64 x 64 = 4096 numbers -> 32 -> 4096); the output
                     panel is the input blurred, an illustration, not a trained reconstruction
 2. unet.png         the same encoder-decoder with skip connections at every level
                     (the shapes of the 4.6 convolutional encoder: 64 -> 32 -> 16 -> 8)

The spectrogram is the highest-SNR event of 200 from mlgeo_synth.seismogram_dataset (seed 0), computed
with the 4.6 parameters (nperseg 128, noverlap 83, 100 Hz).
Regenerate: pixi run python book/slides/2026/figscripts/ae_unet_cartoons.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.signal  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle  # noqa: E402

from mlgeo_synth import seismogram_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "mlgeo_4.6"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 18})
INK, MUTED = "#26241f", "#6e675c"
ENC, DEC, LAT, SKIP = "#0072B2", "#009E73", "#E69F00", "#D55E00"

# one event spectrogram, as in 4.6 section 1
X, y, metas = seismogram_dataset(n_events=200, n_noise=0, fs=100.0, duration_s=30.0, seed=0)
k = int(np.argmax([m["snr"] for m in metas]))
f, t, S = scipy.signal.spectrogram(X[k], fs=100.0, nperseg=128, noverlap=83)
S = np.log10(S[1:, :] + 1e-12)
lo, hi = np.percentile(S, [5, 99.7])
S = np.clip((S - lo) / (hi - lo), 0, 1)    # contrast stretch, for display only
import scipy.ndimage  # noqa: E402

blur = scipy.ndimage.gaussian_filter(S, 1.5)


def image(ax, x0, y0, w, h, img, label):
    ax.imshow(img, origin="lower", cmap="viridis", extent=(x0, x0 + w, y0, y0 + h), aspect="auto", zorder=2)
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=1.5, color=INK, zorder=3))
    ax.text(x0 + w / 2, y0 - 0.45, label, ha="center", va="top", fontsize=17, color=INK)


def arrow(ax, a, b, color=INK, lw=2.5, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=22, lw=lw, color=color,
                                 linestyle=ls, zorder=4))


# ------------------------------------------------------------ 1. autoencoder
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16); ax.set_ylim(-1.6, 5.4); ax.axis("off")
image(ax, 0.3, 0.6, 3.0, 3.0, S, "input: 64 × 64 = 4096 numbers")
ax.add_patch(Polygon([[4.2, 0.2], [7.0, 1.6], [7.0, 2.6], [4.2, 4.0]], color=ENC, alpha=0.85, zorder=2))
ax.text(5.6, 2.1, "encoder", ha="center", va="center", color="white", fontsize=20, weight="bold")
ax.add_patch(Rectangle((7.35, 1.4), 1.3, 1.4, color=LAT, zorder=2))
ax.text(8.0, 2.1, "32", ha="center", va="center", color="white", fontsize=24, weight="bold")
ax.text(8.0, 1.2, "bottleneck:\n32 numbers", ha="center", va="top", fontsize=17, color=INK)
ax.add_patch(Polygon([[9.0, 1.6], [11.8, 0.2], [11.8, 4.0], [9.0, 2.6]], color=DEC, alpha=0.85, zorder=2))
ax.text(10.4, 2.1, "decoder", ha="center", va="center", color="white", fontsize=20, weight="bold")
image(ax, 12.7, 0.6, 3.0, 3.0, blur, "output, for example: the input,\nrebuilt and blurred")
arrow(ax, (3.4, 2.1), (4.15, 2.1)); arrow(ax, (11.85, 2.1), (12.6, 2.1))
ax.text(8.0, 4.9, "the loss compares the output with the input: no label is needed",
        ha="center", fontsize=19, color=MUTED)
fig.savefig(FIGS / "autoencoder.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------ 2. U-Net
fig, ax = plt.subplots(figsize=(16, 7.5))
ax.set_xlim(0, 16); ax.set_ylim(-1.3, 7.2); ax.axis("off")
sizes, hs, ys = [64, 32, 16], [3.0, 2.0, 1.3], [3.4, 2.2, 1.1]
xe, xd = [0.8, 2.7, 4.6], [14.3, 12.4, 10.5]
for lev in range(3):
    for x, col in [(xe[lev], ENC), (xd[lev], DEC)]:
        ax.add_patch(Rectangle((x, ys[lev]), 0.9, hs[lev], color=col, alpha=0.9, zorder=2))
        ax.text(x + 0.45, ys[lev] - 0.12, f"{sizes[lev]}×{sizes[lev]}", ha="center", va="top", fontsize=15, color=INK)
    yy = ys[lev] + 0.8 * hs[lev]
    arrow(ax, (xe[lev] + 0.95, yy), (xd[lev] - 0.05, yy), color=SKIP, lw=2.5, ls="--")
xb, yb, hb = 7.55, 0.3, 0.8
ax.add_patch(Rectangle((xb, yb), 0.9, hb, color=LAT, zorder=2))
ax.text(xb + 0.45, yb - 0.12, "8×8", ha="center", va="top", fontsize=15, color=INK)
down = [(xe[0], ys[0]), (xe[1], ys[1]), (xe[2], ys[2]), (xb, yb)]
for (x0, y0), (x1, y1) in zip(down[:-1], down[1:]):
    arrow(ax, (x0 + 0.9, y0 + 0.2), (x1, y1 + 0.4), color=ENC, lw=2)
up = [(xb, yb), (xd[2], ys[2]), (xd[1], ys[1]), (xd[0], ys[0])]
for (x0, y0), (x1, y1) in zip(up[:-1], up[1:]):
    arrow(ax, (x0 + 0.9, y0 + 0.4), (x1, y1 + 0.2), color=DEC, lw=2)
ax.text(8.0, 6.7, "skip connections (dashed) carry the fine detail across;", ha="center", fontsize=19, color=SKIP)
ax.text(8.0, 6.15, "the path through the bottleneck carries the context", ha="center", fontsize=19, color=MUTED)
ax.text(2.6, -1.0, "encoder", ha="center", fontsize=20, color=ENC, weight="bold")
ax.text(13.4, -1.0, "decoder", ha="center", fontsize=20, color=DEC, weight="bold")
ax.text(8.0, -1.0, "bottleneck", ha="center", fontsize=20, color=LAT, weight="bold")
fig.savefig(FIGS / "unet.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("wrote", sorted(p.name for p in FIGS.glob("*.png")))
