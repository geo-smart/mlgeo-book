"""Cartoons of the convolutional block for lecture 22.

Schematics with small worked numbers, one concept per figure, written to
figs/lec22_cnn/:

 1. cnn_anatomy.png         the whole network: input, three blocks, head, shapes
 2. conv1d_cartoon.png      Conv1D by hand: 10 inputs, kernel of 3, 8 outputs
 3. kernel_size.png         kernels of 7, 21 and 81 weights on a 5 Hz arrival (synthetic)
 4. conv2d_cartoon.png      Conv2D by hand: 6x6 input, 3x3 kernel, 4x4 output
 5. channels_kernels.png    channel against kernel: 3 components in, one filter = 3 kernels
 6. weights_count.png       weights of the 4.3 detector, layer by layer, against a dense layer
 7. activation.png          ReLU, and a feature map before and after it
 8. pooling_cartoon.png     max and average pooling, 1-D and 2-D, and what pooling is for
 9. receptive_cone.png      how stacking grows what one unit sees
10. topology.png            wide and shallow, funnel, long and skinny (numbers of notebook 4.3, 2.5)
11. flatten_gap.png         flatten against global average pooling
12. heads.png               the same block, a classification head or a regression head

Every number on a figure is computed here (and written to numbers.json).
Regenerate: pixi run python book/slides/2026/figscripts/lec22_cnn_cartoons.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lec22_cnn"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 20, "axes.titlesize": 24, "axes.titleweight": "bold"})
INK, MUTED, LIGHT = "#26241f", "#6e675c", "#e8e4dc"
IN_C, K_C, OUT_C, POS_C, NEG_C = "#8a857c", "#E69F00", "#0072B2", "#009E73", "#D55E00"
CH_C = ["#0072B2", "#009E73", "#D55E00"]
numbers = {}


def canvas(w=16, h=7.5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def cells(ax, x0, y0, values, color="white", ec=INK, w=1.0, h=1.0, fs=20, fmt="{:g}", lw=1.5, alpha=1.0, tc=INK):
    for i, v in enumerate(values):
        ax.add_patch(Rectangle((x0 + i * w, y0), w, h, fc=color, ec=ec, lw=lw, alpha=alpha))
        if v is not None:
            ax.text(x0 + i * w + w / 2, y0 + h / 2, fmt.format(v), ha="center", va="center", fontsize=fs, color=tc)


def grid(ax, x0, y0, M, color="white", fs=20, fmt="{:g}", hl=None, hl_color=K_C, cmap=None, vmax=None):
    n, m = M.shape
    for r in range(n):
        for c in range(m):
            fc = color
            if cmap is not None:
                fc = plt.get_cmap(cmap)(0.5 + 0.5 * M[r, c] / vmax)
            ax.add_patch(Rectangle((x0 + c, y0 + n - 1 - r), 1, 1, fc=fc, ec=INK, lw=1.2))
            ax.text(x0 + c + 0.5, y0 + n - 1 - r + 0.5, fmt.format(M[r, c]), ha="center", va="center", fontsize=fs)
    if hl is not None:
        r0, c0, hh, ww = hl
        ax.add_patch(Rectangle((x0 + c0, y0 + n - r0 - hh), ww, hh, fill=False, ec=hl_color, lw=5))


def arrow(ax, p, q, color=INK, lw=2.5, style="-|>", ms=22, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def box(ax, x, y, w, h, text, fc, fs=20, ec="none", tc="white", weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight)


def stack(ax, x, y, w, h, n, color, dx=0.12, dy=0.12):
    """A stack of n feature maps drawn as offset rectangles (front one at x, y)."""
    for i in reversed(range(n)):
        ax.add_patch(Rectangle((x + i * dx, y + i * dy), w, h, fc=color, ec="white", lw=1.0, alpha=0.9))


def save(fig, name):
    fig.savefig(FIGS / name, dpi=110, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ 1. anatomy
detector = nn.Sequential(
    nn.Conv1d(1, 8, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
    nn.Conv1d(8, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
    nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 2))
x = torch.zeros(1, 1, 3000)
shapes = []
for layer in detector:
    x = layer(x)
    shapes.append(tuple(x.shape[1:]))
numbers["anatomy"] = {"shapes_after_each_layer": shapes, "weights": n_params(detector)}

fig, ax = canvas(17, 7.5)
ax.set_xlim(0, 34); ax.set_ylim(-1.5, 13)
# input
tt = np.linspace(0, 1, 300)
wave = 0.15 * np.random.default_rng(1).standard_normal(300) + np.where(tt > 0.45, np.sin(2 * np.pi * 14 * tt) * np.exp(-(tt - 0.45) * 6), 0)
ax.plot(0.5 + 3.5 * tt, 6 + 1.2 * wave, color=IN_C, lw=1.5)
ax.text(2.25, 4.6, "input\n1 × 3000\n(30 s, 100 Hz)", ha="center", va="top", fontsize=18)
# three blocks
xs, cols = [6.0, 12.0, 17.2], ["#9ecae1", "#4292c6", "#08519c"]
lens, chans = [750, 187, 46], [8, 16, 32]
for k, (x0, c, L, ch) in enumerate(zip(xs, cols, lens, chans)):
    w = 3.4 * (L / 750) ** 0.45
    stack(ax, x0, 5.0, w, 2.2, min(ch // 2, 12), c, dx=0.13, dy=0.13)
    ax.text(x0 + w / 2 + 0.6, 3.9, f"{ch} × {L}", ha="center", fontsize=19, weight="bold")
    ax.text(x0 + w / 2 + 0.6, 2.9, f"block {k + 1}", ha="center", fontsize=17, color=MUTED)
    arrow(ax, (x0 - 1.6 if k == 0 else xs[k - 1] + 3.4 * (lens[k - 1] / 750) ** 0.45 + 1.9, 6.1), (x0 - 0.15, 6.1))
ax.text(9.5, 10.3, "convolution → ReLU → max pooling, three times", ha="center", fontsize=19)
ax.annotate("", xy=(4.9, 9.4), xytext=(21.0, 9.4), arrowprops=dict(arrowstyle="-", lw=2.5, color=INK))
ax.text(12.9, 11.6, "the convolutional block: finds patterns", ha="center", fontsize=22, weight="bold", color="#08519c")
# head
gx = 23.3
arrow(ax, (20.2, 6.1), (gx - 0.15, 6.1))
cells(ax, gx, 3.4, [None] * 1, w=0.7, h=0.6)
for i in range(8):
    ax.add_patch(Rectangle((gx, 3.4 + i * 0.7), 0.7, 0.7, fc="#c6dbef", ec=INK, lw=1))
ax.text(gx + 0.35, 2.5, "32", ha="center", fontsize=19, weight="bold")
ax.text(gx + 0.35, 1.5, "average\nover time", ha="center", fontsize=16, color=MUTED, va="top")
ox = 27.0
for i, (lab, p) in enumerate([("event", 0.91), ("noise", 0.09)]):
    yy = 7.0 - i * 2.2
    ax.add_patch(plt.Circle((ox, yy), 0.55, fc=POS_C if i == 0 else NEG_C, ec=INK))
    for j in range(8):
        ax.plot([gx + 0.7, ox - 0.55], [3.75 + j * 0.7, yy], color=MUTED, lw=0.6)
    ax.text(ox + 0.9, yy, f"{lab}", va="center", fontsize=19)
    ax.add_patch(Rectangle((ox + 3.6, yy - 0.35), 2.6 * p, 0.7, fc=POS_C if i == 0 else NEG_C))
    ax.text(ox + 3.7 + 2.6 * p, yy, f" {p:.2f}", va="center", fontsize=18)
ax.text(ox + 4.9, 8.5, "softmax, for example", ha="center", fontsize=17, color=MUTED)
ax.text(ox - 0.2, 2.5, "fully connected\n32 → 2", ha="center", fontsize=17, color=MUTED, va="top")
ax.annotate("", xy=(22.6, 9.4), xytext=(33.6, 9.4), arrowprops=dict(arrowstyle="-", lw=2.5, color=INK))
ax.text(28.1, 11.6, "the head: decides", ha="center", fontsize=22, weight="bold", color=POS_C)
ax.text(17, -1.0, f"the earthquake detector of notebook 4.3: {n_params(detector):,} weights in all",
        ha="center", fontsize=18, color=MUTED)
save(fig, "cnn_anatomy.png")

# ------------------------------------------------------------------ 2. conv1d by hand
xin = np.array([0, 1, 2, 1, 0, 0, -1, -2, -1, 0])
wk = np.array([1, 2, 1])
yout = np.correlate(xin, wk, mode="valid")
pos = 1
numbers["conv1d"] = {"input": xin.tolist(), "kernel": wk.tolist(), "output": yout.tolist(), "position_shown": pos}
fig, ax = canvas(16, 7.5)
ax.set_xlim(-3.2, 13.5); ax.set_ylim(-3.4, 8.2)
ax.text(-3.0, 5.5, "input x", fontsize=21, va="center", weight="bold")
cells(ax, 0, 5.0, xin, color="#f2f0ea")
for i in range(len(xin)):
    ax.text(i + 0.5, 6.25, f"x{i + 1}" if i < 9 else "x10", ha="center", fontsize=14, color=MUTED)
ax.add_patch(Rectangle((pos, 5.0), 3, 1, fill=False, ec=K_C, lw=5))
ax.text(-3.0, 2.7, "kernel w", fontsize=21, va="center", weight="bold", color="#b07800")
cells(ax, pos, 2.2, wk, color="#fde7b0")
for j in range(3):
    arrow(ax, (pos + j + 0.5, 4.95), (pos + j + 0.5, 3.25), color=K_C, lw=2, ms=16)
    ax.text(pos + j + 0.85, 4.1, "×", fontsize=20, color="#b07800")
ax.text(pos + 3.4, 2.7, f"= {xin[pos]}×{wk[0]} + {xin[pos + 1]}×{wk[1]} + {xin[pos + 2]}×{wk[2]} = {yout[pos]}",
        fontsize=22, va="center")
ax.text(-3.0, -0.5, "output y", fontsize=21, va="center", weight="bold", color=OUT_C)
cells(ax, 1, -1.0, yout, color="#d6e6f4")
ax.add_patch(Rectangle((1 + pos, -1.0), 1, 1, fill=False, ec=OUT_C, lw=5))
arrow(ax, (pos + 1.5, 2.15), (pos + 1.5, 0.05), color=OUT_C, lw=2.5)
ax.text(5.0, 7.4, "multiply, add, write one number; slide one step; repeat", ha="center", fontsize=22)
ax.text(5.5, -2.6, f"10 inputs, kernel of 3, no padding → 10 − 3 + 1 = {len(yout)} outputs.   "
        "The same 3 weights at every position.", ha="center", fontsize=18, color=MUTED)
save(fig, "conv1d_cartoon.png")

# ------------------------------------------------------------------ 3. kernel size
fs = 100.0
t = np.arange(0, 6, 1 / fs)
rng = np.random.default_rng(3)


def packet(t, t0, f0=5.0, width=0.25):
    return np.cos(2 * np.pi * f0 * (t - t0)) * np.exp(-((t - t0) / width) ** 2)


rec = packet(t, 3.0) + 0.35 * rng.standard_normal(t.size)
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(4, 2, width_ratios=[1, 5], hspace=0.65, wspace=0.05)
a0 = fig.add_subplot(gs[0, 1])
a0.plot(t, rec, color=IN_C, lw=1)
a0.set_title("record: a 5 Hz arrival (one period = 0.2 s = 20 samples) in noise", loc="left", fontsize=20)
a0.set_yticks([]); a0.set_xticks([])
fig.add_subplot(gs[0, 0]).axis("off")
peaks = {}
for r, n in enumerate([7, 21, 81]):
    tk = (np.arange(n) - n // 2) / fs
    kern = packet(tk, 0.0)
    kern -= kern.mean()
    kern /= np.linalg.norm(kern)
    out = np.correlate(rec, kern, mode="same")
    ak = fig.add_subplot(gs[r + 1, 0])
    ak.plot(np.arange(n), kern, "o-" if n < 30 else "-", color=K_C, lw=2.5, ms=6)
    ak.set_xlim(-2, 83); ak.axis("off")
    ak.text(40, ak.get_ylim()[1] * 1.15, f"{n} weights\n{n / fs:.2f} s", ha="center", fontsize=17, va="bottom")
    ao = fig.add_subplot(gs[r + 1, 1], sharex=a0)
    ao.plot(t, out, color=OUT_C, lw=1.2)
    ao.set_yticks([])
    contrast = np.abs(out[(t > 2.6) & (t < 3.4)]).max() / np.abs(out[(t < 2.2) | (t > 3.8)]).std()
    peaks[n] = float(contrast)
    ao.set_title(f"output: peak on the arrival / spread of the noise = {contrast:.1f}", loc="left", fontsize=18)
ao.set_xlabel("time (s)")
numbers["kernel_size"] = {"peak_over_noise_spread": peaks}
save(fig, "kernel_size.png")

# ------------------------------------------------------------------ 4. conv2d by hand
X2 = np.array([[1, 2, 1, 2, 1, 2],
               [2, 1, 2, 1, 2, 1],
               [1, 2, 2, 1, 2, 1],
               [7, 8, 8, 7, 8, 8],
               [8, 7, 8, 8, 7, 8],
               [8, 8, 7, 8, 8, 7]])
K2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
Y2 = np.array([[np.sum(K2 * X2[i:i + 3, j:j + 3]) for j in range(4)] for i in range(4)])
r0, c0 = 1, 1
numbers["conv2d"] = {"input": X2.tolist(), "kernel": K2.tolist(), "output": Y2.tolist(), "window": [r0, c0]}
fig, ax = canvas(16, 7.5)
ax.set_xlim(-0.5, 22); ax.set_ylim(-2.2, 8.2)
ax.text(3, 6.7, "input: temperature (°C)\ncold north, warm south", ha="center", fontsize=19, weight="bold")
grid(ax, 0, 0, X2, color="#f2f0ea", hl=(r0, c0, 3, 3))
ax.text(9.5, 6.7, "kernel: 3 × 3 weights", ha="center", fontsize=19, weight="bold", color="#b07800")
grid(ax, 8, 2.0, K2, color="#fde7b0")
ax.text(9.5, 1.2, "×  each cell, then add", ha="center", fontsize=19)
ax.text(18, 6.7, "output: 4 × 4", ha="center", fontsize=19, weight="bold", color=OUT_C)
grid(ax, 16, 1.0, Y2, cmap="PuOr_r", vmax=np.abs(Y2).max() * 1.3, hl=(r0, c0, 1, 1), hl_color=OUT_C)
arrow(ax, (4.05, 3.5), (7.9, 3.5), color=K_C)
arrow(ax, (11.1, 3.5), (15.9, 3.5), color=OUT_C)
win = X2[r0:r0 + 3, c0:c0 + 3]
ax.text(10.8, -1.4, f"−({win[0, 0]} + 2×{win[0, 1]} + {win[0, 2]}) + ({win[2, 0]} + 2×{win[2, 1]} + {win[2, 2]}) = {Y2[r0, c0]}"
        "     large where it is warmer to the south: the front", ha="center", fontsize=18)
save(fig, "conv2d_cartoon.png")

# ------------------------------------------------------------------ 5. channels and kernels
fig, ax = canvas(17, 8)
ax.set_xlim(0, 34); ax.set_ylim(-2.5, 14)
rng = np.random.default_rng(5)
tt = np.linspace(0, 1, 200)
names = ["N", "E", "Z"]
for i, (nm, c) in enumerate(zip(names, CH_C)):
    y0 = 9.5 - i * 3.2
    ax.add_patch(Rectangle((0.5, y0 - 1.1), 7, 2.2, fc="white", ec=c, lw=2.5))
    ax.plot(0.7 + 6.6 * tt, y0 + 0.9 * (0.2 * rng.standard_normal(200) + np.where(tt > 0.5, np.sin(30 * tt + i) * np.exp(-(tt - 0.5) * 5), 0)), color=c, lw=1.3)
    ax.text(0.2, y0, nm, ha="right", va="center", fontsize=22, weight="bold", color=c)
    # one kernel per input channel
    for j in range(5):
        ax.add_patch(Rectangle((11 + j * 0.8, y0 - 0.4), 0.8, 0.8, fc=c, ec="white", alpha=0.35 + 0.12 * j))
    arrow(ax, (7.7, y0), (10.8, y0), color=c, lw=2)
    arrow(ax, (15.2, y0), (18.6, 6.3), color=c, lw=2)
ax.text(4, 12.3, "input: 3 channels\n(N, E, Z components)", ha="center", fontsize=19, weight="bold")
ax.text(13, 12.3, "one filter =\n3 kernels of 5 weights", ha="center", fontsize=19, weight="bold", color="#b07800")
ax.add_patch(plt.Circle((19.3, 6.3), 0.75, fc="white", ec=INK, lw=2.5))
ax.text(19.3, 6.3, "+", ha="center", va="center", fontsize=30)
ax.text(19.3, 4.8, "+ bias", ha="center", fontsize=17)
arrow(ax, (20.1, 6.3), (22.6, 6.3))
ax.add_patch(Rectangle((22.8, 5.2), 7, 2.2, fc="#d6e6f4", ec=OUT_C, lw=2.5))
ax.text(26.3, 6.3, "1 output channel\n(a feature map)", ha="center", va="center", fontsize=17)
stack(ax, 22.8, 0.0, 7, 1.6, 8, OUT_C, dx=0.15, dy=0.18)
ax.text(26.6, -1.4, "8 filters → 8 output channels", ha="center", fontsize=19, weight="bold", color=OUT_C)
nw = 8 * (3 * 5 + 1)
numbers["channels"] = {"filters": 8, "in_channels": 3, "kernel": 5, "weights": nw}
ax.text(9.5, -1.0, "kernel: weights along time (size 5)\nchannel: one map, in or out (a component, a band)",
        ha="center", fontsize=18, color=MUTED)
ax.text(26.5, 12.3, f"weights: 8 × (3 × 5 + 1) = {nw}", ha="center", fontsize=20, weight="bold")
save(fig, "channels_kernels.png")

# ------------------------------------------------------------------ 6. counting weights
convs = [m for m in detector if isinstance(m, (nn.Conv1d, nn.Linear))]
rows = []
for m in convs:
    if isinstance(m, nn.Conv1d):
        lab = f"conv {m.in_channels}→{m.out_channels}, kernel {m.kernel_size[0]}"
        formula = f"{m.out_channels} × ({m.in_channels} × {m.kernel_size[0]} + 1)"
    else:
        lab = f"linear {m.in_features}→{m.out_features}"
        formula = f"{m.out_features} × ({m.in_features} + 1)"
    rows.append((lab, formula, n_params(m)))
dense = 64 * (3000 + 1)
c2d = 64 * (3 * 6 * 6 + 1)
numbers["weights"] = {"rows": rows, "total": n_params(detector), "dense_3000_to_64": dense, "conv2d_3_64_k6": c2d}
fig, ax = plt.subplots(figsize=(16, 7.2))
labels = [r[0] for r in rows] + ["the CNN, all layers", "one dense layer: 3000 samples → 64"]
vals = [r[2] for r in rows] + [n_params(detector), dense]
colors = [OUT_C] * len(rows) + [INK, NEG_C]
yy = np.arange(len(vals))[::-1]
ax.barh(yy, vals, color=colors, height=0.62)
ax.set_xscale("log"); ax.set_xlim(30, 3e6)
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=19)
for y, v, r in zip(yy, vals, [r[1] for r in rows] + ["", "64 × (3000 + 1)"]):
    ax.text(v * 1.15, y, f"{v:,}" + (f"   = {r}" if r else ""), va="center", fontsize=18)
ax.set_xlabel("number of weights (log scale)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("weights = filters × (input channels × kernel size + 1)", loc="left", fontsize=23)
save(fig, "weights_count.png")

# ------------------------------------------------------------------ 7. activation
fig = plt.figure(figsize=(16, 6.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.4], hspace=0.6, wspace=0.18)
a = fig.add_subplot(gs[:, 0])
z = np.linspace(-3, 3, 200)
a.plot(z, np.maximum(z, 0), lw=4, color=POS_C)
a.axhline(0, color=INK, lw=0.8); a.axvline(0, color=INK, lw=0.8)
a.set_title("ReLU: max(0, z)", loc="left"); a.set_xlabel("input z"); a.set_ylabel("output")
for s in ("top", "right"):
    a.spines[s].set_visible(False)
tk = (np.arange(81) - 40) / fs
kern = packet(tk, 0.0)
kern -= kern.mean()
fm = np.correlate(rec, kern / np.linalg.norm(kern), mode="same")
b1, b2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
b1.fill_between(t, fm, where=fm > 0, color=OUT_C, lw=0); b1.fill_between(t, fm, where=fm < 0, color=NEG_C, lw=0)
b1.set_title("feature map: positive (blue) and negative (red)", loc="left", fontsize=19)
b2.fill_between(t, np.maximum(fm, 0), color=POS_C, lw=0)
b2.set_title("after ReLU: only where the pattern matches", loc="left", fontsize=19)
for b in (b1, b2):
    b.set_yticks([]); b.set_xlim(0, 6)
    for s in ("top", "right", "left"):
        b.spines[s].set_visible(False)
b2.set_xlabel("time (s)")
save(fig, "activation.png")

# ------------------------------------------------------------------ 8. pooling
v1 = np.array([0, 3, 1, 0, 0, 0, 5, 2])
mp = v1.reshape(-1, 2).max(1)
ap = v1.reshape(-1, 2).mean(1)
M2 = np.array([[1, 0, 2, 3], [4, 6, 6, 8], [3, 1, 1, 0], [1, 2, 2, 4]])
mp2 = M2.reshape(2, 2, 2, 2).max(axis=(1, 3))
fig, ax = canvas(17, 7.5)
ax.set_xlim(-5.6, 25); ax.set_ylim(-3.6, 8.5)
ax.text(-5.5, 6.5, "after ReLU", fontsize=19, va="center")
cells(ax, 0, 6, v1, color="#e5f5ec")
for k in range(4):
    ax.add_patch(Rectangle((2 * k, 6), 2, 1, fill=False, ec=INK, lw=3.5))
ax.text(-5.5, 3.6, "max, by 2", fontsize=19, va="center", color=OUT_C, weight="bold")
cells(ax, 1, 3.1, mp, color="#d6e6f4", w=1.5)
ax.text(-5.5, 1.5, "average, by 2", fontsize=19, va="center", color="#7a4a9c", weight="bold")
cells(ax, 1, 1.0, ap, color="#ece2f4", w=1.5, fmt="{:.1f}")
for k in range(4):
    arrow(ax, (2 * k + 1, 5.95), (1.75 + 1.5 * k, 4.15), lw=1.5, ms=14)
ax.text(4, 8.0, "1-D: 8 values → 4", ha="center", fontsize=21, weight="bold")
ax.text(15.5, 8.0, "2-D: 4 × 4 → 2 × 2", ha="center", fontsize=21, weight="bold")
grid(ax, 12, 1.5, M2, color="#e5f5ec")
for r in range(2):
    for c in range(2):
        ax.add_patch(Rectangle((12 + 2 * c, 1.5 + 2 * (1 - r)), 2, 2, fill=False, ec=INK, lw=3.5))
arrow(ax, (16.2, 3.5), (18.3, 3.5))
grid(ax, 18.5, 2.5, mp2, color="#d6e6f4")
ax.text(19.5, 1.7, "max", ha="center", fontsize=18, color=OUT_C)
ax.text(11, -1.4, "what pooling is for:  shorter maps (less to compute)  ·  a small shift changes little  ·"
        "  the next kernel sees twice as far", ha="center", fontsize=18, color=INK)
ax.text(11, -2.8, "no weights to learn", ha="center", fontsize=18, color=MUTED)
numbers["pooling"] = {"input": v1.tolist(), "max": mp.tolist(), "avg": ap.tolist(), "max2d": mp2.tolist()}
save(fig, "pooling_cartoon.png")

# ------------------------------------------------------------------ 9. receptive field cone
fig, ax = canvas(16, 7.5)
ax.set_xlim(-6, 24); ax.set_ylim(-1.5, 11)
levels = [("input", 20, 1), ("convolution, kernel 3", 18, 1), ("max pooling, by 2", 9, 2), ("convolution, kernel 3", 7, 2)]
ys = [0, 3, 6, 9]
# cone of the top unit (index 3 of the last layer)
deps = {3: {3}}
deps[2] = {3, 4, 5}
deps[1] = {6, 7, 8, 9, 10, 11}
deps[0] = set(range(6, 14))
offsets = {0: 0.0, 1: 1.0, 2: 1.0, 3: 3.0}
for li, ((name, n, step), y) in enumerate(zip(levels, ys)):
    x0 = offsets[li]
    for i in range(n):
        on = i in deps[li]
        ax.add_patch(Rectangle((x0 + i * step, y), step * 0.92, 0.9, fc=(K_C if li == 3 and on else OUT_C if on else LIGHT),
                               ec="white", lw=1))
    ax.text(-0.6, y + 0.45, name, ha="right", va="center", fontsize=18)
rf = len(deps[0])
ax.text(11, -1.1, f"one unit at the top sees {rf} input samples: each convolution adds kernel − 1, "
        "each pooling doubles the step", ha="center", fontsize=18)
numbers["cone"] = {"receptive_field": rf}
save(fig, "receptive_cone.png")

# ------------------------------------------------------------------ 10. topology
nets = {
    "wide and shallow": [(128, 750)],
    "funnel": [(8, 750), (16, 187), (32, 46)],
    "long and skinny": [(8, 1500), (8, 750), (8, 375), (8, 187), (8, 93), (8, 46), (8, 23), (8, 11)],
}
facts = {"wide and shallow": ("1 layer, 128 filters, kernel 31", 4096, 0.34),
         "funnel": ("3 layers, 8-16-32 filters, kernel 7", 4592, 1.90),
         "long and skinny": ("8 layers, 8 filters, kernel 7", 3256, 17.86)}
numbers["topology"] = facts
fig, axes = plt.subplots(3, 1, figsize=(13, 8.5))
for ax, (name, layers) in zip(axes, nets.items()):
    ax.set_xlim(-0.5, 21); ax.set_ylim(-0.3, 5.6); ax.axis("off")
    x = 0.0
    ax.add_patch(Rectangle((x, 1.0), 4.0, 0.25, fc=IN_C))
    x += 4.6
    for ch, L in layers:
        w = 4.0 * (L / 3000) ** 0.5
        h = 0.25 * ch ** 0.5 * 1.25
        ax.add_patch(Rectangle((x, 1.0), w, h, fc=OUT_C, ec="white", alpha=0.85))
        x += w + 0.35
    desc, nw, rfs = facts[name]
    ax.text(0, 5.2, name, fontsize=22, weight="bold", va="center")
    ax.text(8.0, 5.2, desc, fontsize=18, va="center", color=MUTED)
    ax.text(20.8, 2.4, f"{nw:,} weights\nsees {rfs:.2f} s", fontsize=21, ha="right", va="center", weight="bold")
axes[-1].text(0, -0.25, "gray: the 30 s input · blue: feature maps, height ∝ √channels, width ∝ √length", fontsize=16, color=MUTED)
save(fig, "topology.png")

# ------------------------------------------------------------------ 11. flatten against global average
fig, ax = canvas(16, 7.5)
ax.set_xlim(-0.5, 32); ax.set_ylim(-3, 12)
for r in range(8):
    for c in range(12):
        ax.add_patch(Rectangle((c * 0.5, 4 + r * 0.5), 0.5, 0.5, fc=plt.get_cmap("Blues")(0.25 + 0.6 * ((r * 7 + c * 3) % 10) / 10), ec="white", lw=0.5))
ax.text(3, 9, "feature maps\n32 channels × 46", ha="center", fontsize=19, weight="bold")
ax.text(3, 3.2, "(8 × 12 drawn)", ha="center", fontsize=15, color=MUTED)
# flatten
arrow(ax, (6.3, 7.0), (9.0, 9.0))
for i in range(30):
    ax.add_patch(Rectangle((9.2 + i * 0.5, 9.0), 0.5, 0.6, fc=plt.get_cmap("Blues")(0.3 + 0.5 * (i % 7) / 7), ec="white", lw=0.5))
ax.text(9.2, 10.4, "A. flatten: all 32 × 46 = 1,472 values in a row", fontsize=19, weight="bold")
ax.text(9.2, 7.8, "linear to 2 classes: 2 × (1,472 + 1) = 2,946 weights · keeps position · input length fixed", fontsize=17)
# GAP
arrow(ax, (6.3, 5.0), (9.0, 2.6))
for i in range(8):
    ax.add_patch(Rectangle((9.2 + i * 0.5, 1.9), 0.5, 0.6, fc="#6baed6", ec="white", lw=0.5))
ax.text(9.2, 3.3, "B. global average: one mean per channel, 32 values", fontsize=19, weight="bold")
ax.text(9.2, 0.8, "linear to 2 classes: 2 × (32 + 1) = 66 weights · forgets position · any input length", fontsize=17)
ax.text(15.5, -2.2, "detection (an event anywhere): B.   picking an arrival time (position is the answer): keep position.",
        ha="center", fontsize=18, color=MUTED)
save(fig, "flatten_gap.png")

# ------------------------------------------------------------------ 12. heads
fig, ax = canvas(17, 7.5)
ax.set_xlim(0, 34); ax.set_ylim(-1, 14)
box(ax, 0.3, 5.0, 8.5, 3.6, "convolutional block\nconv → ReLU → pool\n× 3", OUT_C, fs=20)
ax.text(4.55, 3.9, "the same for both tasks", ha="center", fontsize=17, color=MUTED)
arrow(ax, (8.9, 6.8), (10.4, 6.8))
for i in range(8):
    ax.add_patch(Rectangle((10.6, 4.9 + i * 0.5), 0.6, 0.5, fc="#c6dbef", ec=INK, lw=0.8))
ax.text(10.9, 3.9, "32", ha="center", fontsize=18, weight="bold")
# classification branch
arrow(ax, (11.4, 8.0), (14.3, 10.6), color=POS_C)
box(ax, 14.5, 9.6, 5.6, 2.2, "linear 32 → 2\nscores (logits)", POS_C, fs=18)
arrow(ax, (20.3, 10.7), (21.8, 10.7), color=POS_C)
box(ax, 22.0, 9.6, 3.6, 2.2, "softmax", "#7fc6a4", fs=19)
arrow(ax, (25.8, 10.7), (27.0, 10.7), color=POS_C)
for i, (lab, p) in enumerate([("event", 0.91), ("noise", 0.09)]):
    ax.add_patch(Rectangle((27.2, 11.0 - i * 0.9), 3.5 * p, 0.7, fc=POS_C if i == 0 else NEG_C))
    ax.text(27.3 + 3.5 * p, 11.35 - i * 0.9, f" {lab} {p:.2f}", va="center", fontsize=16)
ax.text(27.2, 12.4, "for example", fontsize=15, color=MUTED)
ax.text(14.5, 12.7, "classification: which class?", fontsize=21, weight="bold", color=POS_C)
ax.text(14.5, 8.6, "loss: cross-entropy (softmax inside)", fontsize=17, color=MUTED)
# regression branch
arrow(ax, (11.4, 5.4), (14.3, 2.8), color=NEG_C)
box(ax, 14.5, 1.6, 5.6, 2.2, "linear 32 → 1\nno activation", NEG_C, fs=18)
arrow(ax, (20.3, 2.7), (21.8, 2.7), color=NEG_C)
ax.text(22.0, 2.7, "one number: the warming trend\nof a map patch, °C per decade\n(notebook 4.3, section 4)", fontsize=17, va="center")
ax.text(14.5, 4.6, "regression: how much?", fontsize=21, weight="bold", color=NEG_C)
ax.text(14.5, 0.5, "loss: mean squared error (or absolute error)", fontsize=17, color=MUTED)
save(fig, "heads.png")

(FIGS / "numbers.json").write_text(json.dumps(numbers, indent=2, default=str))
print(json.dumps({k: numbers[k] for k in ("conv1d", "conv2d", "kernel_size", "weights", "cone")}, indent=1, default=str)[:2500])
