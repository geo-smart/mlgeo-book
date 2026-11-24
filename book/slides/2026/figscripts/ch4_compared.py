"""Comparison figures for the Chapter 4 summary deck (lec26b_chapter4_compared.qmd), written to
figs/ch4_compared/:

 1. connectivity.png  what one layer connects, on the same 8 samples: fully connected (every input to
                      every output, 64 different weights), convolution (3 neighbors, the same 3 weights
                      at every position), recurrence (one step at a time, a memory carried forward),
                      attention (every pair; the weights are computed from the data). The attention
                      weights drawn are the softmax of illustrative scores, not a trained model.
 2. weights.png       number of weights of the networks trained in this chapter, as printed by the
                      notebooks and recorded with the lecture figures:
                        4.1 one hidden layer 2,116           figs/lec21b_mlp/numbers.json
                        4.3 1-D detector 4,658               figs/lec22_cnn/numbers.json
                        4.3 2-D trend CNN 17,953             lecture 22 (notebook 4.3, section 3)
                        4.4 RNN 2,110, LSTM 5,470,
                            attention 4,126, Transformer 18,142   figs/lec23_rnn/numbers.json
                        4.5 fully connected 32,040,
                            message passing 4,577             figs/lec23b_gnn/numbers.json (read here)
                        one dense layer, 3,000 samples -> 64: 192,064   figs/lec22_cnn/numbers.json
Regenerate: pixi run python book/slides/2026/figscripts/ch4_compared.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import json  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "ch4_compared"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 18})
INK, MUTED = "#26241f", "#6e675c"
FCN, CNN, RNN, ATT = "#6e675c", "#0072B2", "#009E73", "#D55E00"
GNN = "#CC79A7"

# ------------------------------------------------------------------ 1. connectivity
n = 8
x = np.arange(n)
fig, axes = plt.subplots(1, 4, figsize=(18, 6.2))
for ax in axes:
    ax.set_xlim(-0.8, n - 0.2); ax.set_ylim(-1.3, 2.9); ax.axis("off")


def nodes(ax, y, color, label=None, filled=True):
    ax.scatter(x, np.full(n, y), s=230, color=color if filled else "white", edgecolor=color, lw=2.5, zorder=3)
    if label:
        ax.text(-0.7, y, label, ha="right", va="center", fontsize=16, color=MUTED)


# fully connected
ax = axes[0]
for i in range(n):
    for j in range(n):
        ax.plot([i, j], [0, 2], color=FCN, lw=0.9, alpha=0.55, zorder=1)
nodes(ax, 0, INK, "in"); nodes(ax, 2, FCN, "out")
ax.set_title("fully connected", color=FCN, weight="bold", fontsize=21)
ax.text(3.5, -0.75, "every input to every output\n8 × 8 = 64 different weights", ha="center", va="top", fontsize=15)

# convolution, kernel 3, same weights everywhere
ax = axes[1]
kcol = ["#56B4E9", CNN, "#003f63"]
for j in range(1, n - 1):
    for k, d in enumerate([-1, 0, 1]):
        ax.plot([j + d, j], [0, 2], color=kcol[k], lw=2.2, alpha=0.9, zorder=1)
nodes(ax, 0, INK); nodes(ax, 2, CNN)
ax.set_title("convolutional", color=CNN, weight="bold", fontsize=21)
ax.text(3.5, -0.75, "3 neighbors in, the same\n3 weights at every position", ha="center", va="top", fontsize=15)

# recurrence
ax = axes[2]
for t in range(n):
    ax.annotate("", xy=(t, 1.78), xytext=(t, 0.22),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, mutation_scale=16))
for t in range(n - 1):
    ax.annotate("", xy=(t + 0.8, 2), xytext=(t + 0.2, 2),
                arrowprops=dict(arrowstyle="-|>", color=RNN, lw=2.6, mutation_scale=18))
nodes(ax, 0, INK); nodes(ax, 2, RNN)
ax.text(7, 2.45, "memory h", ha="center", fontsize=15, color=RNN)
ax.set_title("recurrent", color=RNN, weight="bold", fontsize=21)
ax.text(3.5, -0.75, "one step at a time, a memory\ncarried forward, same weights", ha="center", va="top", fontsize=15)

# attention from the last position
ax = axes[3]
scores = np.array([0.2, 1.4, 0.1, 0.3, 1.6, 0.0, 0.4, 0.9])
wts = np.exp(scores) / np.exp(scores).sum()
for i in range(n):
    ax.plot([i, n - 1], [0, 2], color=ATT, lw=1 + 22 * wts[i], alpha=0.85, zorder=1)
nodes(ax, 0, INK)
ax.scatter(x[:-1], np.full(n - 1, 2), s=230, color="white", edgecolor="#d9d3c7", lw=2, zorder=3)
ax.scatter([n - 1], [2], s=230, color=ATT, zorder=3)
ax.set_title("attention", color=ATT, weight="bold", fontsize=21)
ax.text(3.5, -0.75, "every pair; the weights (line\nwidths) are computed from the data", ha="center", va="top",
        fontsize=15)
fig.text(0.5, 0.01, "The same 8 samples of a record as input (black). Drawn; the attention widths are an example.",
         ha="center", fontsize=14, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIGS / "connectivity.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ 2. weights
GNN_LB = json.loads((HERE.parent / "figs" / "lec23b_gnn" / "numbers.json").read_text())["leaderboard"]
rows = [
    ("one dense layer: 3,000 samples → 64", 192064, FCN, "for scale, lecture 22"),
    ("4.5 fully connected", GNN_LB["fully connected"]["weights"], FCN, "shaking at 40 stations"),
    ("4.4 Transformer encoder", 18142, ATT, "GNSS forecast"),
    ("4.3 2-D CNN", 17953, CNN, "warming trend from maps"),
    ("4.4 LSTM", 5470, RNN, "GNSS forecast"),
    ("4.3 1-D CNN", 4658, CNN, "earthquake detector"),
    ("4.5 message passing", GNN_LB["message passing"]["weights"], GNN, "shaking at any stations"),
    ("4.4 attention, one head", 4126, ATT, "GNSS forecast"),
    ("4.1 one hidden layer", 2116, FCN, "PNW seismic sources"),
    ("4.4 RNN", 2110, RNN, "GNSS forecast"),
]
fig, ax = plt.subplots(figsize=(16, 8.6))
y = np.arange(len(rows))[::-1]
for yi, (name, w, col, task) in zip(y, rows):
    ax.barh(yi, w, color=col, alpha=0.9 if "dense layer" not in name else 0.35, height=0.62)
    ax.text(w * 1.12, yi, f"{w:,}", va="center", fontsize=17, color=INK)
ax.set_yticks(y, [f"{r[0]}\n({r[3]})" for r in rows], fontsize=15)
ax.set_xscale("log"); ax.set_xlim(1e3, 6e5)
ax.set_xlabel("number of weights (log scale)")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.text(0.99, 0.01, "Counts printed by notebooks 4.1, 4.3, 4.4 and 4.5; recorded in figs/lec21b_mlp, "
         "lec22_cnn, lec23_rnn and lec23b_gnn numbers.json.", ha="right", fontsize=12, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIGS / "weights.png", dpi=110); plt.close(fig)
print("wrote", sorted(p.name for p in FIGS.glob("*.png")))
