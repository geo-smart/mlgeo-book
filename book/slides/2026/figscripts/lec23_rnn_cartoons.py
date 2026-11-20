"""Cartoons of recurrent networks and attention for lecture 23.

One concept per figure, written to figs/lec23_rnn/, every number computed here
(numbers.json), except the attention and Transformer parameter counts, which are
read from the executed notebook 4.4:

 1. rnn_anatomy.png      90 days in, one day at a time, a memory of 32 numbers, 30 days out
 2. rnn_by_hand.png      the recurrence unrolled over 5 days with one hidden number
 3. memory_decay.png     how long a memory lasts: a spike on day 0 with w_h = 0.5, 0.9, 0.99
 4. weights_count.png    weights of RNN, GRU, LSTM cells and the 4.4 models; none depends on 90
 5. lstm_cell.png        the LSTM cell: a cell state changed by addition, three gates
 6. attention_by_hand.png one query against five days: scores, softmax weights, weighted average
 7. stacked_bidir.png    two stacked layers; a bidirectional layer reads the future
 8. heads.png            one forecast of 30 days, one class per window, one output per day

Regenerate: pixi run python book/slides/2026/figscripts/lec23_rnn_cartoons.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lec23_rnn"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold"})
INK, MUTED, LIGHT = "#26241f", "#6e675c", "#e8e4dc"
IN_C, H_C, OUT_C, NEG_C, K_C, GATE_C = "#8a857c", "#0072B2", "#009E73", "#D55E00", "#E69F00", "#CC79A7"
numbers = {}


def canvas(w=16, h=7.5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def arrow(ax, p, q, color=INK, lw=2.5, ms=22, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def box(ax, x, y, w, h, text, fc, fs=20, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold")


def save(fig, name):
    fig.savefig(FIGS / name, dpi=110, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ 1. anatomy
fig, ax = canvas(17, 7.5)
ax.set_xlim(-0.5, 34); ax.set_ylim(-2.2, 13.2)
rng = np.random.default_rng(1)
days = np.arange(90)
series = 0.03 * days + 0.6 * np.sin(2 * np.pi * days / 60) + 0.15 * rng.standard_normal(90)
ax.plot(0.3 + days * 0.11, 8.6 + series * 0.75, color=IN_C, lw=1.6)
ax.text(5.2, 12.6, "90 days of east displacement (one number per day)", ha="center", fontsize=18, weight="bold")
xs = [1.0, 4.0, 7.0, 13.0]
labels = ["day 1", "day 2", "day 3", "day 90"]
for k, (x0, lab) in enumerate(zip(xs, labels)):
    ax.add_patch(Circle((x0 + 0.6, 2.2), 0.45, fc=IN_C, ec=INK))
    ax.text(x0 + 0.6, 1.1, lab, ha="center", fontsize=16)
    box(ax, x0 - 0.2, 4.0, 1.6, 1.8, "RNN", H_C, fs=15)
    arrow(ax, (x0 + 0.6, 2.7), (x0 + 0.6, 3.95), color=IN_C, lw=2)
    if k < 3:
        arrow(ax, (x0 + 1.25, 4.9), (xs[k + 1] - 0.05 if k < 2 else 9.6, 4.9), color=H_C, lw=2.5)
ax.text(10.5, 4.9, "· · ·", ha="center", va="center", fontsize=26)
arrow(ax, (11.3, 4.9), (12.95, 4.9), color=H_C, lw=2.5)
ax.text(5.5, 6.5, "the same weights every day; the arrow carries the memory h", ha="center", fontsize=17, color=H_C)
arrow(ax, (14.3, 4.9), (16.6, 4.9), color=H_C)
for i in range(8):
    ax.add_patch(Rectangle((16.8, 2.9 + i * 0.5), 0.6, 0.5, fc="#c6dbef", ec=INK, lw=0.8))
ax.text(17.1, 2.0, "h on day 90\n32 numbers", ha="center", fontsize=16, va="top")
arrow(ax, (17.6, 4.9), (19.6, 4.9))
box(ax, 19.8, 3.9, 4.6, 2.0, "linear\n32 → 30", OUT_C, fs=18)
arrow(ax, (24.6, 4.9), (26.0, 4.9))
fut = series[-1] + 0.03 * np.arange(1, 31) + 0.6 * (np.sin(2 * np.pi * (90 + np.arange(1, 31)) / 60) - np.sin(2 * np.pi * 89 / 60))
ax.plot(26.3 + np.arange(30) * 0.24, 4.0 + (fut - fut.mean()) * 1.2, "o-", color=OUT_C, ms=4, lw=1.5)
ax.text(29.9, 7.2, "the next 30 days\n(a forecast, regression)", ha="center", fontsize=17)
ax.text(17, -1.6, "tensor shapes: (batch, 90, 1) in  →  (batch, 90, 32) hidden states  →  (batch, 32) last one  →  (batch, 30) out",
        ha="center", fontsize=17, color=MUTED)
save(fig, "rnn_anatomy.png")

# ------------------------------------------------------------------ 2. by hand
wx, wh, bb = 0.8, 0.6, 0.0
xseq = [1.0, 0.0, 0.0, 0.5, 0.0]
h, hs = 0.0, []
for xt in xseq:
    h = float(np.tanh(wx * xt + wh * h + bb))
    hs.append(h)
numbers["by_hand"] = {"w_x": wx, "w_h": wh, "b": bb, "x": xseq, "h": hs}
fig, ax = canvas(17, 7.2)
ax.set_xlim(-2.5, 31); ax.set_ylim(-2.8, 10)
for t, (xt, ht) in enumerate(zip(xseq, hs)):
    x0 = 1 + t * 6
    ax.add_patch(Circle((x0 + 1, 0.8), 0.6, fc=IN_C, ec=INK))
    ax.text(x0 + 1, 0.8, f"{xt:g}", ha="center", va="center", fontsize=18, color="white", weight="bold")
    ax.text(x0 + 1, -0.6, f"x on day {t + 1}", ha="center", fontsize=16)
    box(ax, x0, 3.2, 2.0, 2.0, "tanh", H_C, fs=17)
    arrow(ax, (x0 + 1, 1.45), (x0 + 1, 3.15), color=IN_C, lw=2)
    ax.text(x0 + 1.25, 2.25, "×0.8", fontsize=15, color=MUTED)
    ax.text(x0 + 1, 6.4, f"h = {ht:.2f}", ha="center", fontsize=19, weight="bold", color=H_C)
    arrow(ax, (x0 + 1, 5.25), (x0 + 1, 5.95), color=H_C, lw=2)
    if t < 4:
        arrow(ax, (x0 + 2.05, 4.2), (x0 + 5.95, 4.2), color=H_C, lw=2.5)
        ax.text(x0 + 4.0, 4.5, "×0.6", ha="center", fontsize=15, color=MUTED)
ax.text(14, 8.6, "h(day t) = tanh( 0.8 × x(day t) + 0.6 × h(day t − 1) )", ha="center", fontsize=22)
ax.text(14, -2.2, f"day 3 has no input, yet h = {hs[2]:.2f}: the memory of day 1, fading by a factor of about 0.6 a day",
        ha="center", fontsize=17, color=MUTED)
save(fig, "rnn_by_hand.png")

# ------------------------------------------------------------------ 3. memory decay
fig, ax = plt.subplots(figsize=(16, 6.5))
k = np.arange(0, 120)
reach = {}
for w, c in [(0.5, NEG_C), (0.9, K_C), (0.99, H_C)]:
    ax.semilogy(k, w ** k, lw=3.5, color=c, label=f"w_h = {w}")
    n1 = int(np.ceil(np.log(0.01) / np.log(w)))
    reach[w] = n1
    if n1 < 120:
        ax.plot(n1, 0.01, "o", color=c, ms=12)
        ax.annotate(f"{n1} days", (n1, 0.01), xytext=(n1 + 2, 0.003), fontsize=18, color=c)
    else:
        ax.annotate(f"{n1} days to reach 1%", (119, w ** 119), xytext=(80, 0.5), fontsize=18, color=c)
ax.axhline(0.01, color=MUTED, ls="--", lw=1.5)
ax.text(60, 0.013, "1% left", fontsize=17, color=MUTED)
ax.axvline(90, color=INK, lw=1.5, ls=":")
ax.text(91, 2e-4, "90-day\nwindow", fontsize=16)
ax.set_xlabel("days after a spike in the input"); ax.set_ylabel("what is left of it in h")
ax.set_ylim(1e-4, 1.5); ax.legend(loc="lower left", fontsize=18)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("a memory fades by a factor w_h each day", loc="left", fontsize=22)
numbers["memory_decay_days_to_1pct"] = reach
save(fig, "memory_decay.png")

# ------------------------------------------------------------------ 4. weights
H, I = 32, 1
cells = {"RNN cell": nn.RNN(I, H), "GRU cell": nn.GRU(I, H), "LSTM cell": nn.LSTM(I, H)}
cw = {k_: n_params(m) for k_, m in cells.items()}
head = 30 * (H + 1)
reported = {"attention (4.4)": 4126, "Transformer (4.4)": 18142}
numbers["weights"] = {"cells": cw, "head_32_to_30": head, "rnn_model": cw["RNN cell"] + head,
                      "lstm_model": cw["LSTM cell"] + head, "reported_in_4.4": reported}
fig, ax = plt.subplots(figsize=(16, 7))
rows = [("RNN cell", cw["RNN cell"], f"= 32 × (1 + 32 + 2)"), ("GRU cell: 3 gates", cw["GRU cell"], "= 3 × RNN cell"),
        ("LSTM cell: 4 gates", cw["LSTM cell"], "= 4 × RNN cell"), ("head: linear 32 → 30", head, "= 30 × (32 + 1)"),
        ("RNN model of 4.4", cw["RNN cell"] + head, "cell + head"), ("LSTM model of 4.4", cw["LSTM cell"] + head, "cell + head"),
        ("attention model of 4.4", reported["attention (4.4)"], "as printed in 4.4"),
        ("Transformer of 4.4", reported["Transformer (4.4)"], "as printed in 4.4")]
yy = np.arange(len(rows))[::-1]
cols = [H_C, H_C, H_C, OUT_C, INK, INK, GATE_C, GATE_C]
ax.barh(yy, [r[1] for r in rows], color=cols, height=0.62)
for y, (lab, v, f) in zip(yy, rows):
    ax.text(v + 150, y, f"{v:,}  {f}", va="center", fontsize=18)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=18)
ax.set_xlim(0, 26000); ax.set_xlabel("number of weights")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("hidden size 32: the 90 days appear nowhere in the count", loc="left", fontsize=22)
save(fig, "weights_count.png")

# ------------------------------------------------------------------ 5. LSTM cell
fig, ax = canvas(17, 7.5)
ax.set_xlim(-1, 33); ax.set_ylim(-1.5, 12.5)
ax.add_patch(FancyBboxPatch((3, 1.0), 24, 9.0, boxstyle="round,pad=0.2,rounding_size=0.6", fc="#f4f1ea", ec=MUTED, lw=2))
# cell state line
ax.annotate("", xy=(30.5, 8.6), xytext=(0.0, 8.6), arrowprops=dict(arrowstyle="-|>", lw=5, color=K_C, mutation_scale=30))
ax.text(0.0, 9.4, "cell state c (day t − 1)", fontsize=17, color="#9a6a00")
ax.text(27.5, 9.4, "c (day t)", fontsize=17, color="#9a6a00")
for xg, sym in [(9.0, "×"), (16.0, "+")]:
    ax.add_patch(Circle((xg, 8.6), 0.75, fc="white", ec=INK, lw=2.5))
    ax.text(xg, 8.6, sym, ha="center", va="center", fontsize=28)
box(ax, 7.6, 3.6, 2.8, 1.6, "forget\nσ", GATE_C, fs=16)
box(ax, 12.0, 3.6, 2.8, 1.6, "input\nσ", GATE_C, fs=16)
box(ax, 15.6, 3.6, 2.8, 1.6, "new\ntanh", H_C, fs=16)
box(ax, 21.0, 3.6, 2.8, 1.6, "output\nσ", GATE_C, fs=16)
arrow(ax, (9.0, 5.25), (9.0, 7.8), color=GATE_C)
ax.add_patch(Circle((15.2, 6.6), 0.55, fc="white", ec=INK, lw=2)); ax.text(15.2, 6.6, "×", ha="center", va="center", fontsize=22)
arrow(ax, (13.4, 5.25), (14.8, 6.2), color=GATE_C, lw=2); arrow(ax, (17.0, 5.25), (15.6, 6.2), color=H_C, lw=2)
arrow(ax, (15.4, 7.15), (15.85, 7.85), color=INK, lw=2)
ax.add_patch(Circle((24.8, 6.4), 0.55, fc="white", ec=INK, lw=2)); ax.text(24.8, 6.4, "×", ha="center", va="center", fontsize=22)
arrow(ax, (24.8, 8.55), (24.8, 6.95), color=K_C, lw=2)
arrow(ax, (22.4, 5.25), (24.4, 6.0), color=GATE_C, lw=2)
arrow(ax, (25.35, 6.4), (30.5, 6.4), color=H_C, lw=3)
ax.text(28.0, 5.6, "h (day t)", fontsize=17, color=H_C)
ax.annotate("", xy=(22.4, 2.2), xytext=(0.0, 2.2), arrowprops=dict(arrowstyle="-", lw=3, color=H_C))
ax.text(0.0, 1.3, "h (day t − 1) and x (day t) go to every gate", fontsize=16, color=H_C)
for xg in (9.0, 13.4, 17.0, 22.4):
    ax.plot([xg, xg], [2.2, 3.55], color=H_C, lw=2)
ax.text(16, 11.6, "the memory c changes by multiplication with a gate (0 to 1) and by addition", ha="center", fontsize=19)
ax.text(16, -1.0, "forget: what to keep · input: what to write · output: what to show as h.  σ: sigmoid, between 0 and 1",
        ha="center", fontsize=17, color=MUTED)
save(fig, "lstm_cell.png")

# ------------------------------------------------------------------ 6. attention by hand
vals = np.array([1.0, 3.0, 2.0, 5.0, 4.0])            # values of five days, e.g. mm
keys = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.2]])
query = np.array([0.2, 2.0])
scores = keys @ query / np.sqrt(2)
wts = np.exp(scores) / np.exp(scores).sum()
out = float(wts @ vals)
numbers["attention"] = {"keys": keys.tolist(), "query": query.tolist(), "scores": scores.tolist(),
                        "weights": wts.tolist(), "values": vals.tolist(), "output": out}
fig, ax = canvas(17, 7.8)
ax.set_xlim(-6.5, 30); ax.set_ylim(-3.2, 12)
cols = ["day 1", "day 2", "day 3", "day 4", "day 5"]
rowsy = {"key k": 9.0, "score q·k / √2": 6.8, "weight (softmax)": 4.6, "value v": 2.4}
for r, y in rowsy.items():
    ax.text(-6.3, y, r, fontsize=18, va="center", weight="bold")
for j, c in enumerate(cols):
    x0 = 1 + j * 4.2
    ax.text(x0 + 1.5, 10.6, c, ha="center", fontsize=18)
    ax.text(x0 + 1.5, 9.0, f"({keys[j, 0]:g}, {keys[j, 1]:g})", ha="center", va="center", fontsize=17)
    ax.text(x0 + 1.5, 6.8, f"{scores[j]:.2f}", ha="center", va="center", fontsize=18)
    ax.add_patch(Rectangle((x0 + 0.2, 4.0), 2.6 * wts[j] / wts.max(), 1.2, fc=K_C))
    ax.text(x0 + 1.5, 3.45, f"{wts[j]:.2f}", ha="center", va="center", fontsize=17)
    ax.text(x0 + 1.5, 2.4, f"{vals[j]:g}", ha="center", va="center", fontsize=18)
ax.text(23.2, 9.0, f"query q of day 5:\n({query[0]:g}, {query[1]:g})", fontsize=18, va="center", color=H_C, weight="bold")
ax.text(23.2, 4.6, "weights add\nup to 1", fontsize=17, va="center", color="#9a6a00")
ax.text(11.5, 0.6, "output for day 5 = " + " + ".join(f"{w:.2f}×{v:g}" for w, v in zip(wts, vals)) + f" = {out:.2f}",
        ha="center", fontsize=18)
ax.text(11.5, -2.2, "every day is compared with every other day at once: no chain of 90 steps, no fading",
        ha="center", fontsize=17, color=MUTED)
save(fig, "attention_by_hand.png")

# ------------------------------------------------------------------ 7. stacked and bidirectional
fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
for ax, title in zip(axes, ["two stacked layers", "bidirectional: also reads backward"]):
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-1.5, 8.2); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title(title, fontsize=22)
    for t in range(5):
        x0 = 0.5 + t * 2
        ax.add_patch(Circle((x0 + 0.5, 0), 0.4, fc=IN_C, ec=INK))
for t in range(5):
    x0 = 0.5 + t * 2
    for lev, y in enumerate([2.5, 5.0]):
        axes[0].add_patch(Rectangle((x0, y - 0.5), 1.0, 1.0, fc=H_C if lev == 0 else "#08519c", ec="white"))
        if t < 4:
            arrow(axes[0], (x0 + 1.05, y), (x0 + 1.95, y), color=H_C, lw=2, ms=16)
    arrow(axes[0], (x0 + 0.5, 0.45), (x0 + 0.5, 1.95), color=IN_C, lw=1.5, ms=14)
    arrow(axes[0], (x0 + 0.5, 3.05), (x0 + 0.5, 4.45), color=H_C, lw=1.5, ms=14)
axes[0].text(5, 7.0, "layer 2 reads the memories of layer 1", ha="center", fontsize=17, color=MUTED)
for t in range(5):
    x0 = 0.5 + t * 2
    axes[1].add_patch(Rectangle((x0, 2.0), 1.0, 1.0, fc=H_C, ec="white"))
    axes[1].add_patch(Rectangle((x0, 4.3), 1.0, 1.0, fc=NEG_C, ec="white"))
    arrow(axes[1], (x0 + 0.5, 0.45), (x0 + 0.5, 1.95), color=IN_C, lw=1.5, ms=14)
    if t < 4:
        arrow(axes[1], (x0 + 1.05, 2.5), (x0 + 1.95, 2.5), color=H_C, lw=2, ms=16)
        arrow(axes[1], (x0 + 1.95, 4.8), (x0 + 1.05, 4.8), color=NEG_C, lw=2, ms=16)
axes[1].text(5, 6.3, "day 2 sees days 3, 4, 5: the future", ha="center", fontsize=18, color=NEG_C, weight="bold")
axes[1].text(5, 7.4, "fine for a finished record; never for a forecast", ha="center", fontsize=17, color=MUTED)
for ax in axes:
    ax.text(5, -1.3, "input days 1 to 5", ha="center", fontsize=16, color=MUTED)
save(fig, "stacked_bidir.png")

# ------------------------------------------------------------------ 8. heads
fig, axes = plt.subplots(1, 3, figsize=(17, 6.0))
titles = ["many in, many out later:\na 30-day forecast", "many in, one out:\none class per window", "one out per day:\na pick probability"]
for ax, title in zip(axes, titles):
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-1.2, 8.0); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title(title, fontsize=19)
    for t in range(5):
        x0 = 0.5 + t * 2
        ax.add_patch(Circle((x0 + 0.5, 0), 0.4, fc=IN_C, ec=INK))
        ax.add_patch(Rectangle((x0, 2.0), 1.0, 1.0, fc=H_C, ec="white"))
        arrow(ax, (x0 + 0.5, 0.45), (x0 + 0.5, 1.95), color=IN_C, lw=1.5, ms=14)
        if t < 4:
            arrow(ax, (x0 + 1.05, 2.5), (x0 + 1.95, 2.5), color=H_C, lw=2, ms=16)
# forecast
arrow(axes[0], (9.0, 3.05), (9.0, 4.5), color=OUT_C)
box(axes[0], 5.4, 4.6, 5.0, 1.3, "linear → 30", OUT_C, fs=15)
axes[0].text(5.2, 6.7, "regression: 30 numbers, MSE", ha="center", fontsize=16)
# class
arrow(axes[1], (9.0, 3.05), (9.0, 4.5), color=OUT_C)
box(axes[1], 4.6, 4.6, 5.9, 1.3, "linear → softmax", OUT_C, fs=15)
axes[1].text(5.2, 6.7, "classification: event or noise", ha="center", fontsize=16)
# per step
for t in range(5):
    x0 = 0.5 + t * 2
    arrow(axes[2], (x0 + 0.5, 3.05), (x0 + 0.5, 4.3), color=OUT_C, lw=1.5, ms=14)
    p = [0.02, 0.05, 0.9, 0.3, 0.05][t]
    axes[2].add_patch(Rectangle((x0, 4.4), 1.0, 2.0 * p + 0.05, fc=OUT_C))
axes[2].text(5.2, 6.9, "e.g. P arrival on day 3", ha="center", fontsize=16)
fig.text(0.5, 0.02, "the recurrent layer is the same; what is read from it, and the loss, set the task", ha="center", fontsize=17, color=MUTED)
save(fig, "heads.png")

(FIGS / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
print(json.dumps(numbers, indent=1, default=float)[:2200])
