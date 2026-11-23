"""The training loop of every network in Chapter 4, drawn once, for lecture 23 (part 1) and the
Chapter 4 summary deck. Written to figs/lec23_rnn/training_loop.png.

Inner loop: one update per batch (forward pass, loss, backward pass, optimizer step), each step with
its PyTorch line. Outer loop: after each epoch, the validation loss decides the checkpoint.
No data: a drawing. Regenerate: pixi run python book/slides/2026/figscripts/lec23_training_loop.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figs" / "lec23_rnn" / "training_loop.png"
INK, MUTED = "#26241f", "#6e675c"
DATA, MODEL, LOSS, GRAD, STEP, VAL = "#6e675c", "#0072B2", "#D55E00", "#CC79A7", "#009E73", "#E69F00"


def box(ax, x, y, w, h, color, title, code):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                color=color, alpha=0.92, zorder=2))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", color="white",
            fontsize=21, weight="bold", zorder=3)
    ax.text(x + w / 2, y + h * 0.26, code, ha="center", va="center", color="white",
            fontsize=16, family="monospace", zorder=3)


def arrow(ax, a, b, color=INK, rad=0.0, lw=2.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=24, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


fig, ax = plt.subplots(figsize=(17, 7.6))
ax.set_xlim(0, 17.6); ax.set_ylim(0.6, 7.0); ax.axis("off")

w, h = 3.7, 1.35
xs = [0.2, 4.6, 9.0]
# inner loop: four boxes on a ring
box(ax, xs[0], 4.6, w, h, DATA, "a batch", "X, y = next(batches)")
box(ax, xs[1], 4.6, w, h, MODEL, "forward pass", "pred = model(X)")
box(ax, xs[2], 4.6, w, h, LOSS, "loss", "loss_fn(pred, y)")
box(ax, xs[2], 2.2, w, h, GRAD, "backward pass", "loss.backward()")
box(ax, xs[1], 2.2, w, h, STEP, "update", "opt.step()")
arrow(ax, (xs[0] + w + 0.05, 5.27), (xs[1] - 0.05, 5.27))
arrow(ax, (xs[1] + w + 0.05, 5.27), (xs[2] - 0.05, 5.27))
arrow(ax, (xs[2] + w / 2, 4.55), (xs[2] + w / 2, 3.6))
arrow(ax, (xs[2] - 0.05, 2.87), (xs[1] + w + 0.05, 2.87))
arrow(ax, (xs[1] - 0.05, 2.87), (xs[0] + w / 2, 4.55), rad=-0.25)
ax.text(1.6, 3.0, "next batch", ha="center", fontsize=17, color=MUTED, style="italic")
ax.text(6.45, 6.4, "one update per batch: every network of Chapter 4", ha="center", fontsize=20, color=INK)
ax.text(xs[2] + w / 2, 2.05, "one number per weight:\nhow the loss changes", ha="center", va="top",
        fontsize=15, color=MUTED)
ax.text(xs[1] + w / 2, 2.05, "each weight moves against its\ngradient, by the learning rate", ha="center",
        va="top", fontsize=15, color=MUTED)

# outer loop: per epoch
xv = 13.7
box(ax, xv, 3.4, w, h, VAL, "each epoch", "evaluate(model)")
arrow(ax, (xs[2] + w + 0.1, 3.6), (xv - 0.1, 4.0), color=VAL)
ax.text(xv + w / 2, 6.4, "after all batches", ha="center", fontsize=20, color=VAL, weight="bold")
ax.text(xv + w / 2, 5.6, "validation loss,\ngradients off, no update", ha="center", va="top", fontsize=15,
        color=MUTED)
ax.text(xv + w / 2, 3.2, "lowest so far?\nkeep a copy of the weights\n(the checkpoint)", ha="center", va="top",
        fontsize=16, color=INK)

fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT)
