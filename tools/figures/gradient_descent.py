"""Gradient-descent cartoons for lesson 4.5 (Section 2.4, "Gradient descent and
the learning rate").

Draws four figures in book/img/ from the concept, with the lesson's notation:
loss L(w), one parameter w, learning rate alpha, update
w_(k+1) = w_k - alpha * dL/dw.

    GD_cartoon.jpeg       convex loss, shrinking steps down to the minimum
    GD_non_global.png     local minimum, plateau, global minimum; stuck run
    GD_AlphaTooSmall.png  alpha too small: many tiny steps, far from minimum
    GD_AlphaTooLarge.png  alpha too large: overshoot and diverge

Run:  pixi run python tools/figures/gradient_descent.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OUT = Path(__file__).resolve().parents[2] / "book" / "img"

TEAL = "#116b66"
ORANGE = "#b3402a"
GREY = "#6e675c"
FILL_BLUE = "#cfe3ee"
DPI = 200

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "mathtext.fontset": "dejavusans",
    }
)

XLABEL = "parameter $w$"
YLABEL = r"loss $\mathcal{L}(w)$"


# ----------------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------------
W_STAR = 3.0  # minimum of the convex loss
CONVEX_A = 0.3  # curvature


def convex_loss(w):
    return CONVEX_A * (w - W_STAR) ** 2 + 0.3


def convex_grad(w):
    return 2 * CONVEX_A * (w - W_STAR)


# Non-convex loss built from its derivative: a sum of bumps whose sign pattern
# fixes where the minima, the maximum and the plateau sit, then integrated.
def _bump(w, c, s):
    return np.exp(-(((w - c) / s) ** 2))


def nonconvex_grad(w):
    return (
        -2.2 * _bump(w, 0.9, 0.8)  # slope down into the local minimum
        + 1.3 * _bump(w, 2.5, 0.45)  # climb out to the local maximum
        - 0.35 * _bump(w, 3.5, 0.4)  # gentle drop onto the plateau
        - 0.02  # plateau: almost, but not exactly, flat
        - 1.6 * _bump(w, 7.0, 0.7)  # descent to the global minimum
        + 3.0 * _bump(w, 9.3, 0.6)  # far wall
    )


_W_GRID = np.linspace(0.0, 10.0, 4001)
_L_GRID = np.concatenate(
    [[0.0], np.cumsum(0.5 * (nonconvex_grad(_W_GRID[1:]) + nonconvex_grad(_W_GRID[:-1])) * np.diff(_W_GRID))]
)
_L_GRID -= _L_GRID.min() - 0.25


def nonconvex_loss(w):
    return np.interp(w, _W_GRID, _L_GRID)


def descend(w0, alpha, grad, n_steps):
    w = [w0]
    for _ in range(n_steps):
        w.append(w[-1] - alpha * grad(w[-1]))
    return np.array(w)


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def base_axes(figsize, xlim, ylim, loss, title=None):
    fig, ax = plt.subplots(figsize=figsize)
    w = np.linspace(*xlim, 600)
    L = loss(w)
    ax.fill_between(w, L, ylim[1] + 1, color=FILL_BLUE, alpha=0.35, lw=0)
    ax.plot(w, L, color=TEAL, lw=2.6, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    if title:
        ax.set_title(title, loc="left", color=GREY, fontweight="bold")
    return fig, ax


def draw_path(ax, ws, loss, ms=7.5, arrows=True, start_label=None, start_offset=(0, 0)):
    """Orange markers at each parameter value, grey arrows between them."""
    Ls = loss(ws)
    if arrows:
        for (w0, l0), (w1, l1) in zip(zip(ws[:-1], Ls[:-1]), zip(ws[1:], Ls[1:])):
            ax.annotate(
                "",
                xy=(w1, l1),
                xytext=(w0, l0),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3, shrinkA=ms * 0.6, shrinkB=ms * 0.6),
                zorder=3,
            )
    ax.plot(ws[1:], Ls[1:], "o", color=ORANGE, ms=ms, mec="white", mew=1.0, zorder=4)
    # initial value: same marker, ringed, so it reads as the start of the set
    ax.plot(ws[0], Ls[0], "o", color=ORANGE, ms=ms + 3, mec=ORANGE, mfc="white", mew=1.8, zorder=4)
    ax.plot(ws[0], Ls[0], "o", color=ORANGE, ms=ms - 2, zorder=5)
    if start_label:
        ax.annotate(
            start_label,
            xy=(ws[0], Ls[0]),
            xytext=(ws[0] + start_offset[0], Ls[0] + start_offset[1]),
            ha="center",
            va="center",
            color=ORANGE,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9, shrinkB=8),
        )


def mark_minimum(ax, w_min, loss, label="minimum", dy=-0.13, ymin=None, zorder=3):
    l_min = loss(w_min)
    if ymin is None:
        ymin = ax.get_ylim()[0]
    ax.plot([w_min, w_min], [ymin, l_min], ls=(0, (3, 3)), color=GREY, lw=1.0, zorder=1)
    ax.plot(w_min, l_min, marker="*", color=GREY, ms=12, mec="white", mew=0.6, zorder=zorder)
    ax.text(w_min, l_min + dy, label, ha="center", va="top", color=GREY)


def save(fig, name, **kw):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print("wrote", path)


CONVEX_XLIM = (0.0, 6.0)
CONVEX_YLIM = (-0.3, 3.4)
CONVEX_FIGSIZE = (6.0, 3.75)


# ----------------------------------------------------------------------------
# 1. GD_cartoon.jpeg
# ----------------------------------------------------------------------------
def fig_cartoon():
    fig, ax = base_axes(CONVEX_FIGSIZE, CONVEX_XLIM, CONVEX_YLIM, convex_loss, "Gradient descent")
    ws = descend(0.5, 0.75, convex_grad, 6)  # factor 0.55 per step: steps shrink
    draw_path(ax, ws, convex_loss, start_label="random\ninitial value", start_offset=(0.85, 0.6))
    mark_minimum(ax, W_STAR, convex_loss, zorder=6)
    # learning step: bracket the first update
    w0, w1 = ws[0], ws[1]
    l0, l1 = convex_loss(w0), convex_loss(w1)
    ax.annotate(
        "learning step\n" + r"$-\alpha\,\partial\mathcal{L}/\partial w$",
        xy=((w0 + w1) / 2, (l0 + l1) / 2),
        xytext=(2.6, 2.2),
        ha="center",
        va="center",
        color=GREY,
        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9, shrinkB=6),
    )
    ax.text(
        3.45,
        1.75,
        "steps shrink as the\nslope flattens",
        ha="left",
        va="center",
        color=GREY,
        fontsize=10,
    )
    save(fig, "GD_cartoon.jpeg", format="jpeg", pil_kwargs={"quality": 92})


# ----------------------------------------------------------------------------
# 2. GD_non_global.png
# ----------------------------------------------------------------------------
def fig_non_global():
    xlim = (0.0, 10.0)
    L = nonconvex_loss(_W_GRID)
    ylim = (-0.4, L.max() * 1.03)
    fig, ax = base_axes((7.0, 4.0), xlim, ylim, nonconvex_loss, "Gradient descent on a poorly behaved loss")

    # locate the stationary points from the derivative sign changes
    g = nonconvex_grad(_W_GRID)
    zeros = _W_GRID[:-1][np.sign(g[:-1]) != np.sign(g[1:])]
    w_local, w_max, w_global = zeros[0], zeros[1], zeros[-1]

    # starts on the left wall, converges to the local minimum, stays there
    ws = descend(0.25, 0.35, nonconvex_grad, 4)  # further steps do not move
    draw_path(ax, ws, nonconvex_loss, start_label="initial value", start_offset=(1.5, -0.05))

    mark_minimum(ax, w_local, nonconvex_loss, "local minimum:\nstuck here", dy=-0.18)
    mark_minimum(ax, w_global, nonconvex_loss, "global minimum", dy=-0.18)

    # plateau label
    w_pl = 5.0
    ax.annotate(
        "plateau: gradient near zero,\nprogress crawls",
        xy=(w_pl, nonconvex_loss(w_pl)),
        xytext=(w_pl + 0.9, nonconvex_loss(w_pl) + 0.85),
        ha="center",
        va="center",
        color=GREY,
        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9, shrinkB=6),
    )
    # the hill that blocks the way
    ax.annotate(
        "gradient points back\ninto the local minimum",
        xy=(w_max + 0.05, nonconvex_loss(w_max)),
        xytext=(w_max + 0.1, nonconvex_loss(w_max) + 1.35),
        ha="center",
        va="center",
        color=GREY,
        fontsize=10,
        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9, shrinkB=6),
    )
    save(fig, "GD_non_global.png")


# ----------------------------------------------------------------------------
# 3. GD_AlphaTooSmall.png
# ----------------------------------------------------------------------------
def fig_alpha_small():
    fig, ax = base_axes(
        CONVEX_FIGSIZE, CONVEX_XLIM, CONVEX_YLIM, convex_loss, r"Learning rate $\alpha$ too small"
    )
    ws = descend(0.5, 0.08, convex_grad, 16)  # factor 0.952 per step
    draw_path(ax, ws, convex_loss, ms=5.5, arrows=False, start_label="initial value", start_offset=(1.0, 0.45))
    mark_minimum(ax, W_STAR, convex_loss)
    ax.annotate(
        f"{len(ws) - 1} steps later:\nstill far from the minimum",
        xy=(ws[-1], convex_loss(ws[-1])),
        xytext=(3.5, 2.2),
        ha="center",
        va="center",
        color=GREY,
        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9, shrinkB=6),
    )
    save(fig, "GD_AlphaTooSmall.png")


# ----------------------------------------------------------------------------
# 4. GD_AlphaTooLarge.png
# ----------------------------------------------------------------------------
def fig_alpha_large():
    fig, ax = base_axes(
        CONVEX_FIGSIZE, CONVEX_XLIM, CONVEX_YLIM, convex_loss, r"Learning rate $\alpha$ too large"
    )
    ws = descend(W_STAR + 0.4, 3.75, convex_grad, 8)  # factor -1.25 per step: diverges
    draw_path(ax, ws, convex_loss, start_label="initial value", start_offset=(1.25, -0.15))
    mark_minimum(ax, W_STAR, convex_loss)
    # number the last few steps so the growth reads as a sequence
    for k in (len(ws) - 3, len(ws) - 2, len(ws) - 1):
        wk = ws[k]
        side = 1 if wk > W_STAR else -1
        ax.text(wk + 0.22 * side, convex_loss(wk) + 0.02, f"$k={k}$", ha="left" if side > 0 else "right",
                va="center", color=ORANGE, fontsize=9.5)
    ax.text(
        0.35,
        2.95,
        "each step overshoots the minimum\nand lands at a higher loss",
        ha="left",
        va="center",
        color=GREY,
        fontsize=10,
    )
    save(fig, "GD_AlphaTooLarge.png")


if __name__ == "__main__":
    fig_cartoon()
    fig_non_global()
    fig_alpha_small()
    fig_alpha_large()
