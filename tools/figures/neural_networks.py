"""Neural-network cartoons for Chapters 3 and 4, drawn with matplotlib only.

Replaces six images copied from Géron's *Hands-On ML*, Lilian Weng's blog and
Ronneberger et al. 2015. Each figure is drawn from the concept as the lesson
teaches it, with the lesson's own notation:

  book/img/TLU.png                 4.1  y = f(sum_i w_i x_i + b), f = step
  book/img/MLPReg.png              4.2  regression MLP, linear output units
  book/img/MLPClass.png            4.2  classification MLP, softmax output
  book/img/votingclassifier.png    3.9  naive Bayes + random forest + SVC, majority vote
      (also written to book/Chapter3-MachineLearning/, the path that page loads)
  book/img/autoen_architecture.png 4.6  input -> encoder -> bottleneck -> decoder -> reconstruction
  book/img/Unet.png                4.6  contracting/expanding paths + skip connections

Run:  pixi run python tools/figures/neural_networks.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "book" / "img"

TEAL = "#116b66"
ORANGE = "#b3402a"
GREY = "#6e675c"
BLUE_F = "#cfe3ee"
GREEN_F = "#d9ead3"
ORANGE_F = "#f9cb9c"
DPI = 200

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "mathtext.fontset": "dejavusans",
        "text.color": "#222222",
        "savefig.facecolor": "white",
    }
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def node(ax, xy, r, fill=BLUE_F, edge=TEAL, text=None, fs=11, lw=1.4, ls="-", weight="normal"):
    ax.add_patch(Circle(xy, r, facecolor=fill, edgecolor=edge, lw=lw, ls=ls, zorder=3))
    if text is not None:
        ax.text(*xy, text, ha="center", va="center", fontsize=fs, zorder=4, fontweight=weight)


def box(ax, xy, w, h, fill=BLUE_F, edge=TEAL, text=None, fs=10.5, lw=1.4, ls="-", pad=0.12, weight="normal"):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
            facecolor=fill, edgecolor=edge, lw=lw, ls=ls, zorder=3,
        )
    )
    if text is not None:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4, fontweight=weight, linespacing=1.25)


def arrow(ax, p0, p1, color=GREY, lw=1.3, ls="-", head=8, style="-|>", conn="arc3,rad=0", zorder=2):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=head, color=color, lw=lw, ls=ls,
            connectionstyle=conn, zorder=zorder, shrinkA=0, shrinkB=0,
        )
    )


def on_circle(c, toward, r):
    """Point on the circle of centre c, radius r, in the direction of `toward`."""
    c, t = np.asarray(c, float), np.asarray(toward, float)
    d = t - c
    return tuple(c + r * d / np.linalg.norm(d))


def circle_arrow(ax, c0, r0, c1, r1, **kw):
    arrow(ax, on_circle(c0, c1, r0), on_circle(c1, c0, r1), **kw)


def label(ax, xy, s, fs=10, color=GREY, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(*xy, s, fontsize=fs, color=color, zorder=5, **kw)


def relu_glyph(ax, c, r, color=TEAL):
    x0, y0 = c
    s = 0.55 * r
    ax.plot([x0 - s, x0, x0 + s], [y0 - 0.35 * s, y0 - 0.35 * s, y0 + s], color=color, lw=1.4, zorder=4)


def linear_glyph(ax, c, r, color=TEAL):
    x0, y0 = c
    s = 0.55 * r
    ax.plot([x0 - s, x0 + s], [y0 - s, y0 + s], color=color, lw=1.4, zorder=4)


def step_glyph(ax, c, r, color=TEAL):
    x0, y0 = c
    s = 0.55 * r
    ax.plot([x0 - s, x0, x0, x0 + s], [y0 - 0.6 * s, y0 - 0.6 * s, y0 + 0.6 * s, y0 + 0.6 * s], color=color, lw=1.6, zorder=4)


def save(fig, name, extra=()):
    out = IMG / name
    fig.savefig(out, dpi=DPI, facecolor="white")
    for p in extra:
        fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT), *[Path(p).relative_to(ROOT) for p in extra])


# ----------------------------------------------------------------------------
# 1. Threshold logic unit (4.1):  y = f( sum_i w_i x_i + b ), f = step
# ----------------------------------------------------------------------------
def fig_tlu():
    fig, ax = canvas(7.2, 3.9, (0, 10), (-0.55, 4.85))
    r_in, r_n = 0.38, 0.55
    ins = [(1.2, 4.0), (1.2, 2.5), (1.2, 1.0)]
    c_sum, c_act = (4.7, 2.5), (6.9, 2.5)

    # neuron envelope
    ax.add_patch(
        FancyBboxPatch((3.85, 1.45), 3.85, 2.45, boxstyle="round,pad=0,rounding_size=0.3",
                       facecolor="none", edgecolor=GREY, lw=1.0, ls=(0, (4, 3)), zorder=1)
    )
    label(ax, (5.78, 4.15), "threshold logic unit (one neuron)", fs=10, color=GREY)

    for i, c in enumerate(ins, start=1):
        node(ax, c, r_in, text=rf"$x_{i}$", fs=12)
        circle_arrow(ax, c, r_in, c_sum, r_n, color=TEAL, lw=1.4, head=10)
        # weight label on the arrow, offset perpendicular to it
        p0, p1 = np.array(on_circle(c, c_sum, r_in)), np.array(on_circle(c_sum, c, r_n))
        m = p0 + 0.45 * (p1 - p0)
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        n_ = np.array([-d[1], d[0]]) * (0.28 if i < 3 else -0.28)
        if i == 2:
            n_ = np.array([0, 0.27])
        label(ax, tuple(m + n_), rf"$w_{i}$", fs=11.5, color=ORANGE)

    node(ax, c_sum, r_n, fill=GREEN_F, text=r"$\Sigma$", fs=17)
    label(ax, (c_sum[0], c_sum[1] + r_n + 0.27), "weighted sum", fs=9.5)

    # bias enters the sum from below: one per neuron, not one per input
    arrow(ax, (c_sum[0], 0.45), (c_sum[0], c_sum[1] - r_n), color=ORANGE, lw=1.3, head=9)
    label(ax, (c_sum[0] + 0.2, 0.85), r"bias $b$", fs=10.5, color=ORANGE, ha="left")

    circle_arrow(ax, c_sum, r_n, c_act, r_n, color=GREY, lw=1.3, head=9)
    label(ax, ((c_sum[0] + c_act[0]) / 2, 2.8), r"$z$", fs=11.5, color="#222222")

    node(ax, c_act, r_n, fill=ORANGE_F, edge=ORANGE)
    step_glyph(ax, c_act, r_n, color=ORANGE)
    label(ax, (c_act[0], c_act[1] + r_n + 0.27), "step $f$", fs=9.5)

    arrow(ax, on_circle(c_act, (9.2, 2.5), r_n), (8.9, 2.5), color=TEAL, lw=1.4, head=10)
    label(ax, (9.25, 2.5), r"$y$", fs=13, color="#222222")

    label(ax, (5.0, -0.2), r"$y = f\left(\sum_i w_i x_i + b\right)$,   $f$ = step: fires when $z > 0$",
          fs=11, color="#222222")
    save(fig, "TLU.png")


# ----------------------------------------------------------------------------
# 2-3. Multilayer perceptron (4.2)
# ----------------------------------------------------------------------------
def _mlp(kind):
    fig, ax = canvas(7.6, 4.9, (0, 10.4), (-0.2, 7.2))
    r = 0.36
    x_in, x_h, x_out = 1.6, 5.0, 8.4

    ins = [(x_in, 4.5, r"$x_1$"), (x_in, 3.2, r"$x_2$")]
    bias = (x_in, 1.9)
    hid = [(x_h, y) for y in (5.2, 4.0, 2.8, 1.6)]
    if kind == "reg":
        outs = [(x_out, 4.0), (x_out, 2.8)]
        out_labels = [r"$\hat{y}_1$", r"$\hat{y}_2$"]
    else:
        outs = [(x_out, y) for y in (4.6, 3.4, 2.2)]
        out_labels = [r"$P(Y\!=\!1)$", r"$P(Y\!=\!2)$", r"$P(Y\!=\!3)$"]

    # fully connected edges (thin, behind nodes)
    for (xi, yi, _t) in ins + [(bias[0], bias[1], None)]:
        for (xh, yh) in hid:
            ax.plot([xi, xh], [yi, yh], color=GREY, lw=0.7, alpha=0.75, zorder=1)
    for (xh, yh) in hid:
        for (xo, yo) in outs:
            ax.plot([xh, xo], [yh, yo], color=GREY, lw=0.7, alpha=0.75, zorder=1)

    # nodes
    for (xi, yi, t) in ins:
        node(ax, (xi, yi), r, text=t, fs=12)
    node(ax, bias, r, fill="white", edge=GREY, ls=(0, (3, 2)), text="1", fs=11)
    label(ax, (bias[0], bias[1] - r - 0.26), "bias", fs=9.5)

    for c in hid:
        node(ax, c, r, fill=GREEN_F)
        relu_glyph(ax, c, r)

    for c, t in zip(outs, out_labels):
        if kind == "reg":
            node(ax, c, r, fill=ORANGE_F, edge=ORANGE)
            linear_glyph(ax, c, r, color=ORANGE)
        else:
            node(ax, c, r, fill=ORANGE_F, edge=ORANGE)
        arrow(ax, (c[0] + r, c[1]), (c[0] + r + 0.55, c[1]), color=ORANGE, lw=1.3, head=9)
        label(ax, (c[0] + r + 0.65, c[1]), t, fs=11.5, color="#222222", ha="left")

    if kind == "clf":
        # one softmax couples all the output units: draw it as an envelope
        top, bot = outs[0][1] + r + 0.25, outs[-1][1] - r - 0.55
        ax.add_patch(
            FancyBboxPatch((x_out - r - 0.22, bot), 2 * r + 0.44, top - bot,
                           boxstyle="round,pad=0,rounding_size=0.3",
                           facecolor="none", edgecolor=ORANGE, lw=1.0, ls=(0, (4, 3)), zorder=2)
        )
        label(ax, (x_out, bot + 0.22), "softmax", fs=9.5, color=ORANGE)

    # layer titles
    ytit = 6.55
    label(ax, (x_in, ytit), "Input layer", fs=11.5, color=TEAL, weight="bold")
    label(ax, (x_h, ytit), "Hidden layer", fs=11.5, color=TEAL, weight="bold")
    label(ax, (x_h, ytit - 0.42), "ReLU units", fs=10)
    if kind == "reg":
        label(ax, (x_out + 0.3, ytit), "Output layer", fs=11.5, color=ORANGE, weight="bold")
        label(ax, (x_out + 0.3, ytit - 0.42), "linear units, one per value", fs=10)
    else:
        label(ax, (x_out + 0.3, ytit), "Softmax output layer", fs=11.5, color=ORANGE, weight="bold")
        label(ax, (x_out + 0.3, ytit - 0.42), "one unit per class", fs=10)

    # equation from the lesson
    if kind == "reg":
        eq = r"hidden layer: $h(\mathbf{x}) = \phi(\mathbf{W}\mathbf{x} + \mathbf{b})$,  $\phi$ = ReLU;   output: no activation"
    else:
        eq = r"hidden layer: $h(\mathbf{x}) = \phi(\mathbf{W}\mathbf{x} + \mathbf{b})$,  $\phi$ = ReLU;   output: softmax, probabilities sum to 1"
    label(ax, (5.2, 0.35), eq, fs=10.5, color="#222222")
    save(fig, "MLPReg.png" if kind == "reg" else "MLPClass.png")


def fig_mlp_reg():
    _mlp("reg")


def fig_mlp_clf():
    _mlp("clf")


# ----------------------------------------------------------------------------
# 4. Voting classifier (3.9): naive Bayes + random forest + SVC, majority vote
# ----------------------------------------------------------------------------
def fig_voting():
    fig, ax = canvas(8.4, 4.2, (0, 13.4), (0, 6.4))
    ys = (4.7, 3.2, 1.7)
    clfs = ["Gaussian\nnaive Bayes", "Random forest", "Support vector\nclassifier"]
    votes = ["earthquake", "explosion", "earthquake"]
    fills = [BLUE_F, GREEN_F, ORANGE_F]
    edges = [TEAL, TEAL, ORANGE]

    x_s, x_c, x_v, x_m = 1.75, 5.3, 8.55, 11.4
    box(ax, (x_s, 3.2), 2.75, 1.15, fill=BLUE_F, text="new sample $\\mathbf{x}$\nwaveform features", fs=10.5)
    label(ax, (x_s, 5.75), "one sample", fs=10.5, color=TEAL, weight="bold")
    label(ax, (x_c, 5.75), "three different classifiers", fs=10.5, color=TEAL, weight="bold")
    label(ax, (x_v, 5.75), "one vote each", fs=10.5, color=TEAL, weight="bold")
    label(ax, (x_m, 5.75), "aggregate", fs=10.5, color=TEAL, weight="bold")

    for y, name, v, f, e in zip(ys, clfs, votes, fills, edges):
        rad = 0.0 if y == 3.2 else (-0.18 if y > 3.2 else 0.18)
        arrow(ax, (x_s + 1.375, 3.2), (x_c - 1.45, y), color=GREY, lw=1.2, head=9, conn=f"arc3,rad={rad}")
        box(ax, (x_c, y), 2.9, 1.05, fill=f, edge=e, text=name, fs=10.5)
        arrow(ax, (x_c + 1.45, y), (x_v - 1.1, y), color=GREY, lw=1.2, head=9)
        box(ax, (x_v, y), 2.2, 0.72, fill="white", edge=GREY, text=v, fs=10.5, pad=0.3)
        # arrive at three distinct points on the vote box, so the heads do not pile up
        y_in = 3.2 + 0.32 * np.sign(y - 3.2)
        arrow(ax, (x_v + 1.1, y), (x_m - 1.05, y_in), color=GREY, lw=1.2, head=9, conn=f"arc3,rad={-rad}")

    box(ax, (x_m, 3.2), 2.1, 1.15, fill=GREEN_F, text="majority\nvote", fs=10.5)
    arrow(ax, (x_m, 3.2 - 0.575), (x_m, 1.75), color=TEAL, lw=1.5, head=11)
    label(ax, (x_m + 0.15, 2.18), "2 of 3 votes", fs=9.5, ha="left")
    box(ax, (x_m, 1.3), 2.1, 0.72, fill=TEAL, edge=TEAL, text="earthquake", fs=10.5, pad=0.3)
    ax.texts[-1].set_color("white")

    label(ax, (6.7, 0.4),
          "hard voting counts predicted classes; soft voting averages predicted probabilities",
          fs=9.5)
    save(fig, "votingclassifier.png",
         extra=[ROOT / "book" / "Chapter3-MachineLearning" / "votingclassifier.png"])


# ----------------------------------------------------------------------------
# 5. Autoencoder (4.6): input -> encoder -> bottleneck -> decoder -> reconstruction
# ----------------------------------------------------------------------------
def _image_bar(ax, x0, x1, y0, y1, seed, fill=BLUE_F):
    """A tall bar with a faint grid, standing for an image (a spectrogram)."""
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fill, edgecolor=TEAL, lw=1.4, zorder=3))
    rng = np.random.default_rng(seed)
    nx, ny = 3, 12
    w, h = (x1 - x0) / nx, (y1 - y0) / ny
    for i in range(nx):
        for j in range(ny):
            a = 0.08 + 0.35 * rng.random()
            ax.add_patch(plt.Rectangle((x0 + i * w, y0 + j * h), w, h, facecolor=TEAL, alpha=a,
                                       edgecolor="none", zorder=3))


def fig_autoencoder():
    fig, ax = canvas(8.0, 3.7, (0, 12), (-0.35, 5.15))
    y0, y1 = 1.0, 4.2
    zb, zt = 2.05, 3.15

    # input
    _image_bar(ax, 0.9, 1.7, y0, y1, seed=1)
    label(ax, (1.3, 4.55), "input $x$", fs=11.5, color="#222222")

    arrow(ax, (1.75, 2.6), (2.25, 2.6), color=GREY, lw=1.3, head=9)
    # encoder
    ax.add_patch(Polygon([(2.3, y0), (2.3, y1), (4.9, zt), (4.9, zb)], closed=True,
                         facecolor=GREEN_F, edgecolor=TEAL, lw=1.4, zorder=3))
    label(ax, (3.5, 2.6), "encoder", fs=11.5, color="#222222")
    label(ax, (3.5, 4.55), "compress", fs=10)

    arrow(ax, (4.95, 2.6), (5.35, 2.6), color=GREY, lw=1.3, head=9)
    # bottleneck
    ax.add_patch(plt.Rectangle((5.4, zb), 0.55, zt - zb, facecolor=ORANGE_F, edgecolor=ORANGE, lw=1.6, zorder=3))
    label(ax, (5.675, zt + 0.32), "$z$", fs=13, color="#222222")
    label(ax, (5.675, zb - 0.32), "bottleneck", fs=10, color=ORANGE)
    label(ax, (5.675, zb - 0.62), "latent space", fs=10, color=ORANGE)

    arrow(ax, (6.0, 2.6), (6.4, 2.6), color=GREY, lw=1.3, head=9)
    # decoder
    ax.add_patch(Polygon([(6.45, zb), (6.45, zt), (9.05, y1), (9.05, y0)], closed=True,
                         facecolor=GREEN_F, edgecolor=TEAL, lw=1.4, zorder=3))
    label(ax, (7.85, 2.6), "decoder", fs=11.5, color="#222222")
    label(ax, (7.85, 4.55), "reconstruct", fs=10)

    arrow(ax, (9.1, 2.6), (9.6, 2.6), color=GREY, lw=1.3, head=9)
    # reconstruction
    _image_bar(ax, 9.65, 10.45, y0, y1, seed=1)
    label(ax, (10.05, 4.55), "reconstruction $\\hat{x}$", fs=11.5, color="#222222")

    # training objective: compare x-hat with x (arc dips below the network)
    arrow(ax, (10.05, y0 - 0.12), (1.3, y0 - 0.12), color=ORANGE, lw=1.4, head=10, conn="arc3,rad=-0.13")
    label(ax, (5.675, 0.05), r"loss = MSE($\hat{x}$, $x$): the target is the input itself, no labels needed",
          fs=10.5, color=ORANGE)
    save(fig, "autoen_architecture.png")


# ----------------------------------------------------------------------------
# 6. U-Net (4.6): contracting path, expanding path, skip connections
# ----------------------------------------------------------------------------
def fig_unet():
    fig, ax = canvas(8.4, 5.0, (0, 13.0), (0.15, 7.85))
    lev_y = [5.9, 4.45, 3.15, 2.0]          # level centres, top to bottom
    lev_h = [1.5, 1.05, 0.7, 0.45]          # block heights (feature-map size halves)
    res = ["64 × 64", "32 × 32", "16 × 16", "8 × 8"]
    w = 0.9
    enc_x = [2.4, 3.9, 5.4]
    bot_x = 6.8
    dec_x = [8.2, 9.7, 11.2]

    def block(cx, cy, h, fill, edge=TEAL):
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fill, edgecolor=edge, lw=1.4, zorder=3))

    # feature-map size at each level
    for y, s_ in zip(lev_y, res):
        label(ax, (0.3, y), s_, fs=9.5, ha="left")

    # blocks
    for i, cx in enumerate(enc_x):
        block(cx, lev_y[i], lev_h[i], BLUE_F)
    block(bot_x, lev_y[3], lev_h[3], ORANGE_F, edge=ORANGE)
    for i, cx in enumerate(dec_x):
        lvl = 2 - i
        block(cx, lev_y[lvl], lev_h[lvl], GREEN_F)

    # down arrows: bottom-right of block i -> top-left of block i+1
    for i in range(3):
        cx, lvl = enc_x[i], i
        nxt_x = enc_x[i + 1] if i < 2 else bot_x
        p0 = (cx + w / 2 - 0.05, lev_y[lvl] - lev_h[lvl] / 2 + 0.08)
        p1 = (nxt_x - w / 2 + 0.02, lev_y[lvl + 1] + lev_h[lvl + 1] / 2 - 0.05)
        arrow(ax, p0, p1, color=TEAL, lw=1.4, head=10)
    label(ax, (enc_x[1] - w / 2 - 0.12, lev_y[1] + 0.02), "conv +\ndownsample", fs=9, color=TEAL, ha="right")

    # up arrows: top-right of block -> bottom-left of the next block up
    for i, (cx, lvl) in enumerate([(bot_x, 3), (dec_x[0], 2), (dec_x[1], 1)]):
        nxt_x = dec_x[i]
        p0 = (cx + w / 2 - 0.05, lev_y[lvl] + lev_h[lvl] / 2 - 0.08)
        p1 = (nxt_x - w / 2 + 0.02, lev_y[lvl - 1] - lev_h[lvl - 1] / 2 + 0.05)
        arrow(ax, p0, p1, color=TEAL, lw=1.4, head=10)
    label(ax, (dec_x[1] + w / 2 + 0.12, lev_y[1] + 0.02), "upsample +\nconv", fs=9, color=TEAL, ha="left")

    # skip connections: encoder block -> matching decoder block, same level
    for lvl in range(3):
        ex, dx = enc_x[lvl], dec_x[2 - lvl]
        y = lev_y[lvl] + 0.18 * lev_h[lvl]
        arrow(ax, (ex + w / 2 + 0.05, y), (dx - w / 2 - 0.05, y), color=ORANGE, lw=1.4, ls=(0, (5, 3)), head=10)
    label(ax, (bot_x, lev_y[0] + 0.18 * lev_h[0] + 0.32), "skip connections: copy features across at every level",
          fs=10, color=ORANGE)

    # input / output
    arrow(ax, (enc_x[0] - w / 2 - 0.5, lev_y[0]), (enc_x[0] - w / 2 - 0.05, lev_y[0]), color=GREY, lw=1.3, head=9)
    label(ax, (enc_x[0] - 0.45, lev_y[0] + 1.0), "input", fs=10.5, color="#222222")
    arrow(ax, (dec_x[2] + w / 2 + 0.05, lev_y[0]), (dec_x[2] + w / 2 + 0.5, lev_y[0]), color=GREY, lw=1.3, head=9)
    label(ax, (dec_x[2] + 0.45, lev_y[0] + 1.0), "output", fs=10.5, color="#222222")

    # path titles
    label(ax, (enc_x[1], 7.4), "contracting path (encoder)", fs=11, color=TEAL, weight="bold")
    label(ax, (dec_x[1], 7.4), "expanding path (decoder)", fs=11, color=TEAL, weight="bold")
    label(ax, (bot_x, lev_y[3] - lev_h[3] / 2 - 0.35), "bottleneck", fs=10, color=ORANGE)

    label(ax, (bot_x, 0.75), "the deep path through the bottleneck carries context;\n"
                             "the skips carry high-resolution detail past it", fs=10, color="#222222")
    save(fig, "Unet.png")


if __name__ == "__main__":
    fig_tlu()
    fig_mlp_reg()
    fig_mlp_clf()
    fig_voting()
    fig_autoencoder()
    fig_unet()
