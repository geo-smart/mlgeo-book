"""Literature timelines for the three network lectures (21b MLP, 22 CNN, 23 RNN).

Six dated cards per figure, linked by arrows; each card has a verdict band
(method: gray-blue; applied well: green; a warning: red), a small drawn icon and
one finding. Numbers on the cards come from the papers, except the aftershock
areas under the curve, which are the course's rerun (figs/lit_aftershock_2018/
numbers.json, figscripts/aftershock_auc_rerun.py), and LeNet-5's weight count,
printed by notebook 4.3. Full citations are in the slide credit lines.

Writes figs/lit_timelines/{mlp,cnn,rnn}_thread.png.
Regenerate: pixi run python book/slides/2026/figscripts/lit_timelines.py
"""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lit_timelines"
FIGS.mkdir(parents=True, exist_ok=True)
INK, MUTED, LIGHT = "#26241f", "#6e675c", "#e8e4dc"
BLUE, GREEN, RED, ORANGE, PURPLE, GRAY = "#0072B2", "#009E73", "#D55E00", "#E69F00", "#CC79A7", "#8a857c"
BAND = {"method": "#4a6f8a", "good": GREEN, "bad": RED}
af = json.loads((HERE.parent / "figs" / "lit_aftershock_2018" / "numbers.json").read_text())["models"]
AUC_NET = af["Neural network, 13,451 parameters"]["auc"]
AUC_ONE = af["Maximum shear stress"]["auc"]


# ------------------------------------------------------------------ icons: drawn in a box (x0, y0, w, h)
def ic_neuron(ax, x0, y0, w, h):
    cx, cy = x0 + 0.62 * w, y0 + h / 2
    for dy in (-0.3, 0, 0.3):
        ax.plot([x0 + 0.1 * w, cx - 0.12 * w], [cy + dy * h, cy], color=GRAY, lw=2)
    ax.add_patch(Circle((cx, cy), 0.13 * w, fc=BLUE, ec=INK))
    ax.plot([cx + 0.13 * w, x0 + 0.95 * w], [cy, cy], color=GRAY, lw=2)
    ax.plot([x0 + 0.78 * w, x0 + 0.85 * w, x0 + 0.85 * w, x0 + 0.95 * w], [cy - 0.3 * h, cy - 0.3 * h, cy - 0.12 * h, cy - 0.12 * h], color=INK, lw=2)


def ic_backprop(ax, x0, y0, w, h):
    L = [np.linspace(y0 + 0.2 * h, y0 + 0.8 * h, n) for n in (3, 4, 2)]
    xs = [x0 + 0.15 * w, x0 + 0.5 * w, x0 + 0.85 * w]
    for k in range(2):
        for a in L[k]:
            for b in L[k + 1]:
                ax.plot([xs[k], xs[k + 1]], [a, b], color=GRAY, lw=0.8)
    for x, ys, c in zip(xs, L, [GRAY, BLUE, GREEN]):
        for y in ys:
            ax.add_patch(Circle((x, y), 0.05 * w, fc=c, ec=INK))
    ax.add_patch(FancyArrowPatch((x0 + 0.9 * w, y0 + 0.05 * h), (x0 + 0.1 * w, y0 + 0.05 * h), arrowstyle="-|>",
                                 mutation_scale=16, color=RED, lw=2.5))


def ic_anyfunction(ax, x0, y0, w, h):
    t = np.linspace(0, 1, 200)
    target = 0.5 + 0.3 * np.sin(2 * np.pi * 1.3 * t)
    knots = np.linspace(0, 1, 7)
    approx = np.interp(t, knots, 0.5 + 0.3 * np.sin(2 * np.pi * 1.3 * knots))
    ax.plot(x0 + t * w, y0 + target * h, color=GRAY, lw=4, alpha=0.6)
    ax.plot(x0 + t * w, y0 + approx * h, color=BLUE, lw=2)


def ic_aftershock(ax, x0, y0, w, h):
    n = 7
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    P = np.exp(-((X * 1.3) ** 2 + (Y * 0.6) ** 2) * 1.5)
    cw, chh = w * 0.8 / n, h * 0.85 / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + 0.1 * w + j * cw, y0 + 0.05 * h + i * chh), cw, chh, fc=plt.cm.Oranges(0.15 + 0.8 * P[i, j]), ec="white", lw=0.5))
    ax.plot([x0 + 0.5 * w, x0 + 0.5 * w], [y0 + 0.25 * h, y0 + 0.75 * h], color=INK, lw=3)


def ic_bars(ax, x0, y0, w, h, vals, labels, colors, vmax=None, fmt="{:.3f}"):
    vmax = vmax or max(vals)
    bw = 0.8 * w / len(vals)
    for k, (v, lab, c) in enumerate(zip(vals, labels, colors)):
        bh = 0.62 * h * v / vmax
        ax.add_patch(Rectangle((x0 + 0.1 * w + k * bw + 0.1 * bw, y0 + 0.2 * h), 0.8 * bw, bh, fc=c))
        ax.text(x0 + 0.1 * w + k * bw + 0.5 * bw, y0 + 0.2 * h + bh + 0.03 * h, fmt.format(v), ha="center", va="bottom", fontsize=13)
        ax.text(x0 + 0.1 * w + k * bw + 0.5 * bw, y0 + 0.13 * h, lab, ha="center", va="top", fontsize=11.5)


def ic_tree(ax, x0, y0, w, h):
    def node(x, y):
        ax.add_patch(Circle((x, y), 0.05 * w, fc=GREEN, ec=INK))
    pts = [(0.5, 0.85), (0.3, 0.55), (0.7, 0.55), (0.18, 0.25), (0.42, 0.25), (0.6, 0.25), (0.82, 0.25)]
    for a, b in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
        ax.plot([x0 + pts[a][0] * w, x0 + pts[b][0] * w], [y0 + pts[a][1] * h, y0 + pts[b][1] * h], color=INK, lw=1.5)
    for p in pts:
        node(x0 + p[0] * w, y0 + p[1] * h)


def ic_neocognitron(ax, x0, y0, w, h):
    for k, (n, c) in enumerate([(8, GRAY), (4, BLUE), (2, GREEN)]):
        ys = np.linspace(y0 + 0.15 * h, y0 + 0.85 * h, n)
        x = x0 + (0.15 + 0.35 * k) * w
        for y in ys:
            ax.add_patch(Rectangle((x - 0.04 * w, y - 0.04 * h), 0.08 * w, 0.08 * h, fc=c))
        if k < 2:
            for i, y in enumerate(np.linspace(y0 + 0.15 * h, y0 + 0.85 * h, n // 2)):
                ax.add_patch(plt.Polygon([[x + 0.05 * w, ys[2 * i]], [x + 0.05 * w, ys[2 * i + 1]], [x + 0.3 * w, y]], fc=c, alpha=0.25, ec="none"))


def ic_lenet(ax, x0, y0, w, h):
    for k, (n, s, c) in enumerate([(1, 0.6, GRAY), (3, 0.45, "#9ecae1"), (5, 0.32, BLUE)]):
        for i in range(n):
            ax.add_patch(Rectangle((x0 + (0.05 + 0.24 * k) * w + i * 0.025 * w, y0 + (0.5 - s / 2) * h + i * 0.03 * h), s * 0.5 * w, s * h, fc=c, ec="white"))
    for i in range(6):
        ax.add_patch(Circle((x0 + 0.88 * w, y0 + (0.2 + 0.12 * i) * h), 0.03 * w, fc=GREEN, ec=INK))


def ic_quake(ax, x0, y0, w, h):
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 300)
    tr = 0.12 * rng.standard_normal(300)
    for t0 in (0.32, 0.47, 0.64, 0.85):
        tr += np.where(t > t0, np.sin(90 * (t - t0)) * np.exp(-(t - t0) * 40), 0) * (0.5 if t0 != 0.47 else 1.0)
    ax.plot(x0 + t * w, y0 + 0.6 * h + 0.25 * h * tr, color=GRAY, lw=1)
    for t0 in (0.32, 0.47, 0.64, 0.85):
        ax.plot([x0 + t0 * w] * 2, [y0 + 0.25 * h, y0 + 0.35 * h], color=GREEN, lw=4)
    ax.plot([x0 + 0.47 * w] * 2, [y0 + 0.08 * h, y0 + 0.18 * h], color=INK, lw=4)
    ax.text(x0 + 0.02 * w, y0 + 0.30 * h, "CNN", fontsize=11, color=GREEN, va="center")
    ax.text(x0 + 0.02 * w, y0 + 0.13 * h, "catalog", fontsize=11, color=INK, va="center")


def ic_phasenet(ax, x0, y0, w, h):
    t = np.linspace(0, 1, 300)
    rng = np.random.default_rng(5)
    tr = 0.1 * rng.standard_normal(300) + np.where(t > 0.3, np.sin(80 * (t - 0.3)) * np.exp(-(t - 0.3) * 12), 0) * 0.5 \
        + np.where(t > 0.6, np.sin(50 * (t - 0.6)) * np.exp(-(t - 0.6) * 8), 0)
    ax.plot(x0 + t * w, y0 + 0.68 * h + 0.18 * h * tr, color=GRAY, lw=1)
    ax.plot(x0 + t * w, y0 + 0.1 * h + 0.35 * h * np.exp(-((t - 0.3) / 0.02) ** 2), color=BLUE, lw=2.5)
    ax.plot(x0 + t * w, y0 + 0.1 * h + 0.35 * h * np.exp(-((t - 0.6) / 0.025) ** 2), color=RED, lw=2.5)
    ax.text(x0 + 0.3 * w, y0 + 0.47 * h, "P", color=BLUE, fontsize=13, ha="center", weight="bold")
    ax.text(x0 + 0.6 * w, y0 + 0.47 * h, "S", color=RED, fontsize=13, ha="center", weight="bold")


def ic_shortcut(ax, x0, y0, w, h):
    t = np.linspace(0, 1, 300)
    rng = np.random.default_rng(8)
    tr = 0.15 * rng.standard_normal(300)
    ax.plot(x0 + t * w, y0 + 0.55 * h + 0.3 * h * tr, color=GRAY, lw=1)
    ax.add_patch(Rectangle((x0 + 0.05 * w, y0 + 0.3 * h), 0.9 * w, 0.5 * h, fill=False, ec=RED, lw=2.5, ls="--"))
    ax.text(x0 + 0.5 * w, y0 + 0.08 * h, "\"event\": 93% of noise", ha="center", fontsize=12, color=RED)


def ic_lstm(ax, x0, y0, w, h):
    ax.add_patch(FancyArrowPatch((x0 + 0.02 * w, y0 + 0.75 * h), (x0 + 0.98 * w, y0 + 0.75 * h), arrowstyle="-|>", mutation_scale=16, color=ORANGE, lw=4))
    for xg, s in [(0.3, "×"), (0.6, "+")]:
        ax.add_patch(Circle((x0 + xg * w, y0 + 0.75 * h), 0.08 * w, fc="white", ec=INK, lw=1.5))
        ax.text(x0 + xg * w, y0 + 0.75 * h, s, ha="center", va="center", fontsize=14)
    for xg in (0.3, 0.55, 0.8):
        ax.add_patch(FancyBboxPatch((x0 + (xg - 0.08) * w, y0 + 0.18 * h), 0.16 * w, 0.22 * h, boxstyle="round,pad=0.01", fc=PURPLE, ec="none"))
        ax.plot([x0 + xg * w] * 2, [y0 + 0.4 * h, y0 + 0.66 * h], color=PURPLE, lw=1.5)


def ic_attention_matrix(ax, x0, y0, w, h, n=6, causal=False):
    rng = np.random.default_rng(3)
    A = rng.random((n, n)) ** 3 + np.eye(n) * 0.8
    A = A / A.sum(1, keepdims=True)
    s = min(0.85 * w, 0.85 * h) / n
    ox, oy = x0 + (w - n * s) / 2, y0 + (h - n * s) / 2
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((ox + j * s, oy + (n - 1 - i) * s), s, s, fc=plt.cm.YlOrBr(0.1 + 0.9 * A[i, j] / A.max()), ec="white", lw=0.5))


def ic_rnn_plus_attention(ax, x0, y0, w, h):
    for k in range(4):
        x = x0 + (0.1 + 0.22 * k) * w
        ax.add_patch(Rectangle((x, y0 + 0.2 * h), 0.12 * w, 0.2 * h, fc=BLUE))
        if k < 3:
            ax.add_patch(FancyArrowPatch((x + 0.12 * w, y0 + 0.3 * h), (x + 0.22 * w, y0 + 0.3 * h), arrowstyle="-|>", mutation_scale=10, color=BLUE))
        ax.plot([x + 0.06 * w, x0 + 0.5 * w], [y0 + 0.42 * h, y0 + 0.8 * h], color=ORANGE, lw=1 + 2.5 * [0.1, 0.6, 0.2, 0.1][k])
    ax.add_patch(Circle((x0 + 0.5 * w, y0 + 0.83 * h), 0.05 * w, fc=ORANGE, ec=INK))


def ic_transformer(ax, x0, y0, w, h):
    for k in range(3):
        ax.add_patch(FancyBboxPatch((x0 + 0.2 * w, y0 + (0.08 + 0.3 * k) * h), 0.6 * w, 0.22 * h, boxstyle="round,pad=0.01", fc=[ORANGE, PURPLE, ORANGE][k], ec="none"))
    ax.text(x0 + 0.5 * w, y0 + 0.19 * h, "attention", ha="center", va="center", fontsize=11, color="white", weight="bold")
    ax.text(x0 + 0.5 * w, y0 + 0.49 * h, "feed-forward", ha="center", va="center", fontsize=11, color="white", weight="bold")
    ax.text(x0 + 0.5 * w, y0 + 0.79 * h, "attention", ha="center", va="center", fontsize=11, color="white", weight="bold")


def ic_hydrograph(ax, x0, y0, w, h):
    t = np.linspace(0, 1, 200)
    q = 0.15 + 0.6 * np.exp(-((t - 0.3) / 0.06) ** 2) + 0.35 * np.exp(-((t - 0.7) / 0.09) ** 2)
    rain = np.zeros_like(t); rain[(t > 0.24) & (t < 0.28)] = 1; rain[(t > 0.62) & (t < 0.67)] = 0.6
    ax.bar(x0 + t * w, -0.25 * h * rain, width=w / 200, bottom=y0 + h, color=BLUE)
    ax.plot(x0 + t * w, y0 + 0.05 * h + 0.6 * h * q, color=INK, lw=2)
    ax.plot(x0 + t * w, y0 + 0.05 * h + 0.6 * h * (q * 0.95 + 0.02 * np.sin(20 * t)), color=GREEN, lw=2, ls="--")


def ic_globe(ax, x0, y0, w, h):
    cx, cy, r = x0 + w / 2, y0 + h / 2, 0.42 * min(w, h)
    ax.add_patch(Circle((cx, cy), r, fc="#d6e6f4", ec=INK, lw=1.5))
    for lat in (-0.5, 0, 0.5):
        ax.plot([cx - r * np.sqrt(1 - lat ** 2), cx + r * np.sqrt(1 - lat ** 2)], [cy + lat * r] * 2, color=MUTED, lw=1)
    for k in (-0.6, 0, 0.6):
        tt = np.linspace(-1, 1, 50)
        ax.plot(cx + k * r * np.sqrt(1 - tt ** 2), cy + r * tt, color=MUTED, lw=1)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.fill(cx + 0.35 * r + 0.25 * r * np.cos(th), cy + 0.2 * r + 0.18 * r * np.sin(th), color=PURPLE, alpha=0.6)


def ic_baseline(ax, x0, y0, w, h):
    t = np.linspace(0, 0.65, 120)
    y = 0.45 + 0.12 * np.sin(2 * np.pi * 2.2 * t) + 0.25 * t
    ax.plot(x0 + t * w, y0 + y * h, color=INK, lw=2)
    ax.plot([x0 + 0.65 * w, x0 + 0.98 * w], [y0 + y[-1] * h] * 2, color=GREEN, lw=3)
    ax.text(x0 + 0.82 * w, y0 + y[-1] * h + 0.08 * h, "tomorrow =\ntoday", ha="center", fontsize=11, color=GREEN)
    ax.plot([x0 + 0.65 * w] * 2, [y0 + 0.1 * h, y0 + 0.9 * h], color=MUTED, ls=":", lw=1.5)


# ------------------------------------------------------------------ the figure
def thread(cards, name):
    fig, ax = plt.subplots(figsize=(17.5, 8.0))
    ax.set_xlim(0, 64.2); ax.set_ylim(0, 29); ax.axis("off")
    cw, gap = 9.4, 1.32
    for k, c in enumerate(cards):
        x0 = 0.3 + k * (cw + gap)
        ax.add_patch(FancyBboxPatch((x0, 0.4), cw, 27.8, boxstyle="round,pad=0.05,rounding_size=0.5", fc="#faf7f2", ec=LIGHT, lw=1.5))
        ax.add_patch(FancyBboxPatch((x0, 24.6), cw, 3.6, boxstyle="round,pad=0.05,rounding_size=0.5", fc=BAND[c["kind"]], ec="none"))
        ax.text(x0 + cw / 2, 26.4, c["year"], ha="center", va="center", fontsize=27, color="white", weight="bold")
        ax.text(x0 + cw / 2, 23.0, c["head"], ha="center", va="top", fontsize=18, weight="bold", color=INK, linespacing=1.1)
        c["icon"](ax, x0 + 0.6, 10.8, cw - 1.2, 8.2)
        ax.text(x0 + 0.5, 9.6, "\n".join(textwrap.wrap(c["fact"], 20)), ha="left", va="top", fontsize=14.5, color=INK, linespacing=1.25)
        ax.text(x0 + 0.5, 1.0, "\n".join(textwrap.wrap(c["who"], 22)), ha="left", va="bottom", fontsize=12.5, color=MUTED, style="italic")
        if k < len(cards) - 1:
            ax.add_patch(FancyArrowPatch((x0 + cw - 0.1, 26.4), (x0 + cw + gap + 0.1, 26.4), arrowstyle="-|>", mutation_scale=20, color=INK, lw=2.5))
    fig.savefig(FIGS / f"{name}_thread.png", dpi=110, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print("wrote", name)


thread([
    dict(year="1958", kind="method", head="one neuron", icon=ic_neuron,
         fact="A weighted sum and a step; the weights change after each error.", who="Rosenblatt"),
    dict(year="1986", kind="method", head="hidden layers,\nlearned", icon=ic_backprop,
         fact="Backpropagation: the chain rule sends the error back through the layers.", who="Rumelhart, Hinton, Williams"),
    dict(year="1989", kind="method", head="any function", icon=ic_anyfunction,
         fact="One hidden layer with enough neurons can approximate any continuous function.", who="Cybenko; Hornik et al."),
    dict(year="2018", kind="good", head="aftershocks", icon=ic_aftershock,
         fact=f"13,451 weights read the stress change around a fault: AUC {AUC_NET:.3f}.", who="DeVries et al."),
    dict(year="2019", kind="bad", head="one neuron\nagain", icon=lambda ax, x0, y0, w, h: ic_bars(
        ax, x0, y0, w, h, [AUC_NET, AUC_ONE], ["network", "neuron"], [BLUE, GREEN], vmax=1.0),
         fact=f"Two weights on one stress input: AUC {AUC_ONE:.3f}. Always run the simple model.", who="Mignan & Broccardo"),
    dict(year="2022", kind="bad", head="tables", icon=ic_tree,
         fact="On medium-sized tables, tree ensembles still beat deep networks.", who="Grinsztajn et al."),
], "mlp")

thread([
    dict(year="1980", kind="method", head="local features", icon=ic_neocognitron,
         fact="Neocognitron: small receptive fields, layer after layer, tolerant to shifts.", who="Fukushima"),
    dict(year="1998", kind="method", head="learned\nkernels", icon=ic_lenet,
         fact="LeNet-5: convolution, pooling, a dense head, trained by backpropagation; 61,706 weights.", who="LeCun et al."),
    dict(year="2012", kind="method", head="at scale", icon=lambda ax, x0, y0, w, h: ic_bars(
        ax, x0, y0, w, h, [26.2, 15.3], ["2nd best", "CNN"], [GRAY, BLUE], fmt="{:.1f}%"),
         fact="GPUs and 1.2 million images: top-5 error 15.3% against 26.2%.", who="Krizhevsky et al."),
    dict(year="2018", kind="good", head="earthquakes", icon=ic_quake,
         fact="ConvNetQuake: 17 times more earthquakes than the Oklahoma catalog.", who="Perol, Gharbi, Denolle"),
    dict(year="2019", kind="good", head="phase picks", icon=ic_phasenet,
         fact="PhaseNet: a probability of P and of S at every sample.", who="Zhu & Beroza"),
    dict(year="2020", kind="bad", head="shortcuts", icon=ic_shortcut,
         fact="High scores for the wrong reason; our detector on real PNW noise (4.3).", who="Geirhos et al."),
], "cnn")

thread([
    dict(year="1997", kind="method", head="memory", icon=ic_lstm,
         fact="LSTM: a gated cell state; gradients survive long sequences.", who="Hochreiter & Schmidhuber"),
    dict(year="2014", kind="method", head="attention", icon=ic_rnn_plus_attention,
         fact="Attention added to a recurrent translator: look back at any word.", who="Bahdanau, Cho, Bengio"),
    dict(year="2017", kind="method", head="attention\nonly", icon=ic_transformer,
         fact="The Transformer drops the recurrence: attention and feed-forward layers.", who="Vaswani et al."),
    dict(year="2018", kind="bad", head="baselines\nfirst", icon=ic_baseline,
         fact="On the monthly M3 series, simple statistical methods beat the ML forecasters.", who="Makridakis et al."),
    dict(year="2019", kind="good", head="rivers", icon=ic_hydrograph,
         fact="One LSTM, 531 US basins: rain to streamflow, better than models calibrated basin by basin.", who="Kratzert et al."),
    dict(year="2023", kind="good", head="the weather", icon=ic_globe,
         fact="Pangu-Weather, a 3-D Transformer, beats the operational forecast on its own scores.", who="Bi et al."),
], "rnn")
