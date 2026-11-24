"""Slide figure shared by the Chapter 4 decks: the structure of the data chooses the network.

Rows follow Battaglia et al. (2018, arXiv:1806.01261, Table 1: entities, relations, relational
inductive bias, invariance), redrawn with the geoscience examples of this book. Schematic, no data.

Writes figs/lec_framework/framework_all.png and one version per network with its row highlighted:
framework_fcn.png, framework_cnn.png, framework_rnn.png.

Also writes the statement that opens and closes every Chapter 4 deck, the canonical networks and
what each is good for: canonical_all.png, and canonical_fcn/cnn/rnn/gnn.png with the deck's rows
highlighted (rnn highlights recurrent and attention, the two of notebook 4.4).
Regenerate: pixi run python book/slides/2026/figscripts/lec_framework.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lec_framework"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 20})
INK, MUTED, LIGHT = "#26241f", "#6e675c", "#d9d4ca"
COLORS = {"fcn": "#0072B2", "cnn": "#009E73", "rnn": "#D55E00", "gnn": "#CC79A7"}

ROWS = [
    ("fcn", "Table of features:\nno order", "", "Fully connected", "every input to\nevery neuron", "nothing assumed",
     "seismic event: 61 features (4.1)\nGNSS: 6 summary numbers (4.2)"),
    ("cnn", "Grid: waveform, map;\nneighbors related", "", "Convolutional", "local, same weights\nslid along", "locality; a pattern\nanywhere",
     "detector on waveforms (4.3)\ntrend in a gridded field (4.3)"),
    ("rnn", "Sequence in time:\norder matters", "", "Recurrent or\nattention", "past to present,\nsame rule each day", "order; the same rule\nat every time",
     "GNSS: 90 days in, 30 out (4.4)\nforecasting shootout (4.9)"),
    ("gnn", "Stations with links:\nany neighbors", "", "Graph network", "along the edges\nof a graph", "given relations;\nany station order",
     "shaking at stations\nwithout a recording (4.5)"),
]


def icon(ax, kind, x, y, color):
    """Small schematic of the data structure, centered on (x, y)."""
    if kind == "fcn":
        for r in range(3):
            for c in range(4):
                ax.add_patch(Rectangle((x - 0.8 + 0.4 * c, y + 0.25 - 0.3 * r), 0.36, 0.26, fc="white", ec=color, lw=1.8))
    elif kind == "cnn":
        t = np.linspace(-0.9, 0.9, 200)
        ax.plot(x + t, y + 0.3 * np.exp(-((t - 0.1) / 0.15) ** 2) * np.cos(30 * t) + 0.04 * np.sin(90 * t), color=color, lw=2)
        ax.add_patch(Rectangle((x - 0.15, y - 0.4), 0.5, 0.8, fc="none", ec=color, lw=2.5, ls="--"))
    elif kind == "rnn":
        for k in range(5):
            ax.add_patch(Circle((x - 0.8 + 0.4 * k, y), 0.12, fc="white", ec=color, lw=2))
            if k < 4:
                ax.add_patch(FancyArrowPatch((x - 0.66 + 0.4 * k, y), (x - 0.54 + 0.4 * k, y), arrowstyle="-|>", mutation_scale=10, color=color, lw=1.5))
    elif kind == "att":
        for k in range(5):
            ax.add_patch(Circle((x - 0.8 + 0.4 * k, y - 0.12), 0.12, fc="white", ec=color, lw=2, zorder=3))
            if k < 4:
                ax.add_patch(FancyArrowPatch((x - 0.8 + 0.4 * k, y), (x + 0.8, y), connectionstyle="arc3,rad=-0.5",
                                             arrowstyle="-", color=color, lw=1.4 + 0.5 * k))
    else:
        pts = np.array([[-0.7, 0.2], [-0.2, 0.35], [0.4, 0.25], [-0.4, -0.25], [0.2, -0.3], [0.75, -0.15]])
        for a, b in [(0, 1), (1, 2), (0, 3), (3, 4), (1, 4), (4, 5), (2, 5)]:
            ax.plot([x + pts[a, 0], x + pts[b, 0]], [y + pts[a, 1], y + pts[b, 1]], color=color, lw=1.8)
        for p in pts:
            ax.add_patch(Circle((x + p[0], y + p[1]), 0.08, fc="white", ec=color, lw=2, zorder=3))


def draw(highlight=None, name="framework_all"):
    fig, ax = plt.subplots(figsize=(21, 8.3))
    ax.set(xlim=(0, 22.4), ylim=(0.5, 9.4)); ax.axis("off")
    heads = [(1.9, "Data"), (5.6, "Network"), (9.7, "Connections"), (13.9, "Assumes"), (19.2, "In this book")]
    for x, h in heads:
        ax.text(x, 9.05, h, ha="center", va="center", fontsize=26, fontweight="bold", color=INK)
    for r, (kind, data, rel, net, conn, assume, examples) in enumerate(ROWS):
        y = 7.5 - r * 2.05
        on = highlight is None or highlight == kind
        color = COLORS[kind] if on else LIGHT
        text = INK if on else "#b5afa4"
        ax.add_patch(FancyBboxPatch((0.15, y - 0.92), 22.1, 1.84, boxstyle="round,pad=0.05",
                                    fc=(COLORS[kind] + "1f") if on else "#f4f2ee", ec=color if (highlight == kind) else "none", lw=3.5))
        icon(ax, kind, 1.9, y + 0.35, color)
        ax.text(1.9, y - 0.5, data, ha="center", va="center", fontsize=18, color=text)
        ax.text(5.6, y, net, ha="center", va="center", fontsize=26, fontweight="bold", color=color if on else text)
        ax.text(9.7, y, conn, ha="center", va="center", fontsize=22, color=text)
        ax.text(13.9, y, assume, ha="center", va="center", fontsize=22, color=text)
        ax.text(19.2, y, examples, ha="center", va="center", fontsize=19, color=text)
    fig.text(0.5, 0.06, "After Battaglia et al. (2018), Table 1: relational inductive biases of standard deep-learning components.",
             ha="center", fontsize=16, color=MUTED)
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


CANON = [
    ("fcn", "fcn", "Table of features", "Fully connected", "4.1, 4.2", "nothing; every input\nto every neuron",
     "computed features:\ncatalogs, lab data", "linear or logistic\nregression; a forest"),
    ("cnn", "cnn", "Waveform, map", "Convolutional", "4.3", "local patterns, the\nsame anywhere",
     "detecting, picking;\nmaps and images", "STA/LTA, a template;\nfit with known terms"),
    ("rnn", "rnn", "Sequence in time", "Recurrent, LSTM", "4.4, 4.9", "order; a memory;\none rule each step",
     "forecasts from the recent\npast: GNSS, rivers", "persistence, seasonal\nnaive, SARIMA"),
    ("att", "att", "Long sequence, set", "Attention", "4.4, 4.9", "every element can\nread every other",
     "long context; pretrained\nweather models", "persistence, SARIMA;\na recurrent network"),
    ("gnn", "gnn", "Linked stations", "Graph network", "4.5",
     "relations along\nthe edges", "station networks,\nweather meshes", "spatial\ninterpolation"),
]
CCOL = dict(COLORS, att="#E69F00")


def canonical(highlight=(), name="canonical_all"):
    fig, ax = plt.subplots(figsize=(21, 7.4))
    ax.set(xlim=(0, 22.4), ylim=(0.35, 8.5)); ax.axis("off")
    heads = [(1.9, "Data"), (5.8, "Network"), (10.1, "Assumes"), (14.8, "Good for"), (19.7, "Beat first")]
    for x, h in heads:
        ax.text(x, 8.2, h, ha="center", va="center", fontsize=26, fontweight="bold", color=INK)
    for r, (kind, ik, data, net, nb, assume, good, base) in enumerate(CANON):
        y = 7.15 - r * 1.5
        on = not highlight or kind in highlight
        color = CCOL[kind] if on else LIGHT
        text = INK if on else "#b5afa4"
        ax.add_patch(FancyBboxPatch((0.15, y - 0.66), 22.1, 1.32, boxstyle="round,pad=0.05",
                                    fc=(CCOL[kind] + "1f") if on else "#f4f2ee",
                                    ec=color if kind in highlight else "none", lw=3.5))
        icon(ax, ik, 1.9, y + 0.24, color)
        ax.text(1.9, y - 0.5, data, ha="center", va="center", fontsize=16, color=text)
        ax.text(5.8, y + 0.15, net, ha="center", va="center", fontsize=24, fontweight="bold", color=color if on else text)
        ax.text(5.8, y - 0.42, nb, ha="center", va="center", fontsize=17, color=MUTED if on else text)
        ax.text(10.1, y, assume, ha="center", va="center", fontsize=19, color=text)
        ax.text(14.8, y, good, ha="center", va="center", fontsize=19, color=text)
        ax.text(19.7, y, base, ha="center", va="center", fontsize=19, color=text)
    fig.text(0.5, 0.02, "Assumptions after Battaglia et al. (2018), Table 1. Book notebooks under each network.",
             ha="center", fontsize=16, color=MUTED)
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


draw()
for kind in ("fcn", "cnn", "rnn"):
    draw(kind, f"framework_{kind}")
canonical()
canonical(("fcn",), "canonical_fcn")
canonical(("cnn",), "canonical_cnn")
canonical(("rnn", "att"), "canonical_rnn")
canonical(("gnn",), "canonical_gnn")
print("written")
