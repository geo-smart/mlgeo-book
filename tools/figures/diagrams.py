"""Four concept cartoons, drawn from scratch with matplotlib.

    pixi run python tools/figures/diagrams.py

Outputs (filenames and extensions are the ones the book already references):

    book/img/ValsetApproach.png                          3.8_robust_training
    book/img/Kfold.png                                   3.8_robust_training
    book/img/LOOCV.png                                   3.8_robust_training
    book/Chapter2-DataManipulation/hdf5_structure4.jpeg  2.2_data_formats_rendered
    book/Chapter1-GettingStarted/overview-github-collaboration.png
                                                         1.5_version_control_git

They replace figures copied from ISLR (Springer; the validation-set, K-fold
and LOOCV schematics), the HDF Group tutorial, and CU Boulder Earth Lab
(CC BY-NC-SA). Labels follow the lessons' own wording:
"training" / "validation", "fold", "round", the dataset and attribute names
the 2.2 code writes to data/temperature_anomaly.nc, and the fork -> clone ->
branch -> push -> pull request -> review -> merge list of 1.5.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DPI = 200

TEAL = "#116b66"
ORANGE = "#b3402a"
GREY = "#6e675c"
BLUE_FILL = "#cfe3ee"
GREEN_FILL = "#d9ead3"
ORANGE_FILL = "#f9cb9c"
INK = "#222222"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK,
        "savefig.facecolor": "white",
    }
)


# --------------------------------------------------------------------------- helpers
def blank_axes(fig, xlim, ylim):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    return ax


def rbox(ax, x, y, w, h, fill, edge, lw=1.2, pad=0.0, rounding=0.12, ls="-", z=2):
    """Rounded rectangle whose (x, y) is the lower-left corner."""
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def cell(ax, x, y, w, h, fill, edge, lw=0.8, z=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))


def arrow(ax, p0, p1, color=GREY, lw=1.3, style="-|>", ls="-", rad=0.0, z=3, ms=10):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=ms,
        color=color,
        linewidth=lw,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(a)
    return a


def legend_swatches(ax, x, y, items, gap=0.35, size=0.28, fs=8.5):
    """items: list of (fill, edge, label); laid out left to right from (x, y).

    Text widths are measured with the renderer so the items never overlap.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    cx = x
    for fill, edge, label in items:
        cell(ax, cx, y - size / 2, size, size, fill, edge)
        t = ax.text(cx + size + 0.08, y, label, va="center", ha="left", fontsize=fs)
        bb = t.get_window_extent(renderer=renderer)
        w_data = ax.transData.inverted().transform([(0, 0), (bb.width, 0)])
        cx += size + 0.08 + (w_data[1][0] - w_data[0][0]) + gap


# --------------------------------------------------------------------------- 0. validation set
def draw_valset(out: Path, n_cells: int = 12, n_val: int = 3, seed: int = 3):
    """One random split of all n samples into a training and a validation set."""
    import random

    rng = random.Random(seed)
    held = set(rng.sample(range(n_cells), n_val))

    fig = plt.figure(figsize=(6.4, 2.6))
    ax = blank_axes(fig, (0, 10), (0, 4.05))

    ax.text(0.15, 3.75, "Validation-set approach: a single random split", fontsize=11, weight="bold", va="center")

    x0, w, h = 1.55, 6.1, 0.46
    y_all = 2.85
    fw = w / n_cells
    labels = [str(i + 1) for i in range(n_cells - 2)] + ["…", "n"]
    cell(ax, x0, y_all, w, h, "white", INK, lw=1.0)
    ax.text(x0 - 0.12, y_all + h / 2, "all n samples", ha="right", va="center", fontsize=9)
    for k, lab in enumerate(labels):
        if k < n_cells - 1:
            ax.plot([x0 + (k + 1) * fw] * 2, [y_all, y_all + h], color=INK, lw=0.6)
        ax.text(x0 + (k + 0.5) * fw, y_all + h / 2, lab, ha="center", va="center", fontsize=8.5)
    arrow(ax, (x0 + w + 0.15, y_all + h / 2), (x0 + w + 0.55, y_all + h / 2), color=INK, lw=0.8, ms=7)
    ax.text(x0 + w + 0.62, y_all + h / 2, "sample index\n(time order)", va="center", ha="left", fontsize=7, color=GREY)

    # the one split, drawn at random
    y = 1.9
    arrow(ax, (x0 + w / 2, y_all - 0.06), (x0 + w / 2, y + h + 0.06), color=INK, lw=0.8, ms=7)
    ax.text(x0 + w / 2 + 0.12, (y_all + y + h) / 2, "shuffle, split once", va="center", ha="left", fontsize=7.5, color=GREY, style="italic")
    ax.text(x0 - 0.12, y + h / 2, "one split", ha="right", va="center", fontsize=9)
    for k in range(n_cells):
        is_val = k in held
        cell(
            ax,
            x0 + k * fw,
            y,
            fw,
            h,
            ORANGE_FILL if is_val else BLUE_FILL,
            ORANGE if is_val else TEAL,
            lw=1.0 if is_val else 0.7,
            z=3 if is_val else 2,
        )
    ax.text(x0 + w + 0.15, y + h / 2, "score 1", va="center", ha="left", fontsize=8.5, color=GREY)
    ax.text(x0 + w + 0.15, y - 0.08, "no spread", va="top", ha="left", fontsize=7.5, color=GREY, style="italic")

    ax.text(
        x0,
        y - 0.42,
        "Train on the blue samples, score on the orange ones:\none number, and no way to tell how noisy it is.",
        va="top",
        ha="left",
        fontsize=8.5,
        linespacing=1.4,
    )
    legend_swatches(
        ax,
        x0,
        0.38,
        [(BLUE_FILL, TEAL, "training set"), (ORANGE_FILL, ORANGE, "validation set (held out)")],
    )
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- 1. K-fold
def draw_kfold(out: Path, K: int = 5):
    fig = plt.figure(figsize=(6.4, 3.7))
    ax = blank_axes(fig, (0, 10), (0, 5.8))

    ax.text(0.15, 5.5, f"K-fold cross-validation, K = {K}", fontsize=11, weight="bold", va="center")

    # the full dataset, in time order
    x0, w, h = 1.55, 6.1, 0.46
    y_all = 4.55
    cell(ax, x0, y_all, w, h, "white", INK, lw=1.0)
    ax.text(x0 - 0.12, y_all + h / 2, "all n samples", ha="right", va="center", fontsize=9)
    fw = w / K
    for k in range(K):
        if k < K - 1:
            ax.plot([x0 + (k + 1) * fw] * 2, [y_all, y_all + h], color=INK, lw=0.6)
        ax.text(x0 + (k + 0.5) * fw, y_all + h / 2, f"fold {k + 1}", ha="center", va="center", fontsize=8.5)
    arrow(ax, (x0 + w + 0.15, y_all + h / 2), (x0 + w + 0.55, y_all + h / 2), color=INK, lw=0.8, ms=7)
    ax.text(x0 + w + 0.62, y_all + h / 2, "sample index\n(time order)", va="center", ha="left", fontsize=7, color=GREY)
    ax.text(x0, y_all - 0.12, "contiguous folds, no shuffling", va="top", ha="left", fontsize=7.5, color=GREY, style="italic")

    # one row per round
    row_h, row_gap = 0.46, 0.16
    y_top = 3.75
    for r in range(K):
        y = y_top - r * (row_h + row_gap)
        ax.text(x0 - 0.12, y + row_h / 2, f"round {r + 1}", ha="right", va="center", fontsize=9)
        for k in range(K):
            held = k == r
            cell(
                ax,
                x0 + k * fw,
                y,
                fw,
                row_h,
                ORANGE_FILL if held else BLUE_FILL,
                ORANGE if held else TEAL,
                lw=1.0 if held else 0.7,
                z=3 if held else 2,
            )
            if held:
                ax.text(x0 + (k + 0.5) * fw, y + row_h / 2, "validation", ha="center", va="center", fontsize=7.5, color=ORANGE, weight="bold")
        ax.text(x0 + w + 0.15, y + row_h / 2, f"score {r + 1}", va="center", ha="left", fontsize=8.5, color=GREY)

    # brace-ish line on the right and the outcome
    yb0 = y_top - (K - 1) * (row_h + row_gap)
    yb1 = y_top + row_h
    bx = x0 + w + 0.95
    ax.plot([bx, bx + 0.12, bx + 0.12, bx], [yb1, yb1, yb0, yb0], color=GREY, lw=0.8)
    ax.text(bx + 0.22, (yb0 + yb1) / 2, "mean\n± std", va="center", ha="left", fontsize=8.5, color=INK)

    ax.text(
        x0,
        yb0 - 0.22,
        "Every sample is validated exactly once, and trained on K − 1 times.",
        va="top",
        ha="left",
        fontsize=8.5,
    )
    legend_swatches(
        ax,
        x0,
        0.35,
        [(BLUE_FILL, TEAL, "training set"), (ORANGE_FILL, ORANGE, "validation set (held out)")],
    )
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- 2. LOOCV
def draw_loocv(out: Path, shown: int = 8):
    """Leave-one-out: K = n. Cells 1..shown, an ellipsis cell, then n."""
    n_cells = shown + 2  # shown + "..." + n
    fig = plt.figure(figsize=(6.4, 3.7))
    ax = blank_axes(fig, (0, 10), (0, 5.8))

    ax.text(0.15, 5.5, "Leave-one-out cross-validation, K = n", fontsize=11, weight="bold", va="center")

    x0, w, h = 1.55, 6.1, 0.46
    y_all = 4.55
    fw = w / n_cells
    labels = [str(i + 1) for i in range(shown)] + ["…", "n"]
    cell(ax, x0, y_all, w, h, "white", INK, lw=1.0)
    ax.text(x0 - 0.12, y_all + h / 2, "all n samples", ha="right", va="center", fontsize=9)
    for k, lab in enumerate(labels):
        if k < n_cells - 1:
            ax.plot([x0 + (k + 1) * fw] * 2, [y_all, y_all + h], color=INK, lw=0.6)
        ax.text(x0 + (k + 0.5) * fw, y_all + h / 2, lab, ha="center", va="center", fontsize=8.5)
    arrow(ax, (x0 + w + 0.15, y_all + h / 2), (x0 + w + 0.55, y_all + h / 2), color=INK, lw=0.8, ms=7)
    ax.text(x0 + w + 0.62, y_all + h / 2, "sample index\n(time order)", va="center", ha="left", fontsize=7, color=GREY)
    ax.text(x0, y_all - 0.12, "one held-out sample per round, n rounds", va="top", ha="left", fontsize=7.5, color=GREY, style="italic")

    row_h, row_gap = 0.46, 0.16
    y_top = 3.75
    # rows: round 1, 2, 3, ellipsis, round n
    rows = [("round 1", 0), ("round 2", 1), ("round 3", 2), None, ("round n", n_cells - 1)]
    for r, spec in enumerate(rows):
        y = y_top - r * (row_h + row_gap)
        if spec is None:
            ax.text(x0 - 0.12 - 0.3, y + row_h / 2, "⋮", ha="center", va="center", fontsize=12)
            ax.text(x0 + w / 2, y + row_h / 2, "⋮", ha="center", va="center", fontsize=12)
            ax.text(x0 + w + 0.15 + 0.35, y + row_h / 2, "⋮", ha="center", va="center", fontsize=12, color=GREY)
            continue
        name, held_idx = spec
        ax.text(x0 - 0.12, y + row_h / 2, name, ha="right", va="center", fontsize=9)
        for k in range(n_cells):
            held = k == held_idx
            cell(
                ax,
                x0 + k * fw,
                y,
                fw,
                row_h,
                ORANGE_FILL if held else BLUE_FILL,
                ORANGE if held else TEAL,
                lw=1.0 if held else 0.7,
                z=3 if held else 2,
            )
            if k == shown and not held:
                ax.text(x0 + (k + 0.5) * fw, y + row_h / 2, "…", ha="center", va="center", fontsize=8.5, color=TEAL)
        ax.text(x0 + w + 0.15, y + row_h / 2, "score " + name.split()[1], va="center", ha="left", fontsize=8.5, color=GREY)

    yb0 = y_top - 4 * (row_h + row_gap)
    yb1 = y_top + row_h
    bx = x0 + w + 0.95
    ax.plot([bx, bx + 0.12, bx + 0.12, bx], [yb1, yb1, yb0, yb0], color=GREY, lw=0.8)
    ax.text(bx + 0.22, (yb0 + yb1) / 2, "mean\n± std", va="center", ha="left", fontsize=8.5, color=INK)

    ax.text(
        x0,
        yb0 - 0.22,
        "n model fits; the held-out sample's neighbors are always in the training set.",
        va="top",
        ha="left",
        fontsize=8.5,
    )
    legend_swatches(
        ax,
        x0,
        0.35,
        [(BLUE_FILL, TEAL, "training set (n − 1 samples)"), (ORANGE_FILL, ORANGE, "validation set (1 sample)")],
    )
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- 3. HDF5 tree
def _array_glyph(ax, x, y, shape, size=0.34, z=4):
    """Tiny grid to say 'typed multidimensional array'. shape: 1 or 2 dims."""
    if len(shape) == 1:
        nr, nc = 1, 4
        cw, ch = size / 4, size / 4
        y = y + size * 3 / 8
    else:
        nr, nc = 3, 3
        cw, ch = size / 3, size / 3
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(
                Rectangle((x + j * cw, y + i * ch), cw, ch, facecolor="white", edgecolor=TEAL, linewidth=0.5, zorder=z)
            )


def _folder_glyph(ax, x, y, size=0.34, z=4):
    ax.add_patch(Rectangle((x, y), size, size * 0.7, facecolor="white", edgecolor=TEAL, linewidth=0.7, zorder=z))
    ax.add_patch(Rectangle((x, y + size * 0.7), size * 0.45, size * 0.18, facecolor="white", edgecolor=TEAL, linewidth=0.7, zorder=z))


def draw_hdf5(out: Path):
    fig = plt.figure(figsize=(9.6, 6.1))
    ax = blank_axes(fig, (0, 15), (0, 9.5))

    ax.text(0.3, 9.05, "The HDF5 model: groups, datasets, attributes", fontsize=12, weight="bold", va="center")
    ax.text(
        0.3,
        8.62,
        "data/temperature_anomaly.nc as h5py sees it: a tree that mimics a file system inside one file, "
        "and every node can carry metadata.",
        fontsize=8.5,
        color=GREY,
        va="center",
    )

    # node geometry: one row per node, file-browser indentation
    bh = 0.6
    rows = ["file", "root", "temperature_anomaly", "lat", "lon", "truth", "seasonal_pattern", "zonal_pattern"]
    y_pos = {name: 7.9 - i * 0.9 for i, name in enumerate(rows)}
    indent = {"file": 0.3, "root": 0.9, "temperature_anomaly": 1.5, "lat": 1.5, "lon": 1.5, "truth": 1.5, "seasonal_pattern": 2.1, "zonal_pattern": 2.1}
    widths = {"file": 3.1, "root": 2.5, "temperature_anomaly": 4.2, "lat": 2.9, "lon": 2.9, "truth": 2.5, "seasonal_pattern": 3.6, "zonal_pattern": 3.6}

    def node(name, kind, label, detail=None):
        x, yc, w = indent[name], y_pos[name], widths[name]
        if kind == "file":
            rbox(ax, x, yc - bh / 2, w, bh, "white", INK, lw=1.2)
            ax.text(x + 0.18, yc + 0.02, label, va="center", ha="left", fontsize=9.5, family="monospace", weight="bold")
            ax.text(x + w + 0.15, yc, "HDF5 file (a netCDF4 file is HDF5 underneath)", va="center", ha="left", fontsize=8, color=GREY, style="italic")
        elif kind == "group":
            rbox(ax, x, yc - bh / 2, w, bh, GREEN_FILL, TEAL, lw=1.2)
            _folder_glyph(ax, x + 0.14, yc - 0.17)
            ax.text(x + 0.62, yc + 0.02, label, va="center", ha="left", fontsize=9.5, family="monospace", weight="bold")
            ax.text(x + w + 0.15, yc, detail, va="center", ha="left", fontsize=8, color=GREY, style="italic")
        elif kind == "dataset":
            rbox(ax, x, yc - bh / 2, w, bh, BLUE_FILL, TEAL, lw=1.2)
            _array_glyph(ax, x + 0.14, yc - 0.17, detail)
            ax.text(x + 0.62, yc + 0.02, label, va="center", ha="left", fontsize=9.5, family="monospace", weight="bold")
            ax.text(x + w + 0.15, yc, f"dataset  shape {detail}  float64", va="center", ha="left", fontsize=8, color=GREY, style="italic")

    node("file", "file", "temperature_anomaly.nc")
    node("root", "group", "/", "root group")
    node("temperature_anomaly", "dataset", "temperature_anomaly", (40, 80))
    node("lat", "dataset", "lat", (40,))
    node("lon", "dataset", "lon", (80,))
    node("truth", "group", "truth/", "group")
    node("seasonal_pattern", "dataset", "seasonal_pattern", (40, 80))
    node("zonal_pattern", "dataset", "zonal_pattern", (40, 80))

    # tree connectors: vertical spine from the parent, elbows to the children
    def connect(parent, children):
        px = indent[parent] + 0.32
        py = y_pos[parent] - bh / 2
        last = y_pos[children[-1]]
        ax.plot([px, px], [py, last], color=GREY, lw=1.0, zorder=1)
        for c in children:
            cy = y_pos[c]
            ax.plot([px, indent[c]], [cy, cy], color=GREY, lw=1.0, zorder=1)

    connect("file", ["root"])
    connect("root", ["temperature_anomaly", "lat", "lon", "truth"])
    connect("truth", ["seasonal_pattern", "zonal_pattern"])

    # attributes: orange key = value tags in a column to the right, dashed leader to the node
    attrs = {
        "root": ['title = "Synthetic temperature anomaly, month 0"', 'source = "mlgeo_synth.climate_field"'],
        "temperature_anomaly": ['units = "degC"', 'long_name = "monthly temperature anomaly"'],
        "lat": ['units = "degrees_north"'],
        "lon": ['units = "degrees_east"'],
        "truth": ["trend_c_per_decade = 0.25"],
    }
    ax_x, ax_w = 9.4, 5.4
    for name, kv in attrs.items():
        yc = y_pos[name]
        n = len(kv)
        th = 0.28 * n + 0.14
        rbox(ax, ax_x, yc - th / 2, ax_w, th, ORANGE_FILL, ORANGE, lw=1.0, rounding=0.08)
        # small tab on the top edge naming the box
        cell(ax, ax_x + 0.12, yc + th / 2 - 0.02, 0.6, 0.17, ORANGE, ORANGE, lw=0.5, z=3)
        ax.text(ax_x + 0.42, yc + th / 2 + 0.065, "attrs", va="center", ha="center", fontsize=6.0, color="white", weight="bold", zorder=4)
        for i, line in enumerate(kv):
            ax.text(ax_x + 0.16, yc + th / 2 - 0.21 - 0.28 * i, line, va="center", ha="left", fontsize=7.4, family="monospace")
        ax.plot([ax_x - 0.45, ax_x], [yc, yc], color=ORANGE, lw=0.9, ls=(0, (2, 2)), zorder=1)
        ax.plot([ax_x - 0.45], [yc], marker="o", ms=3, color=ORANGE, zorder=2)

    # legend and footnote
    legend_swatches(
        ax,
        0.3,
        0.75,
        [
            (GREEN_FILL, TEAL, "group: holds datasets and other groups"),
            (BLUE_FILL, TEAL, "dataset: a typed N-dimensional array"),
            (ORANGE_FILL, ORANGE, "attributes: key = value metadata on any node"),
        ],
        gap=0.3,
        size=0.3,
        fs=8.2,
    )
    ax.text(
        0.3,
        0.3,
        "The file the lesson writes has only the root group; truth/ (the noise-free components returned by "
        "mlgeo_synth.climate_field) shows how a group nests inside another.",
        fontsize=7.5,
        color=GREY,
        va="center",
    )
    fig.savefig(out, dpi=DPI, pil_kwargs={"quality": 95})
    plt.close(fig)


# --------------------------------------------------------------------------- 4. GitHub loop
def draw_github(out: Path):
    fig = plt.figure(figsize=(10.0, 5.25))
    ax = blank_axes(fig, (0, 20), (0, 10.5))

    ax.text(0.4, 10.1, "Contributing by pull request: the loop", fontsize=12, weight="bold", va="center")

    # where things live
    band_y0, band_y1 = 0.7, 9.7
    rbox(ax, 0.4, band_y0, 12.3, band_y1 - band_y0, "#f4f3f0", "#d8d4cc", lw=0.8, rounding=0.25, z=0)
    ax.text(0.7, band_y1 - 0.4, "on GitHub", fontsize=9, color=GREY, va="center", weight="bold")
    rbox(ax, 13.1, band_y0, 6.5, band_y1 - band_y0, "#f4f3f0", "#d8d4cc", lw=0.8, rounding=0.25, z=0)
    ax.text(19.3, band_y1 - 0.4, "on your computer", fontsize=9, color=GREY, va="center", ha="right", weight="bold")

    # three repositories
    bw, bh = 4.9, 4.5
    y0 = 2.7
    bx = {"orig": 0.9, "fork": 7.0, "local": 14.0}

    def repo(x, title, subtitle, fill, edge):
        rbox(ax, x, y0, bw, bh, fill, edge, lw=1.4, rounding=0.2)
        ax.text(x + bw / 2, y0 + bh - 0.4, title, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + bw / 2, y0 + bh - 0.8, subtitle, ha="center", va="center", fontsize=7.4, color=GREY, style="italic")

    repo(bx["orig"], "original repository", "someone else's; you cannot push to it", GREEN_FILL, TEAL)
    repo(bx["fork"], "your fork", "your copy on GitHub; the remote origin", ORANGE_FILL, ORANGE)
    repo(bx["local"], "local clone", "on your disk; syncs only when you run git", BLUE_FILL, TEAL)

    # branch sketches: a main line and the fix-readme branch
    def branch_sketch(x, merged=False):
        xl, xr = x + 0.5, x + bw - 0.5
        ym = y0 + 0.7
        yb = ym + 0.65
        ax.plot([xl, xr], [ym, ym], color=INK, lw=1.4, zorder=3)
        for cx in (xl + 0.25, xl + 0.95, xl + 1.65):
            ax.plot(cx, ym, "o", ms=4.5, color=INK, zorder=4)
        ax.text(xl, ym - 0.28, "main", fontsize=7.2, va="top", ha="left", family="monospace")
        b0 = xl + 1.65
        ax.plot([b0, b0 + 0.5, b0 + 1.9], [ym, yb, yb], color=ORANGE, lw=1.4, zorder=3)
        for cx in (b0 + 0.9, b0 + 1.5):
            ax.plot(cx, yb, "o", ms=4.5, color=ORANGE, zorder=4)
        ax.text(b0 + 0.5, yb + 0.16, "fix-readme", fontsize=7.2, va="bottom", ha="left", family="monospace", color=ORANGE)
        if merged:
            ax.plot([b0 + 1.9, b0 + 2.4], [yb, ym], color=ORANGE, lw=1.4, zorder=3)
            ax.plot(b0 + 2.4, ym, "o", ms=4.5, color=INK, zorder=4)

    branch_sketch(bx["orig"], merged=True)
    branch_sketch(bx["fork"])
    branch_sketch(bx["local"])

    # numbered steps that happen inside a repository
    ax.text(
        bx["local"] + 0.35,
        y0 + bh - 1.2,
        "3  git switch -c fix-readme\n   edit, git add, git commit",
        fontsize=7.4,
        va="top",
        ha="left",
        family="monospace",
        linespacing=1.5,
    )
    ax.text(
        bx["orig"] + 0.35,
        y0 + bh - 1.2,
        "6  review: reviewers read the\n   diff; owners accept or\n   request changes\n7  merge: your change is in main",
        fontsize=7.4,
        va="top",
        ha="left",
        family="monospace",
        linespacing=1.5,
    )

    # elbow arrows between repositories, with the step label on the horizontal run
    def elbow(x_from, x_to, y_edge, y_run, label, sub, color, above):
        ax.plot([x_from, x_from, x_to], [y_edge, y_run, y_run], color=color, lw=1.6, zorder=3, solid_capstyle="round")
        arrow(ax, (x_to, y_run), (x_to, y_edge), color=color, lw=1.6, ms=14)
        xm = (x_from + x_to) / 2
        if above:
            ax.text(xm, y_run + 0.14, label, ha="center", va="bottom", fontsize=8.5, weight="bold", family="monospace")
            if sub:
                ax.text(xm, y_run + 0.6, sub, ha="center", va="bottom", fontsize=7.2, color=GREY, style="italic")
        else:
            ax.text(xm, y_run - 0.14, label, ha="center", va="top", fontsize=8.5, weight="bold", family="monospace")
            if sub:
                ax.text(xm, y_run - 0.6, sub, ha="center", va="top", fontsize=7.2, color=GREY, style="italic")

    top, bot = y0 + bh, y0
    y_run_top, y_run_bot = top + 1.0, bot - 0.9
    # 1 fork, 2 clone: rightwards along the top
    elbow(bx["orig"] + bw * 0.72, bx["fork"] + bw * 0.28, top, y_run_top, "1  fork", "the button on GitHub; later: Sync fork", TEAL, True)
    elbow(bx["fork"] + bw * 0.72, bx["local"] + bw * 0.28, top, y_run_top, "2  git clone <your fork>", "later: git pull", TEAL, True)
    # 4 push, 5 pull request: leftwards along the bottom
    elbow(bx["local"] + bw * 0.28, bx["fork"] + bw * 0.72, bot, y_run_bot, "4  git push -u origin fix-readme", None, ORANGE, False)
    elbow(bx["fork"] + bw * 0.28, bx["orig"] + bw * 0.72, bot, y_run_bot, "5  open a pull request", "in the browser, or gh pr create", ORANGE, False)

    ax.text(
        0.4,
        0.3,
        "With write access to the original repository, skip the fork: clone it, branch, push the branch, and open the pull request from there.",
        fontsize=7.8,
        color=GREY,
        va="center",
    )
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main(argv=None):
    """Optional argv[0]: a directory to write previews into instead of the book."""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    base = Path(argv[0]).resolve() if argv else ROOT
    targets = {
        draw_valset: base / "book" / "img" / "ValsetApproach.png",
        draw_kfold: base / "book" / "img" / "Kfold.png",
        draw_loocv: base / "book" / "img" / "LOOCV.png",
        draw_hdf5: base / "book" / "Chapter2-DataManipulation" / "hdf5_structure4.jpeg",
        draw_github: base / "book" / "Chapter1-GettingStarted" / "overview-github-collaboration.png",
    }
    for fn, out in targets.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        fn(out)
        print(out)


if __name__ == "__main__":
    main()
