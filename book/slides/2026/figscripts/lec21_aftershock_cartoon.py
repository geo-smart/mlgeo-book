"""Slide figure for lecture 21, Part 1: the aftershock problem of DeVries et al. (2018) as inputs and output.

Cartoon (left panel is schematic, not data): around a large earthquake the
crust is cut into 5 km cells. For each cell the inputs are the six components
of the static stress change computed from the slip model of the mainshock, and
the observation is 1 if aftershocks were located in the cell within one year,
0 otherwise. DeVries et al. fed the six components to a network of 13,451
weights; Mignan & Broccardo (2019) fed one quantity, the maximum shear stress
change, to one neuron.

The skill numbers are read from figs/lit_aftershock_2018/numbers.json, written
by figscripts/aftershock_auc_rerun.py from the authors' open release (57 test
slip models): the share of all cells with aftershocks that fall in the 10% of
cells each model ranks highest.

Writes figs/lit_aftershock_2018/aftershock_inputs_output.png.
Regenerate: pixi run python book/slides/2026/figscripts/lec21_aftershock_cartoon.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figs" / "lit_aftershock_2018"
plt.rcParams.update({"font.size": 22, "axes.titlesize": 25, "axes.titleweight": "bold"})
BLUE, ORANGE, GREEN, RED, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c"
nums = json.loads((OUT / "numbers.json").read_text())["models"]
net = nums["Neural network, 13,451 parameters"]["share_of_aftershock_cells_in_top10pct"]
msh = nums["Maximum shear stress"]["share_of_aftershock_cells_in_top10pct"]

fig = plt.figure(figsize=(21, 7.0))
grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 2.1, 0.7], wspace=0.06)

# --- 1. the map, cut into cells (schematic) ----------------------------------------------
ax = fig.add_subplot(grid[0])
n = 16
edges = np.linspace(-40, 40, n + 1)
xc = (edges[:-1] + edges[1:]) / 2
X, Y = np.meshgrid(xc, xc)
shear = np.zeros_like(X)
for tip in (-20.0, 20.0):                       # stress change is largest off the two ends of the fault
    dx, dy = X - tip, Y
    r = np.hypot(dx, dy)
    shear += (0.35 + np.abs(np.cos(2 * np.arctan2(dy, dx)))) / (r + 6)
shear += 0.05 * (np.abs(Y) < 6) * (np.abs(X) < 20)
ax.pcolormesh(edges, edges, shear, cmap="Oranges", vmin=0, vmax=np.quantile(shear, 0.97), edgecolors="white", linewidth=0.8)
ax.plot([-20, 20], [0, 0], color=INK, lw=6, solid_capstyle="round")
ax.text(0, -4, "fault that slipped", ha="center", va="top", fontsize=19, color=INK,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
rng = np.random.default_rng(2)
p = (shear ** 2 / (shear ** 2).sum()).ravel()
pick = rng.choice(n * n, 80, p=p)
ax.scatter(X.ravel()[pick] + rng.uniform(-2, 2, 80), Y.ravel()[pick] + rng.uniform(-2, 2, 80), s=18, color=INK, zorder=3)
ix, iy = 13, 12
cell = Rectangle((edges[ix], edges[iy]), 5, 5, fc="none", ec=BLUE, lw=4.5, zorder=4)
ax.add_patch(cell)
ax.set(xlim=(-40, 40), ylim=(-40, 40), aspect="equal")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("5 km cells around\na large earthquake", fontsize=23)
ax.text(0, -43, "color: stress change\ndots: aftershocks (cartoon)", ha="center", va="top", fontsize=18, color=MUTED)

# --- 2. inputs, model, output ----------------------------------------------------------------
d = fig.add_subplot(grid[1])
d.set(xlim=(0, 24), ylim=(-1.2, 12), aspect="equal"); d.axis("off")
fig.add_artist(ConnectionPatch(xyA=(edges[ix] + 5, edges[iy] + 2.5), coordsA=ax.transData, xyB=(0.1, 8.6), coordsB=d.transData,
                               arrowstyle="-|>", mutation_scale=28, color=BLUE, lw=3))


def box(x, y, w, h, text, fc, ec, fontsize=20, weight="normal"):
    d.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", fc=fc, ec=ec, lw=2.5))
    d.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=INK)


def arrow(x0, y0, x1, y1, color=MUTED):
    d.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, color=color, lw=2.2))


OUTPUT = "aftershocks\nin this cell?"
# Row 1: DeVries et al. (2018)
d.text(0.6, 11.9, "DeVries et al. (2018): six inputs", fontsize=22, fontweight="bold", color=INK, va="top")
labels = ["Δσ$_{xx}$", "Δσ$_{yy}$", "Δσ$_{zz}$", "Δσ$_{xy}$", "Δσ$_{xz}$", "Δσ$_{yz}$"]
ys = np.linspace(10.6, 6.6, 6)
for yy, lab in zip(ys, labels):
    d.add_patch(Circle((1.0, yy), 0.32, fc="white", ec=BLUE, lw=2.5))
    d.text(1.6, yy, lab, va="center", fontsize=20, color=INK)
    arrow(3.4, yy, 6.9, 8.6)
d.text(0.6, 5.9, "the stress change in the cell (kPa),\nfrom the slip of the mainshock", fontsize=17, color=BLUE, va="top")
box(7.1, 7.7, 5.6, 1.8, "a network:\n13,451 weights", "#eeeeee", MUTED, fontsize=19)
arrow(13.0, 8.6, 15.4, 8.6)
box(15.7, 7.7, 7.6, 1.8, OUTPUT, "#d9efe6", GREEN, weight="bold")
d.text(19.5, 6.9, "observation: 1 if aftershocks were\nlocated in the cell, 0 otherwise", ha="center", va="top", fontsize=17, color=MUTED)

# Row 2: Mignan & Broccardo (2019)
d.text(0.6, 3.6, "Mignan & Broccardo (2019): one input", fontsize=22, fontweight="bold", color=INK, va="top")
d.add_patch(Circle((1.0, 1.6), 0.32, fc="white", ec=BLUE, lw=2.5))
d.text(1.6, 1.6, "maximum shear\nstress change", va="center", fontsize=20, color=INK)
arrow(6.9, 1.6, 9.5, 1.6)
d.add_patch(Circle((11.0, 1.6), 1.4, fc="#cfe3f1", ec=BLUE, lw=3))
d.text(11.0, 1.6, "one\nneuron", ha="center", va="center", fontsize=17, color=INK)
d.text(11.0, -0.05, "2 weights", ha="center", va="top", fontsize=17, color=MUTED)
arrow(12.5, 1.6, 15.4, 1.6)
box(15.7, 0.7, 7.6, 1.8, OUTPUT, "#d9efe6", GREEN, weight="bold")

# --- 3. how well each does --------------------------------------------------------------------
b = fig.add_subplot(grid[2])
vals = [100 * net, 100 * msh]
bars = b.bar([0, 1], vals, color=[MUTED, BLUE], width=0.65)
for bar, v in zip(bars, vals):
    b.text(bar.get_x() + bar.get_width() / 2, v - 3, f"{v:.0f}%", ha="center", va="top", fontsize=30, fontweight="bold", color="white")
b.set_xticks([0, 1]); b.set_xticklabels(["network", "one\nneuron"], fontsize=21)
b.set_ylim(0, 70); b.set_yticks([])
for s in ("top", "right", "left"):
    b.spines[s].set_visible(False)
b.set_title("Aftershock cells\nin the top 10%", fontsize=21)
fig.text(0.5, -0.04, "Bars: of all cells with aftershocks, the share among the 10% of cells each model ranks highest. 57 test earthquakes, 1.38 million cells.",
         ha="center", fontsize=17, color=MUTED)
fig.savefig(OUT / "aftershock_inputs_output.png", dpi=150, bbox_inches="tight")
print(f"network {net:.3f}, one neuron {msh:.3f}")
