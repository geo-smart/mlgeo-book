"""Slide figure for lecture 21, Part 2: the chain rule, first on a borehole, then on the neuron.

Row 1, geoscience: along a borehole, temperature rises with depth (geothermal
gradient 25 °C/km, a typical continental value) and P-wave velocity falls
with temperature (-0.0005 km/s per °C, a typical value for crustal rock; both
round numbers chosen for the example, not fitted to data). The change of
velocity with depth is the product of the two links:
    dv/dz = dv/dT x dT/dz = (-0.0005) x 25 = -0.0125 km/s per km.

Row 2, the neuron of notebook 4.0 section 2 (travel times, random starting
weights seed 0), for the farthest station: the weight changes the prediction
by the input x, the prediction changes the loss of the sample by 2 x error.

Row 3, a neuron with an activation: one more link, the slope of the activation.

Writes figs/lec21_neuron/chain_rule.png and numbers_chain_rule.json.
Regenerate: pixi run python book/slides/2026/figscripts/lec21_chain_rule.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lec21_neuron"
plt.rcParams.update({"font.size": 22, "mathtext.fontset": "dejavusans"})
BLUE, GREEN, RED, PURPLE, INK, MUTED, BROWN = "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#26241f", "#6e675c", "#8c5a2b"

# the neuron of notebook 4.0, section 2
rng = np.random.default_rng(1)
distance = rng.uniform(5, 150, 100)
travel_time = 1.0 + distance / 6.0 + rng.normal(0, 0.5, 100)
d_train, y = distance[:80], travel_time[:80]
x = (d_train - d_train.mean()) / d_train.std()
r = np.random.default_rng(0)
w, b = r.normal(0, 1), r.normal(0, 1)
i = int(np.argmax(d_train))
yhat = w * x[i] + b
e = yhat - y[i]
link1, link2 = x[i], 2 * e
grad = link1 * link2
dw = 1e-4
measured = (((w + dw) * x[i] + b - y[i]) ** 2 - e ** 2) / dw
numbers = {"station_distance_km": float(d_train[i]), "x": float(x[i]), "prediction_s": float(yhat), "observation_s": float(y[i]),
           "error_s": float(e), "dyhat_dw": float(link1), "dL_dyhat": float(link2), "dL_dw": float(grad),
           "dL_dw_small_change": float(measured), "geotherm_degC_per_km": 25.0, "dv_dT_km_s_per_degC": -0.0005,
           "dv_dz_km_s_per_km": 25.0 * -0.0005}

fig, ax = plt.subplots(figsize=(21, 9.0))
ax.set(xlim=(0, 22.5), ylim=(0.6, 9.8)); ax.axis("off")


def node(xc, yc, text, color, w=3.0, h=1.1):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0.12", fc="white", ec=color, lw=3.5))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=23, color=INK)


def link(x0, x1, yc, top, bottom, color):
    ax.add_patch(FancyArrowPatch((x0, yc), (x1, yc), arrowstyle="-|>", mutation_scale=28, color=color, lw=3))
    ax.text((x0 + x1) / 2, yc + 0.18, top, ha="center", va="bottom", fontsize=20, color=color, linespacing=1.1)
    ax.text((x0 + x1) / 2, yc - 0.2, bottom, ha="center", va="top", fontsize=19, color=MUTED)


def result(xc, yc, text, color):
    ax.text(xc, yc, text, ha="left", va="center", fontsize=23, color=color, fontweight="bold")


XA, XB, XC, XR = 1.7, 8.3, 14.9, 16.9
# row 1: borehole
y1 = 8.1
ax.text(0.15, 9.55, "1. Along a borehole", fontsize=24, fontweight="bold", color=BROWN, va="center")
node(XA, y1, "depth z", BROWN)
link(XA + 1.65, XB - 1.8, y1, "× 25 °C/km", "dT/dz", BROWN)
node(XB, y1, "temperature T", BROWN, w=3.3)
link(XB + 1.8, XC - 1.65, y1, "× (−0.0005)\nkm/s per °C", "dv/dT", BROWN)
node(XC, y1, "velocity v", BROWN)
result(XR, y1, "dv/dz = 25 × (−0.0005)\n= −0.0125 km/s per km", BROWN)

# row 2: the neuron, one sample
y2 = 4.85
ax.text(0.15, 6.3, f"2. The neuron, one station ({d_train[i]:.0f} km)", fontsize=24, fontweight="bold", color=BLUE, va="center")
node(XA, y2, "weight w", BLUE)
link(XA + 1.65, XB - 1.8, y2, f"× input x\n= {link1:.2f}", r"$\partial \hat{y}/\partial w$", BLUE)
node(XB, y2, r"prediction $\hat{y}$", BLUE, w=3.3)
link(XB + 1.8, XC - 1.65, y2, f"× 2 × error\n= {link2:.1f}", r"$\partial \mathcal{L}/\partial \hat{y}$", RED)
node(XC, y2, r"loss $\mathcal{L}_i$", RED)
result(XR, y2, f"gradient\n= {link1:.2f} × ({link2:.1f})\n= {grad:.1f}", RED)

# row 3: with an activation
y3 = 1.6
ax.text(0.15, 3.05, "3. A neuron with an activation: one more link", fontsize=24, fontweight="bold", color=GREEN, va="center")
node(1.35, y3, "weight w", GREEN, w=2.4)
link(2.65, 4.65, y3, "× x", r"$\partial z/\partial w$", GREEN)
node(5.75, y3, "score z", GREEN, w=1.9)
link(6.85, 9.35, y3, r"× slope $\phi'(z)$", r"$\partial \hat{y}/\partial z$", GREEN)
node(10.75, y3, r"prediction $\hat{y}$", GREEN, w=2.6)
link(12.15, 14.0, y3, "× error\npart", r"$\partial \mathcal{L}/\partial \hat{y}$", RED)
node(14.9, y3, r"loss $\mathcal{L}_i$", RED, w=1.6)
result(XR, y3, "gradient = product\nof the three links", GREEN)
fig.text(0.5, 0.0, "Row 1: round values for the example. Row 2: notebook 4.0 section 2, random starting weights (seed 0); "
         f"a small change of w gives {measured:.1f}.", ha="center", fontsize=16, color=MUTED)
fig.savefig(FIGS / "chain_rule.png", dpi=150, bbox_inches="tight")
(FIGS / "numbers_chain_rule.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))

# =============================================================================
# Backpropagation for one neuron: the same station, forward then backward
# (figs/lec21_neuron/bp_single.png)
# =============================================================================
from matplotlib.patches import Circle  # noqa: E402

fig, ax = plt.subplots(figsize=(21, 8.2))
ax.set(xlim=(0, 21), ylim=(0.6, 8.9), aspect="equal"); ax.axis("off")
XX, XS, XY, XE = 1.6, 7.6, 13.0, 17.9
YF, YB = 6.5, 2.3


def circ(xc, yc, text, color, r=0.95, fs=24):
    ax.add_patch(Circle((xc, yc), r, fc="white", ec=color, lw=4, zorder=3))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, zorder=4, color=INK)


def arr(x0, x1, yc, color, top="", bottom="", fs=22):
    ax.add_patch(FancyArrowPatch((x0, yc), (x1, yc), arrowstyle="-|>", mutation_scale=30, color=color, lw=3.5, zorder=2))
    if top:
        ax.text((x0 + x1) / 2, yc + 0.25, top, ha="center", va="bottom", fontsize=fs, color=color)
    if bottom:
        ax.text((x0 + x1) / 2, yc - 0.25, bottom, ha="center", va="top", fontsize=19, color=MUTED)


# forward
ax.text(0.1, 8.65, "Forward pass: from the input to the error", fontsize=25, fontweight="bold", color=BLUE, va="center")
circ(XX, YF, f"{x[i]:.2f}", INK)
ax.text(XX, YF - 1.2, "input x", ha="center", va="top", fontsize=20, color=MUTED)
arr(XX + 1.0, XS - 1.0, YF, BLUE, f"× w = {w:.2f}\n+ b = {b:.2f}", "score z")
circ(XS, YF, f"{yhat:.2f}", BLUE)
arr(XS + 1.0, XY - 1.0, YF, BLUE, "no activation", "prediction ŷ = z")
circ(XY, YF, f"{yhat:.2f}", BLUE)
ax.text(XY, YF - 1.2, "prediction ŷ", ha="center", va="top", fontsize=20, color=MUTED)
arr(XY + 1.0, XE - 1.6, YF, BLUE)
ax.add_patch(FancyBboxPatch((XE - 1.5, YF - 0.95), 4.4, 1.9, boxstyle="round,pad=0.15", fc="#fbeee6", ec=RED, lw=3))
ax.text(XE + 0.7, YF + 0.45, f"observation {y[i]:.2f} s", ha="center", va="center", fontsize=20, color=INK)
ax.text(XE + 0.7, YF - 0.4, f"error {e:+.2f} s", ha="center", va="center", fontsize=24, color=RED, fontweight="bold")

# backward
ax.text(0.1, 4.3, "Backward pass: the error goes back, link by link", fontsize=25, fontweight="bold", color=RED, va="center")
ax.add_patch(FancyBboxPatch((XE - 1.5, YB - 0.95), 4.4, 1.9, boxstyle="round,pad=0.15", fc="#fbeee6", ec=RED, lw=3))
ax.text(XE + 0.7, YB, f"error {e:+.2f} s", ha="center", va="center", fontsize=24, color=RED, fontweight="bold")
arr(XE - 1.6, XY + 1.0, YB, RED, "× 2", "loss = error²")
circ(XY, YB, f"{2 * e:.1f}", RED)
arr(XY - 1.0, XS + 1.0, YB, RED, "× slope = 1", "no activation")
circ(XS, YB, f"{2 * e:.1f}", RED)
arr(XS - 1.0, XX + 1.0, YB, RED, f"× input x = {x[i]:.2f}")
circ(XX, YB, f"{grad:.1f}", RED)
ax.text(XX, YB - 1.2, f"gradient for w = {grad:.1f}", ha="center", va="top", fontsize=20, color=RED)
ax.text(XS, YB - 1.2, f"gradient for b = {2 * e:.1f}", ha="center", va="top", fontsize=20, color=RED)
fig.text(0.5, 0.02, f"Travel-time neuron of notebook 4.0 section 2, random starting weights (seed 0), station at {d_train[i]:.0f} km.",
         ha="center", fontsize=16, color=MUTED)
fig.savefig(FIGS / "bp_single.png", dpi=150, bbox_inches="tight")
plt.close(fig)
