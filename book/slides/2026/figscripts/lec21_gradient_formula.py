"""Slide figure for lecture 21, Part 2: the gradient of the loss for one weight, for three activations.

For one sample i, a neuron computes the score z_i = sum_j w_j x_ij + b and the
prediction yhat_i = phi(z_i). By the chain rule, the gradient of the loss of
that sample with respect to weight w_j is

    dL_i/dw_j = dL_i/dyhat_i  x  phi'(z_i)  x  x_ij
                (error part)    (slope of     (input)
                                 activation)

and the gradient of the loss is the mean of these over the samples of the batch.
  no activation, squared error:  2 e_i x 1 x x_ij           = 2 (yhat_i - y_i) x_ij
  sigmoid, log-loss:             (yhat-y)/(yhat(1-yhat)) x yhat(1-yhat) x x_ij = (yhat_i - y_i) x_ij
  step:                          slope 0 (undefined at z = 0): gradient 0, no update.
                                 The 1958 rule replaces the slope by 1 by hand.

Writes figs/lec21_neuron/gradient_three_activations.png.
Regenerate: pixi run python book/slides/2026/figscripts/lec21_gradient_formula.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lec21_neuron"
plt.rcParams.update({
    "font.size": 22, "axes.titlesize": 25, "axes.titleweight": "bold",
    "axes.labelsize": 21, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "dejavusans",
})
BLUE, GREEN, RED, INK, MUTED = "#0072B2", "#009E73", "#D55E00", "#26241f", "#6e675c"

z = np.linspace(-5, 5, 801)
sig = 1 / (1 + np.exp(-z))
cases = [
    ("No activation", BLUE, z, np.ones_like(z), "slope = 1",
     r"$2\,e_i \;\times\; 1 \;\times\; x_{ij}$", r"= $2\, e_i\, x_{ij}$: error × input", "squared error"),
    ("Step: threshold unit", RED, (z >= 0).astype(float), np.zeros_like(z), "slope = 0",
     r"$\frac{\partial \mathcal{L}_i}{\partial \hat{y}_i} \;\times\; 0 \;\times\; x_{ij} \;=\; 0$", "no gradient: nothing moves", "any loss"),
    ("Sigmoid", GREEN, sig, sig * (1 - sig), r"slope = $\hat{y}(1-\hat{y})$",
     r"$\frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)} \times \hat{y}_i(1-\hat{y}_i) \times x_{ij}$", r"= $(\hat{y}_i - y_i)\, x_{ij}$: error × input", "log-loss"),
]

fig = plt.figure(figsize=(21, 7.4))
grid = fig.add_gridspec(3, 3, height_ratios=[0.8, 0.8, 1.0], hspace=0.55, wspace=0.16)
for col, (name, color, act, slope, slope_lab, formula, result, loss) in enumerate(cases):
    a = fig.add_subplot(grid[0, col])
    a.plot(z, act, color=color, lw=5)
    a.set_title(name, color=color, fontsize=28)
    a.set_ylabel("activation" if col == 0 else "", fontsize=22)
    a.set_xticklabels([])
    a.tick_params(labelsize=17)
    s = fig.add_subplot(grid[1, col])
    s.plot(z, slope, color=color, lw=5)
    s.set_ylim(-0.15, 1.3)
    s.set_ylabel("its slope" if col == 0 else "", fontsize=22)
    s.set_xlabel("score z", fontsize=20)
    s.tick_params(labelsize=17)
    s.text(0.97, 0.95 if col else 0.55, slope_lab, transform=s.transAxes, ha="right", va="top", fontsize=24, color=color)
    t = fig.add_subplot(grid[2, col])
    t.axis("off")
    t.text(0.5, 0.9, ("any loss:" if loss == "any loss" else f"{loss}:"), ha="center", va="top", fontsize=21, color=MUTED, transform=t.transAxes)
    t.text(0.5, 0.55, formula, ha="center", va="center", fontsize=28, color=INK, transform=t.transAxes)
    t.text(0.5, 0.08, result, ha="center", va="center", fontsize=26, color=color, fontweight="bold", transform=t.transAxes)
fig.savefig(FIGS / "gradient_three_activations.png", dpi=150, bbox_inches="tight")
print("written")
