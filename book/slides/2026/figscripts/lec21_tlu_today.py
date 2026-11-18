"""Slide figure for lecture 21: the threshold logic unit in today's research.

The threshold logic unit (TLU; McCulloch & Pitts 1943, trained by Rosenblatt's
rule in 1958) outputs 0 or 1, and its slope is 0 almost everywhere. Two active
research lines keep the step in the forward pass and train it anyway, by
replacing its slope in the backward pass with a stand-in (a "surrogate"):
  - binarized networks: activations (and weights) of +1 or -1 through the sign
    function; the straight-through estimator passes the gradient where |z| <= 1
    and cancels it beyond (Courbariaux et al. 2016, after Bengio et al. 2013);
  - spiking networks: a neuron emits a spike when its membrane potential
    crosses a threshold; the backward pass uses a smooth surrogate, here the
    fast-sigmoid derivative 1 / (1 + beta |z|)^2 (Zenke & Ganguli 2018; review by
    Neftci, Mostafa & Zenke 2019).
The curves are drawn from these formulas, not from data.

Writes figs/lec21_neuron/tlu_surrogate.png.
Regenerate: pixi run python book/slides/2026/figscripts/lec21_tlu_today.py
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
    "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, GREEN, RED, PURPLE, INK, MUTED = "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#26241f", "#6e675c"

z = np.linspace(-3, 3, 1201)
fig, (a, s) = plt.subplots(1, 2, figsize=(20, 7.2), gridspec_kw={"wspace": 0.22})
a.plot(z, (z >= 0).astype(float), color=RED, lw=5)
a.set(xlabel="score z", ylabel="output", ylim=(-0.12, 1.3), title="Forward pass: the step, 0 or 1")
a.text(-2.9, 1.17, "threshold logic unit:\nfires when z crosses the threshold", fontsize=20, color=INK, va="top")

s.plot(z, np.zeros_like(z), color=RED, lw=5, label="true slope: 0 (no gradient)")
s.plot(z, (np.abs(z) <= 1).astype(float), color=BLUE, lw=4.5, label="straight-through: 1 if |z| ≤ 1\n(binarized networks)")
s.plot(z, 1 / (1 + 4 * np.abs(z)) ** 2, color=GREEN, lw=4.5, label="fast-sigmoid surrogate\n(spiking networks)")
s.set(xlabel="score z", ylabel="slope used for training", ylim=(-0.12, 2.25), title="Backward pass: a stand-in slope")
s.legend(frameon=False, fontsize=18, loc="upper center", ncol=1)
fig.text(0.5, -0.04, "Curves drawn from the formulas in Courbariaux et al. (2016) and Zenke & Ganguli (2018), not from data.",
         ha="center", fontsize=17, color=MUTED)
fig.savefig(FIGS / "tlu_surrogate.png", dpi=150, bbox_inches="tight")
print("written")
