"""Slide figures for lecture 21, Part 3: backpropagation, the errors sent back through the network.

Same synthetic ocean temperature profile and same 1-3-1 network as
lec21_network_training.py (depth in, 3 hidden tanh neurons, temperature out),
with the same random starting weights (seed 2). Inputs and targets are
standardized, as in training: depth x = (depth - 300) / 300, temperature in
units of its standard deviation.

One training sample is followed through one forward pass and one backward pass:
  forward:  z_j = W1_j x + b1_j,  h_j = tanh(z_j),  yhat = sum_j W2_j h_j + b2,
            error e = yhat - y,   loss of the sample L = e^2
  backward: output layer   dL/dW2_j = 2 e h_j,     dL/db2 = 2 e
            error sent back to hidden neuron j:  delta_j = 2 e * W2_j * (1 - h_j^2)
            hidden layer   dL/dW1_j = delta_j x,   dL/db1_j = delta_j
The script checks every gradient against a small change of the weight.

Figures, written to figs/lec21_neuron/ (numbers to numbers_backprop.json):
  bp_forward.png    the forward pass for one sample, with its numbers
  bp_backward.png   the backward pass: the error, sent back along each connection

Regenerate: pixi run python book/slides/2026/figscripts/lec21_backprop.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lec21_neuron"
plt.rcParams.update({"font.size": 22})
BLUE, ORANGE, GREEN, RED, PURPLE, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#26241f", "#6e675c"
SYN = "Synthetic ocean temperature profile and network of lecture 21, Part 3. Random starting weights (seed 2). Values in standardized units."


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


def profile(depth):
    return 4 + 6 * sigmoid(-(depth - 60) / 15) + 8 * sigmoid(-(depth - 300) / 50)


rng = np.random.default_rng(3)
depth = np.sort(rng.uniform(0, 600, 60))
temp = profile(depth) + rng.normal(0, 0.3, 60)
t_mean, t_std = temp.mean(), temp.std()

r = np.random.default_rng(2)                       # same starting weights as lec21_network_training.py
W1, b1 = r.normal(0, 1.0, 3), r.normal(0, 0.5, 3)
W2, b2 = r.normal(0, 0.5, 3), r.normal(0, 0.5)

i = int(np.argmin(np.abs(depth - 150)))            # one sample, near 150 m
x = (depth[i] - 300) / 300
y = (temp[i] - t_mean) / t_std


def forward(W1, b1, W2, b2):
    z = W1 * x + b1
    h = np.tanh(z)
    yhat = float(h @ W2 + b2)
    return z, h, yhat


z, h, yhat = forward(W1, b1, W2, b2)
e = yhat - y
L = e ** 2
grad_W2 = 2 * e * h
grad_b2 = 2 * e
slope = 1 - h ** 2
delta = 2 * e * W2 * slope
grad_W1 = delta * x
grad_b1 = delta

# check every gradient with a small change of the weight
eps = 1e-6
check = {}
for name, vec, grad in [("W1", W1, grad_W1), ("W2", W2, grad_W2), ("b1", b1, grad_b1)]:
    for j in range(3):
        v = vec.copy(); v[j] += eps
        args = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        args[name] = v
        _, _, yh = forward(args["W1"], args["b1"], args["W2"], args["b2"])
        check[f"{name}_{j + 1}"] = [float(grad[j]), float(((yh - y) ** 2 - L) / eps)]
numbers = {
    "sample_depth_m": float(depth[i]), "observed_degC": float(temp[i]), "x": float(x), "y": float(y),
    "predicted_degC": float(yhat * t_std + t_mean), "yhat": yhat, "error": float(e), "error_degC": float(e * t_std),
    "loss_of_sample": float(L), "W1": W1.tolist(), "b1": b1.tolist(), "W2": W2.tolist(), "b2": float(b2),
    "z": z.tolist(), "h": h.tolist(), "slope_tanh": slope.tolist(), "delta": delta.tolist(),
    "grad_W2": grad_W2.tolist(), "grad_b2": float(grad_b2), "grad_W1": grad_W1.tolist(), "grad_b1": grad_b1.tolist(),
    "check_analytic_vs_small_change": check, "t_std_degC": float(t_std),
}

# ---------------------------------------------------------------------------------------------
XIN, XH, XOUT, XOBS = 0.0, 6.0, 12.0, 16.2
YH = [2.6, 0.0, -2.6]
COLORS = [ORANGE, GREEN, PURPLE]
R = 0.78


def base(ax):
    ax.set(xlim=(-1.6, 19.6), ylim=(-4.3, 4.6), aspect="equal")
    ax.axis("off")
    ax.add_patch(Circle((XIN, 0), R, fc="white", ec=INK, lw=3, zorder=3))
    for yy, c in zip(YH, COLORS):
        ax.add_patch(Circle((XH, yy), R, fc="white", ec=c, lw=4, zorder=3))
    ax.add_patch(Circle((XOUT, 0), R, fc="white", ec=BLUE, lw=4, zorder=3))
    ax.text(XIN, -4.1, "input", ha="center", fontsize=21, color=MUTED)
    ax.text(XH, -4.1, "hidden layer (tanh)", ha="center", fontsize=21, color=MUTED)
    ax.text(XOUT, -4.1, "output", ha="center", fontsize=21, color=MUTED)


def edge(ax, p0, p1, color, lw=2.5, back=False, shrink=30):
    a, b = (p1, p0) if back else (p0, p1)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=30, color=color, lw=lw, shrinkA=shrink, shrinkB=shrink, zorder=2))


def label(ax, xy, text, color, fs=20, **kw):
    ax.text(*xy, text, color=color, fontsize=fs, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.95), zorder=4, **kw)


# --- forward pass ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 8.6))
base(ax)
ax.text(XIN, 0, f"{x:.2f}", ha="center", va="center", fontsize=21, zorder=5)
ax.text(XIN, 1.1, f"depth\n{depth[i]:.0f} m", ha="center", va="bottom", fontsize=20)
for j, (yy, c) in enumerate(zip(YH, COLORS)):
    edge(ax, (XIN, 0), (XH, yy), c)
    label(ax, ((XIN + XH) / 2, yy / 2 + 0.25), f"× {W1[j]:.2f}", c)
    ax.text(XH, yy, f"{h[j]:.2f}", ha="center", va="center", fontsize=20, zorder=5)
    edge(ax, (XH, yy), (XOUT, 0), c)
    label(ax, ((XH + XOUT) / 2, yy / 2 + 0.25), f"× {W2[j]:.2f}", c)
    ax.text(XH, yy + 0.75, f"tanh({W1[j]:.2f} × {x:.2f} + {b1[j]:.2f})", ha="center", va="bottom", fontsize=17, color=c)
ax.text(XOUT, 0, f"{yhat:.2f}", ha="center", va="center", fontsize=20, zorder=5)
ax.text(XOUT, 0.85, f"sum + {b2:.2f}", ha="center", va="bottom", fontsize=18, color=BLUE)
ax.add_patch(FancyBboxPatch((XOBS - 1.5, -1.9), 4.1, 3.8, boxstyle="round,pad=0.2", fc="#fbeee6", ec=RED, lw=3))
ax.text(XOBS + 0.55, 1.35, f"prediction {yhat:.2f}", ha="center", fontsize=20, color=BLUE)
ax.text(XOBS + 0.55, 0.55, f"observation {y:.2f}", ha="center", fontsize=20, color=INK)
ax.text(XOBS + 0.55, -0.45, f"error {e:+.2f}", ha="center", fontsize=24, color=RED, fontweight="bold")
ax.text(XOBS + 0.55, -1.35, f"({e * t_std:+.1f} °C)", ha="center", fontsize=19, color=RED)
edge(ax, (XOUT, 0), (XOBS - 1.6, 0), BLUE)
fig.text(0.5, 0.02, SYN, ha="center", fontsize=16, color=MUTED)
fig.savefig(FIGS / "bp_forward.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- backward pass -----------------------------------------------------------------------------
fig = plt.figure(figsize=(21, 8.4))
ax = fig.add_axes([0.0, 0.04, 0.44, 0.92])
ax.set(xlim=(-1.2, 15.4), ylim=(-3.9, 3.9), aspect="equal")
ax.axis("off")
ax.add_patch(Circle((XIN, 0), 1.0, fc="white", ec=INK, lw=3, zorder=3))
ax.text(XIN, 0, f"{x:.2f}", ha="center", va="center", fontsize=25, zorder=5)
ax.text(XIN, -1.3, "input", ha="center", va="top", fontsize=24, color=MUTED)
for j, (yy, c) in enumerate(zip(YH, COLORS)):
    ax.add_patch(Circle((XH, yy), 1.0, fc="white", ec=c, lw=4, zorder=3))
    ax.text(XH, yy, f"{delta[j]:+.2f}", ha="center", va="center", fontsize=25, color=RED, zorder=5)
    edge(ax, (XH, yy), (XOUT, 0), RED, lw=3.5, back=True, shrink=48)
    label(ax, ((XH + XOUT) / 2, yy / 2 + 0.45), f"× {W2[j]:.2f}", c, fs=25)
    edge(ax, (XIN, 0), (XH, yy), RED, lw=2.5, back=True, shrink=48)
ax.add_patch(Circle((XOUT, 0), 1.0, fc="white", ec=BLUE, lw=4, zorder=3))
ax.text(XOUT, 0, f"{2 * e:+.2f}", ha="center", va="center", fontsize=25, color=RED, zorder=5)
ax.text(XOUT, -1.3, "output:\n2 × error", ha="center", va="top", fontsize=23, color=RED)
ax.text(XOUT + 1.6, 0.45, f"error\n{e:+.2f}", ha="left", va="bottom", fontsize=25, color=RED, fontweight="bold")
edge(ax, (XOUT, 0), (XOUT + 3.4, 0), RED, lw=3.5, back=True, shrink=48)

tx = fig.add_axes([0.46, 0.04, 0.54, 0.92])
tx.axis("off")
tx.text(0.0, 1.0, "Error sent back to each hidden neuron =", fontsize=27, fontweight="bold", va="top")
tx.text(0.0, 0.91, "output error × weight of the connection × slope of tanh", fontsize=25, color=RED, va="top")
for j, c in enumerate(COLORS):
    y0 = 0.74 - 0.25 * j
    tx.text(0.0, y0, f"neuron {j + 1}:  {2 * e:.2f} × ({W2[j]:.2f}) × {slope[j]:.2f} = {delta[j]:+.2f}", fontsize=26, color=c, va="top", fontweight="bold")
    tx.text(0.04, y0 - 0.085, f"gradient, output weight:  {2 * e:.2f} × {h[j]:.2f} = {grad_W2[j]:+.2f}", fontsize=23, color=INK, va="top")
    tx.text(0.04, y0 - 0.15, f"gradient, input weight:   {delta[j]:+.2f} × {x:.2f} = {grad_W1[j]:+.2f}", fontsize=23, color=INK, va="top")
tx.text(0.0, -0.03, "Gradient of a weight = error at its end × value at its start", fontsize=23, color=MUTED, va="top")
fig.savefig(FIGS / "bp_backward.png", dpi=150, bbox_inches="tight")
plt.close(fig)

(FIGS / "numbers_backprop.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps({k: numbers[k] for k in ("sample_depth_m", "observed_degC", "predicted_degC", "error", "error_degC",
                                          "h", "slope_tanh", "delta", "grad_W2", "grad_W1")}, indent=1))
print("max |analytic - small change|:", max(abs(a - b) for a, b in check.values()))
