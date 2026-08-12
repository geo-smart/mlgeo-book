"""Slide figure: how the learning rate changes gradient descent (lecture 21).

Re-plots the 2x2 learning-rate grid from book notebook 4.0 (section 4) at
lecture scale: same data (numpy default_rng seeds 1), same zero-initialized
single-neuron linear model, same four learning rates. Panel titles name the
concept, not the parameter value alone.

Regenerate: pixi run python book/slides/2026/figscripts/lec21_learning_rate_grid.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

# --- the notebook's noisy-line dataset (mlgeo_4.0_perceptrons.ipynb, section 3) ---
num = 100
rng = np.random.default_rng(1)
x = np.linspace(0, 5, num) + (rng.random(num) * 2 - 1) * 0.5
y = np.linspace(0, 2.5, num) + (rng.random(num) * 2 - 1) * 0.5
p = rng.permutation(num)
x, y = x[p], y[p]
X = x[:50].reshape(-1, 1)
t = y[:50]


def gradient_descent(lr, n_iter=2000, tol=1e-8):
    """The notebook's loop: single linear neuron, MSE cost, zero start."""
    w, b = np.zeros(1), 0.0
    costs, prev = [], None
    for _ in range(n_iter):
        pred = X @ w + b
        cost = np.mean((pred - t) ** 2)
        if not np.isfinite(cost):
            break
        if prev is not None and abs(prev - cost) <= tol:
            break
        prev = cost
        costs.append(cost)
        er = t - pred
        w = w + lr * (X.T @ er)
        b = b + lr * er.sum()
    return np.array(costs)


panels = [
    (1e-5, "Too small: still falling"),
    (1e-4, "Small: slow convergence"),
    (1e-3, "Right: converges fast"),
    (1e-2, "Too large: diverges"),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 8.4), sharex=True)
with np.errstate(over="ignore", invalid="ignore"):
    for ax, (lr, verdict) in zip(axes.ravel(), panels):
        costs = gradient_descent(lr)
        ax.semilogy(costs[costs < 1e12], color="#b3402a", lw=2.5)
        ax.set_title(verdict, loc="left")
        ax.text(0.98, 0.92, f"learning rate {lr:g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=19, color="#6e675c")
        ax.grid(color="gray", linestyle="dashed", alpha=0.5)
for ax in axes[1]:
    ax.set_xlabel("iteration")
for ax in axes[:, 0]:
    ax.set_ylabel("MSE cost")
fig.suptitle("Same neuron, same data, four step sizes — synthetic line fit",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.9)

out = (Path(__file__).resolve().parent.parent / "figs" / "mlgeo_4.0_perceptrons"
       / "learning_rate_grid_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
