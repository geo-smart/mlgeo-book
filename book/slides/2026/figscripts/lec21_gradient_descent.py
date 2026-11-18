"""Slide figures for lecture 21, Part 2: gradient descent on one neuron, one step at a time.

Same data, same seed and same loop as notebook 4.0, section 2, so that the
numbers on the slides are the numbers the students print.

Synthetic data (written here, no data file): one earthquake, P-wave travel
times picked at 100 stations between 5 and 150 km,
    t = 1.0 + d / 6.0 + noise (0.5 s),
80 samples to train and 20 to test. The input is the distance, standardized on
the training samples. The model is one neuron with no activation,
yhat = w x + b, trained on the mean squared error. Notation of notebook 3.6:
observation y, prediction yhat, error = yhat - y, loss L, learning rate alpha.

Figures, written to figs/lec21_neuron/ (numbers to numbers_gradient_descent.json):
  gd_start_seeds.png       the data and the lines three seeds start from
  gd_errors.png            errors: prediction minus observation, sample by sample
  gd_loss_per_sample.png   the loss of each sample of a batch of 16, and their mean
  gd_gradient.png          the gradient of each sample and their mean; the slope of the loss
  gd_one_update.png        one update of w and of b, each against its own gradient
  gd_update.png            the path of (w, b) on the map of the loss
  gd_loss_epochs.png       predictions at four epochs and the loss at each epoch
  gd_batches.png           the loss of each batch of 16 samples, epoch by epoch
  gd_learning_rate.png     four learning rates

Regenerate: pixi run python book/slides/2026/figscripts/lec21_gradient_descent.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 22, "axes.titlesize": 25, "axes.titleweight": "bold",
    "axes.labelsize": 23, "xtick.labelsize": 19, "ytick.labelsize": 19,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, PURPLE, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#26241f", "#6e675c"
SYN = "Synthetic: P travel times of one earthquake at 80 stations, 6 km/s, noise 0.5 s (notebook 4.0, section 2)"
FIGS = HERE.parent / "figs" / "lec21_neuron"
numbers = {}


def save(fig, name, tag=True):
    if tag:
        fig.text(0.5, -0.03, SYN, ha="center", fontsize=17, color=MUTED)
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- the data of notebook 4.0, section 2 ----------------------------------------
rng = np.random.default_rng(1)
distance = rng.uniform(5, 150, 100)
travel_time = 1.0 + distance / 6.0 + rng.normal(0, 0.5, 100)
train_d, y = distance[:80], travel_time[:80]
d_mean, d_std = train_d.mean(), train_d.std()
x = (train_d - d_mean) / d_std
x_line = np.linspace(x.min(), x.max(), 50)
d_line = x_line * d_std + d_mean
SEED, RATE, EPOCHS = 0, 0.1, 60


def start(seed):
    r = np.random.default_rng(seed)
    return r.normal(0, 1), r.normal(0, 1)


def mean_loss(w, b):
    return float(np.mean((w * x + b - y) ** 2))


# --- 1. random starting weights ---------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7))
ax.scatter(train_d, y, s=60, color=INK, alpha=0.75, label="observations (80 training samples)")
seeds = {}
for s, color in zip((0, 1, 2), (BLUE, ORANGE, GREEN)):
    ws, bs = start(s)
    seeds[str(s)] = {"w": ws, "b": bs, "loss": mean_loss(ws, bs)}
    ax.plot(d_line, ws * x_line + bs, color=color, lw=4.5, label=f"seed {s}: w = {ws:.2f}, b = {bs:.2f}")
ax.set(xlabel="Distance to the epicenter (km)", ylabel="P travel time (s)")
ax.legend(frameon=False, loc="upper left", fontsize=20)
numbers["starting_weights"] = seeds
save(fig, "gd_start_seeds")

# --- 2. errors --------------------------------------------------------------------
w, b = start(SEED)
yhat = w * x + b
error = yhat - y
loss_per_sample = error ** 2
loss = loss_per_sample.mean()
fig, ax = plt.subplots(figsize=(13, 7))
ax.vlines(train_d, y, yhat, color=RED, lw=2.2, alpha=0.85, label="error = prediction − observation")
ax.scatter(train_d, y, s=60, color=INK, zorder=3, label="observations $y_i$")
ax.plot(d_line, w * x_line + b, color=BLUE, lw=4.5, label="predictions $\\hat{y}_i$ (starting weights, seed 0)")
i_far = int(np.argmax(train_d))
ax.annotate(f"error of this sample: {error[i_far]:.1f} s", xy=(train_d[i_far], 6), xytext=(62, 5.2),
            fontsize=21, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=2.5),
            bbox=dict(fc="white", ec=RED, lw=1.5, boxstyle="round,pad=0.3"))
ax.set(xlabel="Distance to the epicenter (km)", ylabel="P travel time (s)")
ax.legend(frameon=False, loc="upper left", fontsize=20)
numbers["first_epoch"] = {"w": w, "b": b, "loss_s2": float(loss), "loss_per_sample_min": float(loss_per_sample.min()),
                          "loss_per_sample_max": float(loss_per_sample.max()), "error_farthest_station_s": float(error[i_far])}
save(fig, "gd_errors")

# --- 3. the loss of each sample of a batch, and the mean --------------------------
show = np.arange(16)[np.argsort(train_d[:16])]
labels = [f"{d:.0f}" for d in train_d[show]]
batch_mean = float(loss_per_sample[show].mean())
fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(range(16), loss_per_sample[show], color=RED)
ax.axhline(batch_mean, color=INK, lw=3, ls="--")
ax.text(-0.4, 560, f"dashed line: the mean over the 16 samples,\nthe loss of the batch, {batch_mean:.0f} s²", fontsize=23, color=INK, va="top")
ax.set_xticks(range(16))
ax.set_xticklabels(labels)
ax.set(xlabel="Distance of the station (km): a batch of 16 samples", ylabel="Loss of the sample: error² (s²)")
numbers["first_epoch"]["batch_of_16_mean_loss_s2"] = batch_mean
numbers["first_epoch"]["batch_of_16_min_max_s2"] = [float(loss_per_sample[show].min()), float(loss_per_sample[show].max())]
save(fig, "gd_loss_per_sample")

# --- 4. the gradient ---------------------------------------------------------------
grad_w_per_sample = 2 * error * x
grad_w, grad_b = float(grad_w_per_sample.mean()), float(np.mean(2 * error))
fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"wspace": 0.25})
axes[0].bar(range(16), grad_w_per_sample[show], color=BLUE)
axes[0].axhline(grad_w, color=INK, lw=3, ls="--")
axes[0].text(-0.4, -42, f"dashed line: the mean\nover the 80 samples,\n{grad_w:.1f}", fontsize=21, color=INK, va="top")
axes[0].set_xticks(range(16))
axes[0].set_xticklabels(labels, fontsize=14)
axes[0].set(xlabel="Distance of the station (km)", ylabel="2 × error × input", title="Gradient of each sample, for $w$")
w_grid = np.linspace(-4, 18, 200)
axes[1].plot(w_grid, [mean_loss(wg, b) for wg in w_grid], color=INK, lw=4)
tangent = np.array([w - 3.5, w + 3.5])
axes[1].plot(tangent, loss + grad_w * (tangent - w), color=RED, lw=5)
axes[1].scatter([w], [loss], s=160, color=RED, zorder=3)
axes[1].text(-4.6, 222, f"slope =\ngradient\n{grad_w:.1f}", fontsize=21, color=RED, va="bottom")
axes[1].annotate("", xy=(w + 5.5, loss + 14), xytext=(w + 0.6, loss + 14), arrowprops=dict(arrowstyle="->", color=GREEN, lw=4))
axes[1].text(w + 0.8, loss + 20, "the update goes\nagainst the gradient", fontsize=21, color=GREEN)
axes[1].set(xlabel="Weight $w$ (bias fixed)", ylabel="Loss (s²)", title="The gradient is the slope of the loss")
numbers["first_epoch"].update({"grad_w": grad_w, "grad_b": grad_b})
save(fig, "gd_gradient")

# --- 4b. one update, in numbers (notebook 4.0, section 2.5) --------------------------
w0, b0 = start(SEED)
w1_, b1_ = w0 - RATE * grad_w, b0 - RATE * grad_b
fig, axes = plt.subplots(1, 2, figsize=(19, 7.2), gridspec_kw={"wspace": 0.22})
for ax, name, now, new, grad, loss_of in [
        (axes[0], "weight $w$", w0, w1_, grad_w, lambda v: mean_loss(v, b0)),
        (axes[1], "bias $b$", b0, b1_, grad_b, lambda v: mean_loss(w0, v))]:
    grid = np.linspace(now - 2.6, new + 1.6, 200)
    ax.plot(grid, [loss_of(v) for v in grid], color=INK, lw=4)
    tangent = np.array([now - 1.0, now + 1.0])
    ax.plot(tangent, loss_of(now) + grad * (tangent - now), color=RED, lw=4, ls="--")
    ax.scatter([now], [loss_of(now)], s=260, color=RED, zorder=3)
    ax.scatter([new], [loss_of(new)], s=260, color=GREEN, zorder=3)
    ax.annotate("", xy=(new, loss_of(now)), xytext=(now, loss_of(now)), arrowprops=dict(arrowstyle="->", color=GREEN, lw=4))
    ax.text(0.97, 0.97, f"dashed: slope at 'now'\n= gradient = {grad:.2f}", transform=ax.transAxes, ha="right", va="top",
            color=RED, fontsize=21)
    ax.annotate(f"now: {now:.2f}", xy=(now, loss_of(now)), xytext=(-18, -18), textcoords="offset points",
                ha="right", va="top", color=RED, fontsize=21, fontweight="bold")
    ax.annotate(f"after: {new:.2f}", xy=(new, loss_of(new)), xytext=(-18, -18), textcoords="offset points",
                ha="right", va="top", color=GREEN, fontsize=21, fontweight="bold")
    ax.annotate(f"change = −{RATE} × ({grad:.2f}) = {-RATE * grad:+.2f}", xy=((now + new) / 2, loss_of(now)),
                xytext=(0, 14), textcoords="offset points", ha="center", va="bottom", color=GREEN, fontsize=21)
    ax.set(xlabel=name, ylabel="Loss (s²)", title=f"One update of the {name.split()[0]}")
    ax.set_ylim(None, loss_of(now) + 0.45 * (loss_of(now) - loss_of(new)) + 25)
numbers["one_update"] = {"w": [w0, w1_], "b": [b0, b1_], "change_w": -RATE * grad_w, "change_b": -RATE * grad_b,
                         "loss_after_both": mean_loss(w1_, b1_)}
save(fig, "gd_one_update")

# --- the full loop (notebook 4.0, section 2.6) --------------------------------------
w, b = start(SEED)
hist = {"loss": [], "w": [], "b": [], "grad_w": [], "grad_b": []}
for epoch in range(EPOCHS):
    error = w * x + b - y
    gw, gb = np.mean(2 * error * x), np.mean(2 * error)
    for key, value in zip(hist, (np.mean(error ** 2), w, b, gw, gb)):
        hist[key].append(float(value))
    w, b = w - RATE * gw, b - RATE * gb
numbers["training"] = {
    "rate": RATE, "epochs": EPOCHS, "seed": SEED,
    "loss_at_epoch": {str(k): hist["loss"][k] for k in (0, 1, 2, 5, 10, 20, 40, 59)},
    "after_one_update": {"w": hist["w"][1], "b": hist["b"][1], "loss": hist["loss"][1]},
    "final_w": hist["w"][-1], "final_b": hist["b"][-1], "final_grad_w": hist["grad_w"][-1],
    "velocity_km_s": float(d_std / w), "delay_s": float(b - w * d_mean / d_std),
}

# --- 5. the path of the weights on the map of the loss -------------------------------
W, B = np.meshgrid(np.linspace(-1.5, 9.5, 160), np.linspace(-2, 19, 160))
L = np.mean((W[..., None] * x + B[..., None] - y) ** 2, axis=-1)
fig, ax = plt.subplots(figsize=(12, 7.6))
cs = ax.contourf(W, B, L, levels=np.geomspace(0.2, 400, 14), cmap="YlGnBu_r", norm=matplotlib.colors.LogNorm())
cbar = fig.colorbar(cs, ax=ax, ticks=[0.3, 1, 3, 10, 30, 100, 300], format="%g")
cbar.set_label("Loss (s²)")
ax.plot(hist["w"], hist["b"], "-o", color=RED, lw=2.5, ms=7)
ax.scatter([hist["w"][0]], [hist["b"][0]], s=260, color=RED, zorder=3)
ax.text(hist["w"][0] + 0.25, hist["b"][0] - 0.3, "start: random weights (seed 0)", fontsize=21, color=RED, va="top")
ax.annotate("one update per epoch:\nlarger where the\ngradient is larger", xy=(hist["w"][2], hist["b"][2]), xytext=(4.3, 1.2),
            fontsize=20, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=2))
ax.text(hist["w"][-1] - 0.25, hist["b"][-1] + 0.9, "gradient zero:\nthe updates stop", fontsize=20, color=INK, ha="right")
ax.set(xlabel="Weight $w$", ylabel="Bias $b$")
save(fig, "gd_update")

# --- 6. predictions at four epochs, and the loss at each epoch -----------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"wspace": 0.25})
axes[0].scatter(train_d, y, s=45, color=INK, alpha=0.7)
for k, color in zip((0, 2, 5, 59), (BLUE, ORANGE, GREEN, RED)):
    axes[0].plot(d_line, hist["w"][k] * x_line + hist["b"][k], color=color, lw=4.5, label=f"epoch {k}: loss {hist['loss'][k]:.3g} s²")
    axes[1].scatter([k], [hist["loss"][k]], s=220, color=color, zorder=3)
axes[0].set(xlabel="Distance to the epicenter (km)", ylabel="P travel time (s)", title="Predictions")
axes[0].legend(frameon=False, fontsize=17, loc="upper left")
axes[1].semilogy(range(EPOCHS), hist["loss"], color=INK, lw=4)
axes[1].axhline(0.25, color=MUTED, lw=2, ls=":")
axes[1].text(EPOCHS - 1, 0.29, "variance of the noise, 0.25 s²", fontsize=18, color=MUTED, ha="right", va="bottom")
axes[1].set(xlabel="Epoch", ylabel="Loss (s²)", title="Loss at each epoch")
save(fig, "gd_loss_epochs")

# --- 7. batches of 16 samples (notebook 4.0, section 2.7) ----------------------------
rng = np.random.default_rng(SEED)
w, b = rng.normal(0, 1), rng.normal(0, 1)
batch_loss = []
for epoch in range(20):
    shuffled = rng.permutation(len(x))
    this_epoch = []
    for first in range(0, len(x), 16):
        in_batch = shuffled[first:first + 16]
        error = w * x[in_batch] + b - y[in_batch]
        this_epoch.append(float(np.mean(error ** 2)))
        w, b = w - RATE * np.mean(2 * error * x[in_batch]), b - RATE * np.mean(2 * error)
    batch_loss.append(this_epoch)
fig, ax = plt.subplots(figsize=(14, 7))
for k, values in enumerate(batch_loss):
    ax.semilogy(k + np.linspace(0, 0.8, 5), values, "o", color=BLUE, ms=11, label="loss of one batch (16 samples)" if k == 0 else None)
ax.semilogy(np.arange(20) + 0.4, [np.mean(v) for v in batch_loss], color=RED, lw=4, label="mean over the 5 batches of the epoch")
ax.set(xlabel="Epoch (5 batches, so 5 updates, in each)", ylabel="Loss (s²)")
ax.set_xticks(range(0, 21, 5))
ax.legend(frameon=False, fontsize=21, loc="upper right")
numbers["batches"] = {"batch_size": 16, "batches_per_epoch": 5, "loss_epoch_0": batch_loss[0], "loss_epoch_1": batch_loss[1],
                      "loss_epoch_19": batch_loss[-1]}
save(fig, "gd_batches")

# --- 8. learning rates ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
rates = {}
for rate, color, lab in [(0.001, BLUE, "0.001: too small"), (0.01, ORANGE, "0.01: slow"),
                         (0.1, GREEN, "0.1: reaches the lowest loss in 30 epochs"), (1.05, RED, "1.05: too large, the loss diverges")]:
    w, b = start(SEED)
    losses = []
    for epoch in range(200):
        error = w * x + b - y
        losses.append(float(np.mean(error ** 2)))
        if losses[-1] > 1e5:
            break
        w, b = w - rate * np.mean(2 * error * x), b - rate * np.mean(2 * error)
    ax.semilogy(losses, color=color, lw=4.5, label=lab)
    rates[str(rate)] = {"epochs_run": len(losses), "last_loss_s2": losses[-1]}
ax.set(xlabel="Epoch", ylabel="Loss (s²)", ylim=(0.1, 3e5))
ax.legend(frameon=False, fontsize=20, loc="upper right", title="learning rate", title_fontsize=20)
numbers["learning_rates"] = rates
save(fig, "gd_learning_rate")

(FIGS / "numbers_gradient_descent.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
