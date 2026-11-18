"""Slide figures for lecture 21: schematics, and the parts of training a network
shown on a simple synthetic ocean temperature profile.

Synthetic profile (written here, no data file): temperature against depth,
0 to 600 m, with a mixed layer, a seasonal thermocline near 60 m and a main
thermocline near 300 m, sampled at 60 depths with 0.3 degC of noise:
    T(z) = 4 + 6 s(-(z - 60)/15) + 8 s(-(z - 300)/50),  s = sigmoid.
A second set of 40 samples, with new noise, is the validation set.
The network has 1 input (depth), 3 hidden neurons (tanh) and 1 output
(temperature): 10 weights. It is trained by plain gradient descent on the mean
squared error, written out in numpy below.

Figures, written to figs/lec21_neuron/ (numbers to numbers_training.json):
  activations.png          the three activations and the model each one gives
  correction_size.png      size of the error against the score, 1958 rule and log-loss
  boundary_passes.png      the 1958 rule on the separable windows of notebook 4.0
  network_diagram.png      a neuron beside a network with one hidden layer
  profile_line.png         the profile and the best line (one neuron)
  forward_pass.png         what each hidden neuron outputs, and their weighted sum
  loss_misfit.png          the loss drawn as error bars, sample by sample
  training_snapshots.png   the fit after 0, 5, 1,000 and 20,000 updates, and the loss curve
  learning_rate_profile.png  three learning rates on the same network
  accuracy_bars.png        four source types: no hidden layer, hidden layer, forest

Regenerate: pixi run python book/slides/2026/figscripts/lec21_network_training.py
(run lec21_one_neuron.py first: accuracy_bars reads its numbers.json)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # book/, for mlgeo_synth
import mlgeo_synth  # noqa: E402

plt.rcParams.update({
    "font.size": 22, "axes.titlesize": 25, "axes.titleweight": "bold",
    "axes.labelsize": 23, "xtick.labelsize": 19, "ytick.labelsize": 19,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, PURPLE, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#26241f", "#6e675c"
SYN = "Synthetic ocean temperature profile (written in the figure script), 60 samples, noise 0.3 °C"
FIGS = HERE.parent / "figs" / "lec21_neuron"
numbers = {}


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


# =============================================================================
# The three activations
# =============================================================================
z = np.linspace(-6, 6, 400)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
for ax, out, title, sub, color in [
        (axes[0], z, "No activation", "linear regression\n(lecture 13)", BLUE),
        (axes[1], sigmoid(z), "Sigmoid", "logistic regression\n(lecture 16)", GREEN),
        (axes[2], (z >= 0).astype(float), "Step", "threshold logic\nunit (1943)", RED)]:
    ax.plot(z, out, color=color, lw=5)
    ax.axvline(0, color=MUTED, lw=1.2, ls=":")
    ax.set_title(title, color=color)
    ax.set_xlabel("score $z$")
    ax.text(0.04, 0.95, sub, transform=ax.transAxes, va="top", fontsize=22, color=INK)
axes[0].set_ylabel("output of the neuron")
axes[1].set_ylim(-0.08, 1.45)
axes[2].set_ylim(-0.08, 1.45)
fig.tight_layout()
save(fig, "activations")

# =============================================================================
# Size of the correction against the score, for a true event (t = 1)
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 6.4))
ax.plot(z, 1 - (z >= 0).astype(float), color=RED, lw=5, label="1958 rule: prediction 0 or 1, all or nothing")
ax.plot(z, 1 - sigmoid(z), color=GREEN, lw=5, label="log-loss: prediction is a probability, in proportion to the error")
ax.axvline(0, color=MUTED, lw=1.2, ls=":")
ax.set(xlabel="score $z$ of a window that contains an event ($y = 1$)", ylabel="size of the error $|\\hat{y} - y|$", ylim=(-0.05, 1.3))
ax.annotate("missed with confidence:\nlarge error, large update", xy=(-4.5, 0.99), xytext=(-5.9, 0.42), fontsize=20, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=2))
ax.annotate("detected with confidence:\nalmost no update", xy=(4.5, 0.02), xytext=(2.0, 0.42), fontsize=20, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=2))
ax.legend(frameon=False, loc="upper right", fontsize=20)
save(fig, "correction_size")

# =============================================================================
# The 1958 rule on the separable windows of notebook 4.0 (same selection, same seed)
# =============================================================================
df = mlgeo_synth.detector_features(n=2000, event_fraction=0.5, seed=0)
score = df["sta_lta"] + df["spectral_centroid_hz"]
subset = df[((df["label"] == 0) & (score < 5)) | ((df["label"] == 1) & (score > 9))]
rng = np.random.default_rng(42)
noise_idx = rng.choice(subset.index[subset["label"] == 0], 300, replace=False)
event_idx = rng.choice(subset.index[subset["label"] == 1], 300, replace=False)
subset = subset.loc[np.concatenate([noise_idx, event_idx])]
Xs = subset[["sta_lta", "spectral_centroid_hz"]].to_numpy()
ts = subset["label"].to_numpy()
order = rng.permutation(len(Xs))
Xs, ts = Xs[order], ts[order]
start = np.random.default_rng(0)                      # random starting weights, seed 0, as in notebook 4.0
w, b, rate = start.normal(0, 1, 2), start.normal(0, 1), 0.1
mistakes, lines = [], []
for _ in range(50):
    n_wrong = 0
    for i in range(len(Xs)):
        error = ts[i] - float(Xs[i] @ w + b >= 0)
        if error != 0:
            n_wrong += 1
            w, b = w + rate * error * Xs[i], b + rate * error
            if len(mistakes) == 0 and n_wrong in (1, 5, 15):
                lines.append((n_wrong, w.copy(), b))
    mistakes.append(n_wrong)
    if n_wrong == 0:
        break
numbers["separable_rule"] = {"nonzero_errors_per_epoch": mistakes, "n_windows": len(ts)}
fig, ax = plt.subplots(figsize=(13, 7))
for lab, name, color in [(0, "noise windows", BLUE), (1, "event windows", ORANGE)]:
    ax.scatter(Xs[ts == lab, 0], Xs[ts == lab, 1], s=28, alpha=0.6, color=color, label=name)
x1 = np.linspace(0, 12, 50)
for k, wk, bk in lines:
    if abs(wk[1]) > 1e-12:
        ax.plot(x1, -(wk[0] * x1 + bk) / wk[1], color=MUTED, lw=2, ls="--")
ax.plot([], [], color=MUTED, lw=2, ls="--", label="after 1, 5 and 15 updates")
ax.plot(x1, -(w[0] * x1 + b) / w[1], color=INK, lw=4.5, label=f"final: every error is zero ({sum(mistakes)} updates)")
ax.set(xlabel="STA/LTA ratio", ylabel="Spectral centroid (Hz)", xlim=(0, 16), ylim=(-0.5, 15))
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=20)
fig.text(0.5, -0.02, "Synthetic (mlgeo_synth): 600 windows chosen so that a line separates the classes", ha="center", fontsize=18, color=MUTED)
save(fig, "boundary_passes")

# =============================================================================
# A neuron beside a network
# =============================================================================


def draw_layers(ax, sizes, x_positions, labels, colors, radius=0.33, top_labels=None):
    centers = []
    for size, xp in zip(sizes, x_positions):
        ys = np.linspace(-(size - 1) / 2, (size - 1) / 2, size) * 1.15
        centers.append([(xp, yy) for yy in ys])
    for a, c in zip(centers[:-1], centers[1:]):
        for (xa, ya) in a:
            for (xc, yc) in c:
                ax.plot([xa + radius, xc - radius], [ya, yc], color=MUTED, lw=1.6, zorder=1)
    for layer, color in zip(centers, colors):
        for (xp, yy) in layer:
            ax.add_patch(Circle((xp, yy), radius, fc="white" if color == INK else color, ec=color if color != INK else INK, lw=3, zorder=2))
    for layer, lab in zip(centers, labels):
        ax.text(layer[0][0], -3.4, lab, ha="center", va="top", fontsize=24, color=INK)
    if top_labels:
        for layer, lab in zip(centers, top_labels):
            ax.text(layer[0][0], max(yy for _, yy in layer) + 0.6, lab, ha="center", va="bottom", fontsize=21, color=MUTED)
    ax.set_aspect("equal")
    ax.axis("off")


fig, (left, right) = plt.subplots(1, 2, figsize=(17, 7.2), gridspec_kw={"width_ratios": [1, 1.5]})
draw_layers(left, [4, 1], [0, 2.6], ["inputs", "neuron"], [INK, "#cfe3f1"])
left.set(xlim=(-0.8, 3.8), ylim=(-4.4, 3.2))
left.set_title("One neuron", color=BLUE)
draw_layers(right, [4, 5, 2], [0, 2.8, 5.6], ["inputs", "hidden layer", "outputs"],
            [INK, "#cfe3f1", "#fbe3b3"])
right.set(xlim=(-0.8, 6.6), ylim=(-4.4, 3.2))
right.set_title("A network with one hidden layer", color=BLUE)
save(fig, "network_diagram")

# =============================================================================
# The synthetic temperature profile and the 1-3-1 network
# =============================================================================


def profile(depth):
    return 4 + 6 * sigmoid(-(depth - 60) / 15) + 8 * sigmoid(-(depth - 300) / 50)


rng = np.random.default_rng(3)
depth = np.sort(rng.uniform(0, 600, 60))
temp = profile(depth) + rng.normal(0, 0.3, 60)
depth_val = np.sort(rng.uniform(0, 600, 40))
temp_val = profile(depth_val) + rng.normal(0, 0.3, 40)
t_mean, t_std = temp.mean(), temp.std()
x, y = (depth - 300) / 300, (temp - t_mean) / t_std            # standardized input and target
x_val, y_val = (depth_val - 300) / 300, (temp_val - t_mean) / t_std
grid_depth = np.linspace(0, 600, 400)
grid_x = (grid_depth - 300) / 300


def hidden(xx, W1, b1):
    return np.tanh(np.outer(xx, W1) + b1)


def predict(xx, weights):
    W1, b1, W2, b2 = weights
    return hidden(xx, W1, b1) @ W2 + b2


def train(n_hidden, rate, n_steps, seed=0, keep=()):
    """Plain gradient descent on the mean squared error; the gradient by the chain rule."""
    r = np.random.default_rng(seed)
    W1, b1 = r.normal(0, 1.0, n_hidden), r.normal(0, 0.5, n_hidden)
    W2, b2 = r.normal(0, 0.5, n_hidden), r.normal(0, 0.5)     # every weight and bias random, none at zero
    loss, loss_val, kept = [], [], {}
    for step in range(n_steps + 1):
        if step in keep:
            kept[step] = (W1.copy(), b1.copy(), W2.copy(), b2)
        h = hidden(x, W1, b1)                       # predict
        error = h @ W2 + b2 - y
        loss.append(float(np.mean(error ** 2)))     # measure
        loss_val.append(float(np.mean((predict(x_val, (W1, b1, W2, b2)) - y_val) ** 2)))
        if not np.isfinite(loss[-1]) or loss[-1] > 1e6:
            break
        back = np.outer(error, W2) * (1 - h ** 2) * 2 / len(x)   # gradient, layer by layer
        W2 = W2 - rate * (h.T @ error) * 2 / len(x)              # step
        b2 = b2 - rate * 2 * error.mean()
        W1 = W1 - rate * (back * x[:, None]).sum(axis=0)
        b1 = b1 - rate * back.sum(axis=0)
    return np.array(loss), np.array(loss_val), (W1, b1, W2, b2), kept


RATE, STEPS, KEEP = 0.3, 20000, (0, 5, 1000, 20000)
NET_SEED = 2   # random starting weights; seed 2 trains without spikes at this learning rate (seeds 0-11 scanned)
loss, loss_val, final, kept = train(3, RATE, STEPS, seed=NET_SEED, keep=KEEP)
to_degc = lambda mse: float(np.sqrt(mse) * t_std)  # noqa: E731
slope, intercept = np.polyfit(depth, temp, 1)
line_rmse = float(np.sqrt(np.mean((slope * depth + intercept - temp) ** 2)))
numbers["profile"] = {
    "n_train": 60, "n_val": 40, "noise_degC": 0.3, "hidden_neurons": 3, "weights": 10, "rate": RATE, "steps": STEPS,
    "line_rmse_degC": line_rmse, "line_slope_degC_per_100m": float(slope * 100),
    "rmse_degC_at_steps": {str(k): to_degc(loss[k]) for k in KEEP},
    "final_train_rmse_degC": to_degc(loss[-1]), "final_val_rmse_degC": to_degc(loss_val[-1]),
}


def profile_axes(ax, title=None):
    ax.set(xlabel="Temperature (°C)", ylabel="Depth (m)", ylim=(600, 0), xlim=(2, 20.5))
    if title:
        ax.set_title(title)


# --- one neuron: the best line -------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 8))
ax.scatter(temp, depth, s=70, color=BLUE, alpha=0.8, label="measured (60 samples)")
ax.plot(slope * grid_depth + intercept, grid_depth, color=RED, lw=4.5, label=f"one neuron: a line, error {line_rmse:.1f} °C")
for d, tx, ha, lab in [(22, 2.4, "left", "mixed layer"), (70, 2.4, "left", "seasonal thermocline"), (300, 20.2, "right", "main thermocline")]:
    ax.text(tx, d, lab, ha=ha, va="center", fontsize=20, color=MUTED)
profile_axes(ax)
ax.legend(frameon=False, loc="lower right", fontsize=20)
save(fig, "profile_line")

# --- forward pass: hidden outputs and their weighted sum ---------------------
W1, b1, W2, b2 = final
H = hidden(grid_x, W1, b1)
fig, axes = plt.subplots(1, 3, figsize=(20, 7.6), gridspec_kw={"wspace": 0.42})
for j, color in enumerate([ORANGE, GREEN, PURPLE]):
    axes[0].plot(H[:, j], grid_depth, color=color, lw=4.5, label=f"hidden neuron {j + 1}")
    axes[1].plot(H[:, j] * W2[j] * t_std, grid_depth, color=color, lw=4.5)
axes[0].set(xlabel="output (between −1 and 1)", ylabel="Depth (m)", ylim=(600, 0), title="1. Each hidden neuron")
axes[0].legend(frameon=False, fontsize=18, loc="lower center")
axes[1].set(xlabel="contribution (°C)", ylim=(600, 0), title="2. Times its output weight")
axes[2].scatter(temp, depth, s=45, color=BLUE, alpha=0.6, label="measured")
axes[2].plot(predict(grid_x, final) * t_std + t_mean, grid_depth, color=INK, lw=4.5, label="network")
axes[2].set(xlabel="Temperature (°C)", ylim=(600, 0), xlim=(2, 20.5), title="3. Sum, plus a bias")
axes[2].legend(frameon=False, fontsize=18, loc="lower right")
for ax in axes[1:]:
    ax.set_yticklabels([])
save(fig, "forward_pass")

# --- the loss as misfit bars, early in training ---------------------------------
early = kept[1000]
pred_early = predict(x, early) * t_std + t_mean
fig, ax = plt.subplots(figsize=(9.5, 8))
for d, t_obs, t_pred in zip(depth, temp, pred_early):
    ax.plot([t_obs, t_pred], [d, d], color=RED, lw=2.2, alpha=0.8)
ax.plot([], [], color=RED, lw=2.2, label="error of each sample")
ax.scatter(temp, depth, s=70, color=BLUE, alpha=0.9, label="measured", zorder=3)
ax.plot(predict(grid_x, early) * t_std + t_mean, grid_depth, color=INK, lw=4.5, label="network after 1,000 updates")
profile_axes(ax)
ax.legend(frameon=False, loc="lower right", fontsize=20)
ax.text(2.4, 130, f"loss = mean of error²\nroot: {to_degc(loss[1000]):.2f} °C", fontsize=21, color=RED, va="top")
save(fig, "loss_misfit")

# --- snapshots of training and the loss curve ----------------------------------
fig, axes = plt.subplots(1, 5, figsize=(21, 7.2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.7], "wspace": 0.12})
for ax, step in zip(axes[:4], KEEP):
    ax.scatter(temp, depth, s=30, color=BLUE, alpha=0.6)
    ax.plot(predict(grid_x, kept[step]) * t_std + t_mean, grid_depth, color=INK, lw=4.5)
    ax.set(ylim=(600, 0), xlim=(2, 20.5), xlabel="°C")
    ax.set_title(f"update {step:,}\n{to_degc(loss[step]):.2f} °C", fontsize=22)
    if step:
        ax.set_yticklabels([])
axes[0].set_ylabel("Depth (m)")
ax = axes[4]
steps_axis = np.arange(1, len(loss) + 1)
ax.loglog(steps_axis, np.sqrt(loss) * t_std, color=INK, lw=4, label="training samples")
ax.loglog(steps_axis, np.sqrt(loss_val) * t_std, color=ORANGE, lw=4, ls="--", label="validation samples")
ax.axhline(0.3, color=MUTED, lw=2, ls=":")
ax.text(1.3, 0.31, "noise in the data, 0.3 °C", fontsize=18, color=MUTED, va="bottom")
ax.set(xlabel="update (one per epoch)", title="Root of the loss (°C)")
ax.set_yticks([0.3, 1, 3])
ax.set_yticklabels(["0.3", "1", "3"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.tick_right()
ax.legend(frameon=False, fontsize=19, loc="upper right")
save(fig, "training_snapshots")

# --- three learning rates -------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6.8))
rates = {}
for rate, color, lab in [(0.01, BLUE, "0.01: too small, still far after 20,000 updates"),
                         (0.3, GREEN, "0.3: reaches the noise level"),
                         (1.0, RED, "1.0: too large, the loss blows up")]:
    with np.errstate(all="ignore"):
        l_r, _, _, _ = train(3, rate, STEPS, seed=NET_SEED)
    good = np.isfinite(l_r) & (l_r < 1e6)
    ax.loglog(np.arange(1, len(l_r) + 1)[good], np.sqrt(l_r[good]) * t_std, color=color, lw=4.5, label=lab)
    rates[str(rate)] = {"steps_run": int(len(l_r)), "last_finite_rmse_degC": to_degc(l_r[good][-1])}
numbers["profile"]["learning_rates"] = rates
ax.axhline(0.3, color=MUTED, lw=2, ls=":")
ax.set(xlabel="update (one per epoch)", ylabel="Root of the training loss (°C)", ylim=(0.2, 300))
ax.legend(frameon=False, fontsize=20, loc="upper right", title="learning rate", title_fontsize=20)
ax.text(1.2e4, 0.31, "noise", fontsize=18, color=MUTED, va="bottom")
save(fig, "learning_rate_profile")

# =============================================================================
# Four source types: what the hidden layer adds (numbers from lec21_one_neuron.py)
# =============================================================================
four = json.loads((FIGS / "numbers.json").read_text())["four_classes_split_of_4.1"]
names = ["No hidden layer\n(logistic regression)\n248 weights", "One hidden layer\nof 32 neurons\n2,116 weights", "Random forest\n300 trees"]
values = [100 * four["no_hidden_layer_test_accuracy"], 100 * four["from_notebook_4.1_one_hidden_layer_32"]["test_accuracy"],
          100 * four["forest_300_trees_default_test_accuracy"]]
fig, ax = plt.subplots(figsize=(13, 7.4))
bars = ax.bar(names, values, color=[BLUE, GREEN, MUTED], width=0.6)
se = [100 * np.sqrt(v / 100 * (1 - v / 100) / four["n_test"]) for v in values]
ax.errorbar(names, values, yerr=se, fmt="none", ecolor=INK, lw=3, capsize=10)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, 80.6, f"{v:.1f}%", ha="center", fontsize=30, fontweight="bold", color="white")
ax.set(ylabel="Accuracy, 800 test events (%)", ylim=(80, 92))
fig.text(0.5, -0.2, "Four source types, 61 waveform features. Pacific Northwest, Zenodo record 14025693 (real data).\nBars: one standard error.",
         ha="center", fontsize=18, color=MUTED)
save(fig, "accuracy_bars")

(FIGS / "numbers_training.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
