"""Slide figures and numbers for lecture 21, Part 1: one neuron.

Four figures, written to figs/lec21_neuron/, and every number quoted on a
slide, written to figs/lec21_neuron/numbers.json.

1. neuron_diagram.png      one neuron drawn as inputs, weights, sum, activation.
2. template_weights.png    a neuron trained on waveform windows of a repeating
                           earthquake: its weights against the noise-free
                           waveform (the template). Synthetic (mlgeo_synth).
3. shift_blindness.png     the same neuron when the arrival moves inside the
                           window, against a template slid over all lags.
4. rule_vs_loss.png        explosion against earthquake, Pacific Northwest, the
                           lecture 16 split (random_state=2026, 25% test): the
                           1958 perceptron rule against gradient descent on the
                           log-loss.

It also scores, on the split of notebook 4.1 (60/20/20, random_state=42), the
four-class model with no hidden layer and a default random forest, to set
beside the 87.1% of the 4.1 network.

Regenerate: pixi run python book/slides/2026/figscripts/lec21_one_neuron.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # book/, for mlgeo_synth
from mlgeo_synth.seismic import synthetic_seismogram  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c"
PNW_TAG = "Pacific Northwest seismic events, Zenodo record 14025693 (real data)"
SYN_TAG = "Synthetic (mlgeo_synth): one source at 20 km, new noise in every window"
SEISMIC_FILES = {
    "1000_earthquakes_physical_features.csv": "md5:28129c8dd1b3e14f655d489577b841b5",
    "1000_explosion_physical_features.csv": "md5:af1342d32e163e961e043364136359b0",
    "1000_noise_physical_features.csv": "md5:16cdb992fed6cf6273d5624f5df905da",
    "1000_surface_physical_features.csv": "md5:9a2c2643030cf058704d68e130654e9d",
}
FIGS = HERE.parent / "figs" / "lec21_neuron"
FIGS.mkdir(parents=True, exist_ok=True)
numbers = {}


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def perceptron_rule(X, y, n_pass, seed=None, X_test=None, y_test=None):
    """Rosenblatt's rule as in notebook 4.0: update the weights after each nonzero error.
    Random starting weights (seed 0), not zeros."""
    start = np.random.default_rng(0)
    w, b = start.normal(0, 1, X.shape[1]), start.normal(0, 1)
    rng = np.random.default_rng(seed) if seed is not None else None
    mistakes, test_acc = [], []
    for _ in range(n_pass):
        n_wrong = 0
        order = rng.permutation(len(X)) if rng is not None else range(len(X))
        for i in order:
            error = y[i] - float(X[i] @ w + b >= 0)
            if error != 0:
                n_wrong += 1
                w = w + error * X[i]
                b = b + error
        mistakes.append(n_wrong)
        if X_test is not None:
            test_acc.append(float(((X_test @ w + b >= 0) == y_test).mean()))
        if n_wrong == 0:
            break
    return w, b, mistakes, test_acc


def logistic_descent(X, y, n_pass, rate, X_test=None, y_test=None):
    """Gradient descent on the mean log-loss: the loop of notebook 3.6, in numpy.
    Random starting weights (seed 0), not zeros."""
    start = np.random.default_rng(0)
    w, b = start.normal(0, 1, X.shape[1]), start.normal(0, 1)
    loss, test_acc = [], []
    for _ in range(n_pass):
        p = sigmoid(X @ w + b)
        loss.append(float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))))
        w = w + rate * X.T @ (y - p) / len(y)
        b = b + rate * np.mean(y - p)
        if X_test is not None:
            test_acc.append(float(((sigmoid(X_test @ w + b) >= 0.5) == y_test).mean()))
    return w, b, loss, test_acc


# =============================================================================
# 1. One neuron, drawn
# =============================================================================
fig, ax = plt.subplots(figsize=(15, 6.2))
ax.set(xlim=(0, 15), ylim=(0, 6.2))
ax.axis("off")
ys = [5.2, 3.9, 2.6, 1.0]
labels = ["$x_1$", "$x_2$", "$x_3$", "$x_n$"]
weights = ["$w_1$", "$w_2$", "$w_3$", "$w_n$"]
for yy, lab, wt in zip(ys, labels, weights):
    ax.add_patch(Circle((1.2, yy), 0.45, fc="white", ec=INK, lw=2.5))
    ax.text(1.2, yy, lab, ha="center", va="center", fontsize=24)
    ax.add_patch(FancyArrowPatch((1.7, yy), (5.5, 3.1 + (yy - 3.1) * 0.12), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=MUTED))
    ax.text(3.3, yy + (3.1 - yy) * 0.42 + 0.3, wt, ha="center", fontsize=22, color=BLUE)
ax.text(1.2, 1.85, "⋮", ha="center", va="center", fontsize=26)
ax.add_patch(Circle((6.3, 3.1), 0.8, fc="#eaf3fa", ec=BLUE, lw=3))
ax.text(6.3, 3.1, "Σ", ha="center", va="center", fontsize=40, color=BLUE)
ax.add_patch(FancyArrowPatch((6.3, 0.9), (6.3, 2.25), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=MUTED))
ax.text(6.3, 0.45, "bias $b$", ha="center", fontsize=22, color=BLUE)
ax.add_patch(FancyArrowPatch((7.15, 3.1), (9.2, 3.1), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=MUTED))
ax.text(8.15, 3.4, "score $z$", ha="center", fontsize=22)
ax.add_patch(plt.Rectangle((9.3, 2.2), 2.2, 1.8, fc="#fdf3e0", ec=ORANGE, lw=3))
zz = np.linspace(-1, 1, 100)
ax.plot(10.4 + zz * 0.85, 3.1 + (sigmoid(6 * zz) - 0.5) * 1.2, color=ORANGE, lw=4)
ax.text(10.4, 4.25, "activation", ha="center", fontsize=22, color=ORANGE)
ax.add_patch(FancyArrowPatch((11.55, 3.1), (13.2, 3.1), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=MUTED))
ax.text(14.0, 3.1, "output", ha="center", va="center", fontsize=24)
ax.text(1.2, 5.95, "inputs", ha="center", fontsize=22, color=MUTED)
ax.text(3.3, 5.95, "weights", ha="center", fontsize=22, color=BLUE)
ax.text(6.3, 4.25, "weighted sum", ha="center", fontsize=22, color=BLUE)
save(fig, "neuron_diagram")
import shutil  # noqa: E402
shutil.copy(FIGS / "neuron_diagram.png", HERE.parents[2] / "img" / "neuron_diagram.png")   # also used in notebook 4.1

# =============================================================================
# 2 and 3. A neuron on waveform windows: weights against the template
# =============================================================================
FS = 100.0
I0, I1 = int(7.5 * FS), int(13.5 * FS)  # a 6 s window; P at 8.33 s, S at 10.71 s


def window(event, snr, seed, shift=0):
    """One standardized 6 s window; shift > 0 moves the arrival later in the window."""
    if event:
        _, trace, _ = synthetic_seismogram(magnitude=2.0, distance_km=20.0, snr=snr, seed=seed)
    else:
        _, trace, _ = synthetic_seismogram(magnitude=-10, distance_km=20.0, snr=1e-12, seed=seed)
    w = trace[I0 - shift:I1 - shift]
    return (w - w.mean()) / w.std()


def windows(n, seed, shift=0, snr_range=(1.5, 6.0)):
    rng = np.random.default_rng(seed)
    X, y, snrs = [], [], []
    for k in range(n):
        event = k % 2
        snr = float(np.exp(rng.uniform(np.log(snr_range[0]), np.log(snr_range[1]))))
        X.append(window(event, snr, int(rng.integers(2**31)), shift))
        y.append(event)
        snrs.append(snr if event else np.nan)
    return np.array(X), np.array(y), np.array(snrs)


_, clean, meta = synthetic_seismogram(magnitude=2.0, distance_km=20.0, snr=1e9, seed=0)
template = clean[I0:I1]
template = (template - template.mean()) / template.std()
t_win = np.arange(I1 - I0) / FS

Xw_train, yw_train, snr_train = windows(800, seed=10)
Xw_test, yw_test, snr_test = windows(400, seed=20)

# Logistic regression with a penalty on the size of the weights (C = 0.001). The noise
# of the generator is colored (power falls as 1/f); with a weak penalty (C = 1) the
# weights also work at cancelling the low-frequency noise and look less like the event.
PENALTY_C = 0.001
neuron = LogisticRegression(C=PENALTY_C, max_iter=5000).fit(Xw_train, yw_train)
w_log, b_log = neuron.coef_[0], float(neuron.intercept_[0])
weak = LogisticRegression(C=1.0, max_iter=5000).fit(Xw_train, yw_train)
w_per, b_per, mistakes_w, _ = perceptron_rule(Xw_train, yw_train, n_pass=100)

# template matching at zero lag: correlation with the template, threshold from noise windows
cc_train = Xw_train @ template / len(template)
cc_test = Xw_test @ template / len(template)
cc_threshold = float(np.quantile(cc_train[yw_train == 0], 0.99))


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


numbers["waveform"] = {
    "window_s": (I1 - I0) / FS, "n_weights": int(I1 - I0), "n_train": len(yw_train), "n_test": len(yw_test),
    "snr_range": [1.5, 6.0], "t_p_s": meta["t_p"], "t_s_s": meta["t_s"],
    "logistic_penalty_C": PENALTY_C,
    "weak_penalty_C1_weights_corr_template": corr(weak.coef_[0], template),
    "weak_penalty_C1_test_accuracy": float(weak.score(Xw_test, yw_test)),
    "logistic_test_accuracy": float(((sigmoid(Xw_test @ w_log + b_log) >= 0.5) == yw_test).mean()),
    "logistic_weights_corr_template": corr(w_log, template),
    "perceptron_epochs_to_zero_errors": len(mistakes_w), "perceptron_nonzero_errors_per_epoch": mistakes_w,
    "perceptron_test_accuracy": float(((Xw_test @ w_per + b_per >= 0) == yw_test).mean()),
    "perceptron_weights_corr_template": corr(w_per, template),
    "template_threshold_cc": cc_threshold,
    "template_test_accuracy": float(((cc_test > cc_threshold) == yw_test).mean()),
    "template_test_recall": float((cc_test[yw_test == 1] > cc_threshold).mean()),
    "template_test_false_alarm": float((cc_test[yw_test == 0] > cc_threshold).mean()),
}

fig = plt.figure(figsize=(21, 7.6))
grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], hspace=0.5, wspace=0.12)
low = int(np.nanargmin(np.where(yw_test == 1, snr_test, np.nan)))
noise = int(np.flatnonzero(yw_test == 0)[0])
for row, (idx, title, color) in enumerate([
        (low, "Given: an event window", ORANGE),
        (noise, "Given: a noise window", BLUE)]):
    ax = fig.add_subplot(grid[row, 0])
    ax.plot(t_win, Xw_test[idx], color=color, lw=1.4)
    ax.set_title(title, fontsize=26, loc="left")
    ax.set_yticks([])
    if row == 1:
        ax.set_xlabel("Time in the window (s)")
    else:
        ax.set_xticklabels([])
for row, (vec, title, color) in enumerate([
        (template, "Not given: the template", INK),
        (w_log, f"Found by training: the 600 weights", GREEN)]):
    ax = fig.add_subplot(grid[row, 1])
    ax.plot(t_win, vec / np.abs(vec).max(), color=color, lw=2.2)
    ax.set_title(title, fontsize=26, loc="left")
    ax.set_yticks([])
    if row == 1:
        ax.set_xlabel("Time in the window (s): one weight per sample")
    else:
        ax.set_xticklabels([])
fig.text(0.5, -0.01, SYN_TAG + f". {len(yw_train)} training windows, {I1 - I0} weights.", ha="center", fontsize=17, color=MUTED)
save(fig, "template_weights")

# --- the same neuron when the arrival moves inside the window ----------------
shifts = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 33, 36, 40, 45, 50]
max_lag = 60
recall_neuron, recall_slid, false_alarm_slid = [], [], []


def slid_correlation(X):
    """Largest normalized correlation with the template over lags of +-max_lag samples."""
    n = len(template)
    best = np.full(len(X), -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = X[:, lag:], template[:n - lag]
        else:
            a, b = X[:, :n + lag], template[-lag:]
        a = (a - a.mean(axis=1, keepdims=True)) / a.std(axis=1, keepdims=True)
        b = (b - b.mean()) / b.std()
        best = np.maximum(best, a @ b / len(b))
    return best


slid_threshold = float(np.quantile(slid_correlation(Xw_train[yw_train == 0]), 0.99))
for s in shifts:
    Xs, ys_, _ = windows(400, seed=100 + s, shift=s)
    recall_neuron.append(float((sigmoid(Xs[ys_ == 1] @ w_log + b_log) >= 0.5).mean()))
    slid = slid_correlation(Xs)
    recall_slid.append(float((slid[ys_ == 1] > slid_threshold).mean()))
    false_alarm_slid.append(float((slid[ys_ == 0] > slid_threshold).mean()))
numbers["shift"] = {"shift_s": [s / FS for s in shifts], "neuron_recall": recall_neuron,
                    "slid_template_recall": recall_slid, "slid_template_false_alarm": false_alarm_slid,
                    "slid_threshold_cc": slid_threshold, "max_lag_s": max_lag / FS}

fig, ax = plt.subplots(figsize=(13, 6.6))
ax.plot([s / FS for s in shifts], recall_slid, "o-", color=INK, lw=3, ms=10, label="Template slid over every lag (cross-correlation)")
ax.plot([s / FS for s in shifts], recall_neuron, "o-", color=GREEN, lw=3, ms=10, label="One neuron, weights fixed to window positions")
ax.set(xlabel="Arrival later than in training (s)", ylabel="Fraction of events detected", ylim=(-0.03, 1.08))
s_period = 1.0 / (0.6 * 5.0)  # the S wavelet of the generator: 0.6 x 5 Hz = 3 Hz
ax.annotate("one S-wave period,\n0.33 s: peaks line up again", xy=(s_period, 0.86), xytext=(0.385, 0.42),
            fontsize=18, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=2))
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
numbers["shift"]["s_wave_period_s"] = s_period
fig.text(0.5, -0.02, SYN_TAG + ". 200 event windows per point.", ha="center", fontsize=17, color=MUTED)
save(fig, "shift_blindness")

# =============================================================================
# 4. Explosion against earthquake: the 1958 rule against the log-loss
# =============================================================================
frames = []
for fname, checksum in SEISMIC_FILES.items():
    path = pooch.retrieve(url=f"https://zenodo.org/api/records/14025693/files/{fname}/content",
                          known_hash=checksum, fname=fname, path=pooch.os_cache("mlgeo"))
    frames.append(pd.read_csv(path, index_col=0))
seismic = pd.concat(frames, ignore_index=True).dropna(axis=1)
X = seismic.drop(columns=["source", "serial_no"])
y = seismic["source"]

two = y.isin(["earthquake", "explosion"])
Xb, yb = X[two].to_numpy(float), (y[two] == "explosion").astype(int).to_numpy()
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, random_state=2026, stratify=yb)
scaler = StandardScaler().fit(Xtr)
Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

N_PASS = 300
_, _, mistakes, acc_rule = perceptron_rule(Xtr, ytr, n_pass=N_PASS, seed=0, X_test=Xte, y_test=yte)
_, _, loss, acc_loss = logistic_descent(Xtr, ytr, n_pass=N_PASS, rate=0.5, X_test=Xte, y_test=yte)
reference = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
last = slice(N_PASS - 100, N_PASS)
numbers["explosion_earthquake"] = {
    "n_train": len(ytr), "n_test": len(yte), "n_features": Xtr.shape[1], "passes": N_PASS,
    "rule_nonzero_errors_first_epoch": mistakes[0], "rule_nonzero_errors_last_100_min": int(np.min(mistakes[last])),
    "rule_nonzero_errors_last_100_mean": float(np.mean(mistakes[last])),
    "rule_test_accuracy_last_100_min": float(np.min(acc_rule[last])),
    "rule_test_accuracy_last_100_max": float(np.max(acc_rule[last])),
    "loss_first": loss[0], "loss_last": loss[-1], "descent_rate": 0.5,
    "descent_test_accuracy_last": acc_loss[-1],
    "descent_test_accuracy_last_100_min": float(np.min(acc_loss[last])),
    "descent_test_accuracy_last_100_max": float(np.max(acc_loss[last])),
    "lecture16_logistic_test_accuracy": float(reference.score(Xte, yte)),
}

passes = np.arange(1, N_PASS + 1)
fig, (left, right) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"wspace": 0.28})
left.plot(passes, mistakes, color=RED, lw=2.5)
left.set(xlabel="Epoch (1,500 training events)", ylabel="Events with a nonzero error",
         title="1958 rule: nonzero errors per epoch", ylim=(0, max(mistakes) * 1.08))
right.plot(passes, acc_rule, color=RED, lw=2, label="1958 rule")
right.plot(passes, acc_loss, color=GREEN, lw=3.5, label="Gradient descent on the log-loss")
right.set(xlabel="Epoch (1,500 training events)", ylabel="Accuracy, 500 test events",
          title="Accuracy on the test events", ylim=(0.74, 0.97))
right.legend(frameon=False, loc="upper left", fontsize=18)
fig.text(0.5, -0.02, "Explosion against earthquake, 61 waveform features. " + PNW_TAG + ".", ha="center", fontsize=17, color=MUTED)
save(fig, "rule_vs_loss")

# =============================================================================
# Four source types on the split of notebook 4.1: no hidden layer, and a forest
# =============================================================================
frames = [pd.read_csv(pooch.os_cache("mlgeo") / fname) for fname in SEISMIC_FILES]
table = pd.concat(frames, ignore_index=True)
class_names = ["earthquake", "explosion", "noise", "surface event"]
labels4 = table["source"].map({name: i for i, name in enumerate(class_names)}).to_numpy()
features = table.drop(columns=["Unnamed: 0", "serial_no", "source"])
features = features.drop(columns=features.columns[features.isna().all()])
X4 = features.to_numpy(dtype=np.float32)
X_train, X_tmp, y_train, y_tmp = train_test_split(X4, labels4, test_size=0.4, random_state=42, stratify=labels4)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
scaler4 = StandardScaler().fit(X_train)
no_hidden = LogisticRegression(max_iter=5000).fit(scaler4.transform(X_train), y_train)
forest = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_train, y_train)
numbers["four_classes_split_of_4.1"] = {
    "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
    "no_hidden_layer_weights": int(no_hidden.coef_.size + no_hidden.intercept_.size),
    "no_hidden_layer_test_accuracy": float(no_hidden.score(scaler4.transform(X_test), y_test)),
    "no_hidden_layer_val_accuracy": float(no_hidden.score(scaler4.transform(X_val), y_val)),
    "forest_300_trees_default_test_accuracy": float(forest.score(X_test, y_test)),
    "from_notebook_4.1_one_hidden_layer_32": {"weights": 2116, "test_accuracy": 0.871, "val_accuracy": 0.885},
}

(FIGS / "numbers.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
