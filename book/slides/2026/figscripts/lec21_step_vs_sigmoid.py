"""Slide figure for lecture 21, Part 1: why the 1958 rule cannot settle, and why the sigmoid can.

Data: the 2,000 synthetic detector windows of notebook 4.0, section 4
(mlgeo_synth.detector_features, seed 0): STA/LTA and spectral centroid,
1,500 windows to train, standardized on them. The classes overlap.

Left panel: what each neuron can lower, drawn against one weight (the weight
of STA/LTA), the other weight and the bias held at the values the log-loss
training reaches. With the step, the only measure is the share of windows
with a nonzero error: flat steps, slope zero almost everywhere. With the
sigmoid, the mean log-loss: a smooth curve with a slope at every point and
one lowest point.

Right panel: the direction of the boundary line at the end of each epoch,
for the 1958 rule and for gradient descent on the log-loss, both from the
same random starting weights (seed 0).

Writes figs/lec21_neuron/step_vs_sigmoid.png and numbers_step_vs_sigmoid.json.
Regenerate: pixi run python book/slides/2026/figscripts/lec21_step_vs_sigmoid.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # book/, for mlgeo_synth
import mlgeo_synth  # noqa: E402

plt.rcParams.update({
    "font.size": 22, "axes.titlesize": 24, "axes.titleweight": "bold",
    "axes.labelsize": 22, "xtick.labelsize": 19, "ytick.labelsize": 19,
    "axes.spines.top": False, "axes.spines.right": False,
})
GREEN, RED, INK, MUTED = "#009E73", "#D55E00", "#26241f", "#6e675c"
FIGS = HERE.parent / "figs" / "lec21_neuron"

df = mlgeo_synth.detector_features(n=2000, event_fraction=0.5, seed=0)
X = df[["sta_lta", "spectral_centroid_hz"]].to_numpy()[:1500]
y = df["label"].to_numpy()[:1500]
X = (X - X.mean(axis=0)) / X.std(axis=0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def start():
    r = np.random.default_rng(0)
    return r.normal(0, 1, 2), r.normal(0, 1)


def angle(w):
    return float(np.degrees(np.arctan2(w[1], w[0])))


EPOCHS = 300
# gradient descent on the mean log-loss (notebook 4.0, section 4.1)
w, b = start()
ang_loss = []
for _ in range(EPOCHS):
    err = sigmoid(X @ w + b) - y
    w, b = w - 2.0 * X.T @ err / len(y), b - 2.0 * err.mean()
    ang_loss.append(angle(w))
w_fit, b_fit = w.copy(), b

# the 1958 rule, one sample at a time (notebook 4.0, section 4.2)
w, b = start()
ang_rule = []
for _ in range(EPOCHS):
    for i in range(len(y)):
        err = float(X[i] @ w + b >= 0) - y[i]
        if err != 0:
            w, b = w - err * X[i], b - err
    ang_rule.append(angle(w))

# what each neuron can lower, against the weight of STA/LTA, on 40 of the windows so that the steps show
Xs, ys = X[:40], y[:40]
w1 = np.linspace(-1, 2 * w_fit[0], 1200)
share_wrong = np.array([np.mean(((Xs @ np.array([a, w_fit[1]]) + b_fit) >= 0) != ys) for a in w1])
logloss = []
for a in w1:
    p = np.clip(sigmoid(Xs @ np.array([a, w_fit[1]]) + b_fit), 1e-9, 1 - 1e-9)
    logloss.append(-np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p)))
logloss = np.array(logloss)

fig = plt.figure(figsize=(20, 7.8))
grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.35, wspace=0.25)
top, bot, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
top.plot(w1, 100 * share_wrong, color=RED, lw=4, drawstyle="steps-mid")
top.set_title("Step: % of windows with an error", color=RED, fontsize=22, loc="left")
top.set_xticklabels([])
top.text(0.98, 0.9, "flat, then a jump:\nslope zero, no direction", transform=top.transAxes, ha="right", va="top", fontsize=19, color=RED)
bot.plot(w1, logloss, color=GREEN, lw=4.5)
k = np.argmin(np.abs(w1 - 0.35 * w_fit[0]))
seg = slice(k - 90, k + 90)
slope = np.gradient(logloss, w1)[k]
bot.plot(w1[seg], logloss[k] + slope * (w1[seg] - w1[k]), color=INK, lw=3, ls="--")
bot.scatter([w1[k]], [logloss[k]], s=120, color=INK, zorder=3)
bot.set_title("Sigmoid: mean log-loss", color=GREEN, fontsize=22, loc="left")
bot.text(0.98, 0.9, "a slope at every point:\nthe gradient gives the direction", transform=bot.transAxes, ha="right", va="top", fontsize=19, color=GREEN)
bot.set_xlabel("weight of STA/LTA (other weight and bias fixed)")
right.plot(range(1, EPOCHS + 1), ang_rule, color=RED, lw=2.2, label="1958 rule (step)")
right.plot(range(1, EPOCHS + 1), ang_loss, color=GREEN, lw=4.5, label="gradient descent on the log-loss (sigmoid)")
right.set(xlabel="epoch", ylabel="direction of the boundary (degrees)")
right.set_title("The boundary at the end of each epoch")
right.legend(frameon=False, fontsize=19, loc="upper right")
lo, hi = min(ang_rule[20:] + ang_loss[20:]), max(ang_rule[20:] + ang_loss[20:])
right.set_ylim(lo - 4, hi + 12)
fig.text(0.5, -0.06, "Synthetic (mlgeo_synth) detector windows of notebook 4.0, section 4.\n"
         "Left: 40 of the windows. Right: all 1,500 training windows, same random starting weights (seed 0).", ha="center", fontsize=17, color=MUTED)
fig.savefig(FIGS / "step_vs_sigmoid.png", dpi=150, bbox_inches="tight")

last = slice(EPOCHS - 100, EPOCHS)
numbers = {
    "distinct_values_share_wrong_along_w1": int(len(np.unique(share_wrong))),
    "rule_angle_last100_min_max_deg": [float(np.min(ang_rule[last])), float(np.max(ang_rule[last]))],
    "loss_angle_last100_min_max_deg": [float(np.min(ang_loss[last])), float(np.max(ang_loss[last]))],
    "loss_fit_w": w_fit.tolist(), "loss_fit_b": float(b_fit),
}
(FIGS / "numbers_step_vs_sigmoid.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
