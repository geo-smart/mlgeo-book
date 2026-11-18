"""Slide figures for lecture 21b: loss, optimization, training and evaluation, for a regression and a classification.

Regression data (real): GNSS station P395, Oregon Coast Range, NGL daily east position, course file
book/Chapter1-GettingStarted/data/gps_P395_relative_position.csv. Same task as lecture 18 and notebook 4.2:
six numbers from the past 60 days predict the mean east position of the next 30 days, one sample every
5 days. Split in time: first 70% train, gap of 6 samples, 15% validation, gap of 6, last 15% test.

The network is the one of notebook 4.2: 6 (or 5) inputs, two hidden layers of 32 ReLU neurons, one output
with no activation, Adam, batch 64, starting weights drawn with a seed. Every number quoted on a slide is
written to figs/lec21b_train_evaluate/numbers.json.

Figures:
  losses_regression.png     loss of one sample against its error: squared, absolute, Huber
  loss_classification.png   cross-entropy of one sample against the probability given to the observed class
  p395_samples.png          the series, one sample's windows, and the split in time
  learning_rates.png        training and validation loss per epoch for three learning rates
  early_stopping.png        training and validation loss over 1,000 epochs, with the checkpoint
  range_trap.png            training and test target ranges; the network on the absolute target in the test period
  p395_scores.png           test error in mm: baselines and networks, three seeds each

Regenerate: pixi run python book/slides/2026/figscripts/lec21b_train_evaluate.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FIGS = HERE.parent / "figs" / "lec21b_train_evaluate"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 22, "axes.titlesize": 24, "axes.titleweight": "bold",
    "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, PURPLE, INK, MUTED, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#26241f", "#6e675c", "#b8b1a4"
TAG = "GNSS station P395, Oregon Coast Range, daily east position (NGL)"
numbers = {}
torch.set_num_threads(1)


def save(fig, name, tag=None):
    if tag:
        fig.text(0.5, -0.03, tag, ha="center", fontsize=17, color=MUTED)
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Losses
# =============================================================================
e = np.linspace(-4, 4, 400)
huber = np.where(np.abs(e) <= 1, 0.5 * e ** 2, np.abs(e) - 0.5)
fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(e, e ** 2, color=BLUE, lw=5, label="squared error (MSE)")
ax.plot(e, np.abs(e), color=GREEN, lw=5, label="absolute error (MAE)")
ax.plot(e, huber, color=ORANGE, lw=5, ls="--", label="Huber (δ = 1)")
ax.set(xlabel="error of one sample: prediction − observation", ylabel="loss of the sample", ylim=(0, 8))
ax.annotate("an error of 2.6 costs 6.8:\nlarge errors dominate the mean", xy=(2.6, 6.76), xytext=(-3.9, 5.2), fontsize=19, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
ax.annotate("costs 3: outliers\ncount less", xy=(3, 3), xytext=(2.0, 1.0), fontsize=19, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
ax.legend(frameon=False, loc="upper center", fontsize=20)
save(fig, "losses_regression")

p = np.linspace(0.005, 1, 400)
fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(p, -np.log(p), color=PURPLE, lw=5)
for pp, lab in [(0.25, "1/4 to each of 4 classes:\nloss ln 4 = 1.39"), (0.9, "0.9: loss 0.11")]:
    ax.scatter([pp], [-np.log(pp)], s=200, color=INK, zorder=3)
    ax.annotate(lab, xy=(pp, -np.log(pp)), xytext=(pp + 0.08, -np.log(pp) + 0.9), fontsize=19, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=2))
ax.annotate("confident and wrong:\nthe loss grows without bound", xy=(0.02, -np.log(0.02)), xytext=(0.12, 4.6), fontsize=19, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.set(xlabel="probability the network gives to the observed class", ylabel="cross-entropy of the sample", ylim=(0, 5.5))
save(fig, "loss_classification")

# =============================================================================
# P395: samples and split
# =============================================================================
gnss = pd.read_csv(ROOT / "book" / "Chapter1-GettingStarted" / "data" / "gps_P395_relative_position.csv")
east = gnss["east_m"].to_numpy() * 1000
year = gnss["decimal_year"].to_numpy()
rows = []
for i in range(60, len(east) - 30, 5):
    past = east[i - 60:i]
    rows.append({"mean_60d": past.mean(), "mean_10d": east[i - 10:i].mean(), "last": east[i - 1],
                 "slope_60d": np.polyfit(np.arange(60), past, 1)[0],
                 "season_sin": np.sin(2 * np.pi * year[i]), "season_cos": np.cos(2 * np.pi * year[i]),
                 "target": east[i:i + 30].mean(), "year": year[i], "i": i})
s = pd.DataFrame(rows)
n = len(s)
GAP = 6
i1, i2 = int(0.7 * n), int(0.85 * n)
tr, va, te = np.arange(0, i1), np.arange(i1 + GAP, i2), np.arange(i2 + GAP, n)
F = ["mean_60d", "mean_10d", "last", "slope_60d", "season_sin", "season_cos"]
X, y, last = s[F].to_numpy(), s["target"].to_numpy(), s["last"].to_numpy()
Xc = np.column_stack([s.mean_60d - last, s.mean_10d - last, s.slope_60d, s.season_sin, s.season_cos])
yc = y - last
velocity = np.polyfit(year, east, 1)[0]
numbers["data"] = {"samples": n, "train": len(tr), "val": len(va), "test": len(te),
                   "train_years": [float(s.year[tr[0]]), float(s.year[tr[-1]])], "val_years": [float(s.year[va[0]]), float(s.year[va[-1]])],
                   "test_years": [float(s.year[te[0]]), float(s.year[te[-1]])], "mean_east_velocity_mm_per_yr": float(velocity),
                   "train_observation_range_mm": [float(y[tr].min()), float(y[tr].max())], "test_observation_range_mm": [float(y[te].min()), float(y[te].max())]}

fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(year, east, ".", ms=2, color=GRAY)
for idx, color, lab in [(tr, BLUE, "training samples"), (va, ORANGE, "validation"), (te, RED, "test")]:
    ax.axvspan(s.year[idx[0]], s.year[idx[-1]], color=color, alpha=0.12, label=lab)
k = s.i[200]
ax.plot(year[k - 60:k], east[k - 60:k], color=BLUE, lw=4)
ax.hlines(east[k:k + 30].mean(), year[k], year[k + 29], color=RED, lw=5)
ax.annotate("one sample: 60 days in (6 numbers),\nmean of the next 30 days out", xy=(year[k], east[k]), xytext=(year[k] + 1.5, east[k] + 25),
            fontsize=19, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=2))
ax.set(xlabel="Year", ylabel="East position (mm)")
ax.legend(frameon=False, loc="lower left", fontsize=19)
save(fig, "p395_samples", TAG + f". {n} samples, one every 5 days.")


# =============================================================================
# The network of notebook 4.2
# =============================================================================
def fit_mlp(Xa, ya, loss="mse", seed=0, epochs=300, lr=1e-3, batch=64, record=False):
    """Same training as notebook 4.2: seeded starting weights, DataLoader with a seeded generator, CPU."""
    from torch.utils.data import DataLoader, TensorDataset
    sx, sy = StandardScaler().fit(Xa[tr]), StandardScaler().fit(ya[tr, None])
    Xt = torch.tensor(sx.transform(Xa).astype(np.float32))
    yt = torch.tensor(sy.transform(ya[:, None]).astype(np.float32))
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(Xa.shape[1], 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    crit = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss(delta=1.0)}[loss]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xt[tr], yt[tr]), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    hist = {"train": [], "val": []}
    best = (np.inf, None, 0)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            l = crit(model(xb), yb)
            l.backward()
            opt.step()
        if record:
            model.eval()
            with torch.no_grad():
                lt, lv = crit(model(Xt[tr]), yt[tr]).item(), crit(model(Xt[va]), yt[va]).item()
            hist["train"].append(lt)
            hist["val"].append(lv)
            if lv < best[0]:
                best = (lv, None, ep)
    model.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(model(Xt).numpy())[:, 0]
    return pred, hist, best


mae = lambda a, b: float(np.mean(np.abs(a - b)))  # noqa: E731

# --- learning rates ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
lr_runs = {}
for lr, color in [(1e-4, BLUE), (1e-3, GREEN), (3e-2, RED)]:
    _, h, _ = fit_mlp(Xc, yc, "mse", seed=0, epochs=200, lr=lr, record=True)
    ax.semilogy(h["train"], color=color, lw=4, label=f"learning rate {lr:g}")
    lr_runs[str(lr)] = {"train_last": h["train"][-1], "val_last": h["val"][-1], "val_min": min(h["val"])}
numbers["learning_rates"] = lr_runs
ax.set(xlabel="Epoch (17 batches of 64 samples each)", ylabel="Training loss: MSE (standardized units)")
ax.legend(frameon=False, fontsize=20, loc="upper right")
save(fig, "learning_rates", TAG + ". Network predicting the change, Adam, seed 0.")

# --- early stopping and checkpoint -----------------------------------------------------
_, h, best = fit_mlp(Xc, yc, "mse", seed=0, epochs=1000, lr=1e-3, record=True)
fig, ax = plt.subplots(figsize=(14, 7))
ax.semilogy(h["train"], color=INK, lw=4, label="training samples")
ax.semilogy(h["val"], color=ORANGE, lw=4, ls="--", label="validation samples")
ax.axvline(best[2], color=GREEN, lw=3)
ax.text(best[2] + 25, 0.5, f"checkpoint: lowest validation loss,\nepoch {best[2] + 1}", color=GREEN, fontsize=20, va="center")
ax.set(xlabel="Epoch", ylabel="Mean squared error (standardized units)")
ax.legend(frameon=False, fontsize=20, loc="center right")
numbers["early_stopping"] = {"best_epoch": best[2] + 1, "val_min": best[0], "train_last": h["train"][-1], "val_last": h["val"][-1]}
save(fig, "early_stopping", TAG + ". Network predicting the change, 1,000 epochs, seed 0.")

# --- the range trap ---------------------------------------------------------------------
pred_abs = {seed: fit_mlp(X, y, "mse", seed=seed)[0] for seed in (0, 1, 2)}
fig, (left, right) = plt.subplots(1, 2, figsize=(19, 7), gridspec_kw={"width_ratios": [1, 2.1], "wspace": 0.25})
left.boxplot([y[tr], y[te]], widths=0.55, patch_artist=True, boxprops=dict(facecolor="#dbe9f4", color=INK), medianprops=dict(color=INK))
left.set_xticks([1, 2], ["training\nsamples", "test\nsamples"])
left.set_ylabel("Mean east position, next 30 days (mm)")
left.set_title("No overlap")
right.plot(s.year[te], y[te], color=INK, lw=3, label="observed")
right.plot(s.year[te], last[te], color=GREEN, lw=2, label="persistence")
right.plot(s.year[te], pred_abs[0][te], color=RED, lw=3, label="network, predicts the position")
right.set(xlabel="Year (test period)", ylabel="East position (mm)", title="The network cannot follow")
right.legend(frameon=False, fontsize=19)
save(fig, "range_trap", TAG + ".")

# --- scores ----------------------------------------------------------------------------
scores = {"persistence": [mae(y[te], last[te])],
          "linear regression": [mae(y[te], LinearRegression().fit(X[tr], y[tr]).predict(X[te]))]}
scores["network, predicts the position"] = [mae(y[te], pred_abs[sd][te]) for sd in (0, 1, 2)]
for loss in ("mse", "mae", "huber"):
    scores[f"network, predicts the change, {loss.upper() if loss != 'huber' else 'Huber'}"] = [
        mae(y[te], last[te] + fit_mlp(Xc, yc, loss, seed=sd)[0][te]) for sd in (0, 1, 2)]
numbers["test_mae_mm"] = {k: {"values": v, "mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores.items()}
names = list(scores)
fig, ax = plt.subplots(figsize=(15, 7.4))
means = [np.mean(scores[k]) for k in names]
stds = [np.std(scores[k]) for k in names]
colors = [GRAY, BLUE, RED, GREEN, GREEN, GREEN]
ax.barh(range(len(names)), means, xerr=stds, color=colors, height=0.6, error_kw=dict(lw=3, capsize=8, ecolor=INK))
for k, (m, sd) in enumerate(zip(means, stds)):
    ax.text(m + sd + 0.05, k, f"{m:.2f}" + ((" ± < 0.01" if sd < 0.005 else f" ± {sd:.2f}") if sd > 0 else ""), va="center", fontsize=20)
ax.set_yticks(range(len(names)), names)
ax.invert_yaxis()
ax.set(xlabel="Test error: mean absolute error of the 30-day position (mm)", xlim=(0, max(means) * 1.35))
save(fig, "p395_scores", TAG + ". Test period " + f"{s.year[te[0]]:.1f}–{s.year[te[-1]]:.1f}; networks: mean ± std over seeds 0, 1, 2.")

(FIGS / "numbers.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
