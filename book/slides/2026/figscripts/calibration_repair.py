"""Slide figure for lecture 16: the overconfident forest, repaired.

Reproduces notebook 3.6's reliability diagram — the raw 10-tree forest against
its isotonic-calibrated version on the water-potability test set — at 20pt.
Brier scores are asserted against the executed notebook (0.239 raw, 0.220
calibrated), so the slide numbers always match the book.

Regenerate: pixi run python book/slides/2026/figscripts/calibration_repair.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

TAG = ("Water-potability teaching table (provenance undocumented, likely "
      "synthetic) — 604 held-out samples")

path = pooch.retrieve(
    url="https://raw.githubusercontent.com/UW-MLGEO/MLGeo-dataset/main/data/water_potability.csv",
    known_hash=None, fname="water_potability.csv", path=pooch.os_cache("mlgeo"))
data = pd.read_csv(path).dropna().reset_index(drop=True)

x = data.drop(columns=["Potability"]).to_numpy()
y = data.Potability.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler().fit(x_train)
x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(x_train, y_train)
proba_rf = rf.predict_proba(x_test)[:, 1]
cal_rf = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=10, random_state=0),
    method="isotonic", cv=5).fit(x_train, y_train)
proba_cal = cal_rf.predict_proba(x_test)[:, 1]

brier_rf = brier_score_loss(y_test, proba_rf)
brier_cal = brier_score_loss(y_test, proba_cal)
acc_rf = np.mean((proba_rf > 0.5) == y_test)
acc_cal = np.mean((proba_cal > 0.5) == y_test)
assert f"{brier_rf:.3f}" == "0.239" and f"{brier_cal:.3f}" == "0.220", \
    f"Brier scores drifted from notebook: {brier_rf:.3f}, {brier_cal:.3f}"
print(f"raw forest        Brier {brier_rf:.3f}  accuracy {acc_rf:.3f}")
print(f"calibrated forest Brier {brier_cal:.3f}  accuracy {acc_cal:.3f}")

fig, ax = plt.subplots(figsize=(10.2, 8.8))
ax.plot([0, 1], [0, 1], "k--", lw=2, label="honest: stated = observed")
for proba, brier, label, color in [
        (proba_rf, brier_rf, "raw forest", "#1f77b4"),
        (proba_cal, brier_cal, "re-labeled (isotonic)", "#ff7f0e")]:
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", lw=3, ms=10, color=color,
            label=f"{label}  ·  Brier {brier:.3f}")
ax.annotate("claims 80–90%,\ndelivers ~50%",
            xy=(0.80, 0.51), xytext=(0.56, 0.24), fontsize=19,
            color="#b3402a", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#b3402a", lw=2))
ax.set_xlabel("Stated probability of potable")
ax.set_ylabel("Observed fraction potable")
ax.set_title("Stated probability vs observed frequency, held-out samples",
             loc="left")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=17)
fig.suptitle(TAG, x=0.01, ha="left", fontsize=18, fontweight="normal",
             color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = (Path(__file__).resolve().parent.parent / "figs"
       / "3.6_logistic_regression" / "calibration_repair_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
