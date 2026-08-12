"""Slide figure: ROC vs precision-recall on the 1:50 imbalanced detector.

Reproduces notebook 3.4's imbalanced experiment exactly (mlgeo_synth
detector_features, n=5000, event_fraction=0.02, seed=42; stratified 60/40
split, random_state=42; scaler fit on the training split only; logistic
regression and a 100-tree random forest) and draws the two curves side by
side at lecture scale. Prints the operating-point metrics so quoted slide
numbers can be verified against the notebook output.

Regenerate: pixi run python book/slides/2026/figscripts/roc_pr_imbalanced.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

BOOK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BOOK))
import mlgeo_synth  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay,  # noqa: E402
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

df = mlgeo_synth.detector_features(n=5000, event_fraction=0.02, seed=42)
X = np.column_stack([np.log10(df["sta_lta"]), df["spectral_centroid_hz"]])
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.4, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

logreg = LogisticRegression().fit(X_train, y_train)
forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
logreg_bal = LogisticRegression(class_weight="balanced").fit(X_train, y_train)

# verification against the executed notebook 3.4 outputs
for name, yp in [
    ("logreg thr 0.5   ", logreg.predict(X_test)),
    ("logreg balanced  ", logreg_bal.predict(X_test)),
    ("logreg thr 0.25  ", (logreg.predict_proba(X_test)[:, 1] >= 0.25).astype(int)),
    ("forest thr 0.5   ", forest.predict(X_test)),
]:
    print(f"{name} precision {precision_score(y_test, yp):.3f} "
          f"recall {recall_score(y_test, yp):.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.2))
for name, model, color in [("log. regression", logreg, "#1f77b4"),
                           ("forest", forest, "#b3402a")]:
    kw = {"color": color, "linewidth": 2.5}
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax1, name=name,
                                   curve_kwargs=kw)
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax2,
                                          name=name, curve_kwargs=kw)
ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5)
ax1.set_title("ROC: everything looks solved", loc="left")
ax1.set_xlabel("false positive rate")
ax1.set_ylabel("true positive rate (recall)")
ax2.set_title("Precision–recall: the real trade-off", loc="left")
ax2.set_xlabel("recall — events caught")
ax2.set_ylabel("precision — trigger list quality")
ax1.legend(loc="lower right", fontsize=15, framealpha=0.95)
ax2.legend(loc="lower left", fontsize=15, framealpha=0.95)
for ax in (ax1, ax2):
    ax.grid(True, alpha=0.4)

fig.suptitle("Event vs noise, 100 events among 4900 noise windows — synthetic (mlgeo_synth)",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = (Path(__file__).resolve().parent.parent / "figs" / "3.4_binary_classification"
       / "roc_pr_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
