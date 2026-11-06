"""Slide figures and numbers for lecture 16: binary, multiclass, logistic regression.

All on the Pacific Northwest seismic source tables (Zenodo record 14025693,
1000 per class, 61 waveform features), with the canonical leaderboard split of
notebook 3.5 (random_state=2026, stratified, 25% test). Writes the figures to
figs/lec16_classification/ and every number quoted on a slide to numbers.json.

Regenerate: pixi run python book/slides/2026/figscripts/lec16_classification.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, auc, average_precision_score, balanced_accuracy_score,  # noqa: E402
                             brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler, label_binarize  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c"
TAG = "Pacific Northwest seismic events, Zenodo record 14025693 (real data)"
SEISMIC_FILES = {
    "1000_earthquakes_physical_features.csv": "md5:28129c8dd1b3e14f655d489577b841b5",
    "1000_explosion_physical_features.csv": "md5:af1342d32e163e961e043364136359b0",
    "1000_noise_physical_features.csv": "md5:16cdb992fed6cf6273d5624f5df905da",
    "1000_surface_physical_features.csv": "md5:9a2c2643030cf058704d68e130654e9d",
}
FIGS = Path(__file__).resolve().parent.parent / "figs" / "lec16_classification"
FIGS.mkdir(parents=True, exist_ok=True)
numbers = {}


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def binary_metrics(y, p, threshold=0.5):
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"threshold": threshold, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "specificity": tn / (tn + fp), "f1": f1_score(y, yhat), "mcc": matthews_corrcoef(y, yhat)}


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


frames = []
for fname, checksum in SEISMIC_FILES.items():
    path = pooch.retrieve(url=f"https://zenodo.org/api/records/14025693/files/{fname}/content",
                          known_hash=checksum, fname=fname, path=pooch.os_cache("mlgeo"))
    frames.append(pd.read_csv(path, index_col=0))
seismic = pd.concat(frames, ignore_index=True).dropna(axis=1)
X = seismic.drop(columns=["source", "serial_no"])
y = seismic["source"]

# --- Binary: explosion (positive) against earthquake, balanced -----------------
two = y.isin(["earthquake", "explosion"])
Xb, yb = X[two], (y[two] == "explosion").astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, random_state=2026, stratify=yb)
binary = logistic().fit(Xtr, ytr)
p = binary.predict_proba(Xte)[:, 1]
numbers["binary_test_size"] = len(yte)
numbers["binary"] = [binary_metrics(yte, p, t) for t in (0.2, 0.5, 0.8)]
numbers["binary_scores"] = {"roc_auc": roc_auc_score(yte, p), "average_precision": average_precision_score(yte, p),
                            "brier": brier_score_loss(yte, p), "log_loss": log_loss(yte, p)}

fig, ax = plt.subplots(figsize=(9.5, 6.2))
bins = np.linspace(0, 1, 21)
ax.hist(p[yte == 0], bins=bins, color=BLUE, alpha=0.75, label="earthquakes")
ax.hist(p[yte == 1], bins=bins, color=ORANGE, alpha=0.75, label="explosions")
for t in (0.2, 0.5, 0.8):
    ax.axvline(t, color=INK, ls="--", lw=2)
    ax.text(t, ax.get_ylim()[1] * 1.01, f"{t}", ha="center", va="bottom", fontsize=18, color=INK)
ax.set(xlabel="Probability of explosion from logistic regression", ylabel="Events in the test set")
ax.legend(frameon=False, loc=(0.22, 0.72))
save(fig, "threshold_histogram")

# --- Binary: the three thresholds as points on the ROC and precision-recall curves --------
fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.0))
fpr, tpr, _ = roc_curve(yte, p)
prec, rec, _ = precision_recall_curve(yte, p)
left.plot(fpr, tpr, color=BLUE, lw=3, label=f"AUC {numbers['binary_scores']['roc_auc']:.3f}")
right.plot(rec, prec, color=BLUE, lw=3, label=f"AP {numbers['binary_scores']['average_precision']:.3f}")
for row in numbers["binary"]:
    t, r, f, q = row["threshold"], row["recall"], row["fp"] / (row["fp"] + row["tn"]), row["precision"]
    left.plot(f, r, "o", ms=15, mfc="white", mec=RED, mew=3, zorder=5)
    left.annotate(f"{t}", (f, r), xytext=(12, -24), textcoords="offset points", fontsize=19, color=RED, fontweight="bold")
    right.plot(r, q, "o", ms=15, mfc="white", mec=RED, mew=3, zorder=5)
    right.annotate(f"{t}", (r, q), xytext=(-34, -36), textcoords="offset points", fontsize=19, color=RED, fontweight="bold")
left.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=2)
left.set(xlabel="False-positive rate", ylabel="Recall", title="ROC curve", xlim=(0, 1), ylim=(0, 1.02))
right.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.02))
left.legend(frameon=False, loc="lower right")
right.legend(frameon=False, loc="lower left")
save(fig, "threshold_to_curves")

# --- Binary: explosions made rare (5%), repeated on 20 random subsets ----------
eq_index, ex_index = yb[yb == 0].index, yb[yb == 1].index
n_rare = round(len(eq_index) * 0.05 / 0.95)
rows, curves = [], None
for seed in range(20):
    chosen = np.random.default_rng(seed).choice(ex_index, size=n_rare, replace=False)
    idx = np.r_[eq_index, chosen]
    pr = cross_val_predict(logistic(), Xb.loc[idx], yb.loc[idx], method="predict_proba",
                           cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
    row = binary_metrics(yb.loc[idx], pr)
    row.update(roc_auc=roc_auc_score(yb.loc[idx], pr), average_precision=average_precision_score(yb.loc[idx], pr))
    rows.append(row)
    if seed == 0:
        curves = (yb.loc[idx].to_numpy(), pr)
rare = pd.DataFrame(rows)
numbers["rare"] = {"explosions": n_rare, "earthquakes": len(eq_index), "never_trigger_accuracy": 1 - n_rare / (n_rare + len(eq_index)),
                   **rare.drop(columns="threshold").mean().to_dict(), "sd_average_precision": rare["average_precision"].std(),
                   "sd_recall": rare["recall"].std()}
balanced_p = cross_val_predict(logistic(), Xb, yb, method="predict_proba", cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]

fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6.2))
for name, yy, pp, color in [("balanced (50%)", yb.to_numpy(), balanced_p, BLUE), (f"rare ({n_rare} explosions, 5%)", *curves, RED)]:
    fpr, tpr, _ = roc_curve(yy, pp)
    prec, rec, _ = precision_recall_curve(yy, pp)
    left.plot(fpr, tpr, color=color, lw=3, label=f"{name}: AUC {auc(fpr, tpr):.2f}")
    right.plot(rec, prec, color=color, lw=3, label=f"{name}: AP {average_precision_score(yy, pp):.2f}")
left.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=2)
left.set(xlabel="False-positive rate", ylabel="Recall", title="ROC curve", xlim=(0, 1), ylim=(0, 1.02))
right.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.02))
left.legend(frameon=False, fontsize=16, loc="lower right")
right.legend(frameon=False, fontsize=16, loc="lower left")
save(fig, "roc_pr_rare")

# --- Multiclass: canonical leaderboard split --------------------------------------
classes = ["earthquake", "explosion", "noise", "surface event"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2026, stratify=y)
multi = logistic().fit(X_train, y_train)
pred = multi.predict(X_test)
forest = RandomForestClassifier(random_state=42).fit(X_train, y_train)   # the forest of notebook 3.5
numbers["multi"] = {"accuracy": accuracy_score(y_test, pred), "macro_f1": f1_score(y_test, pred, average="macro"),
                    "micro_f1": f1_score(y_test, pred, average="micro"), "weighted_f1": f1_score(y_test, pred, average="weighted"),
                    "forest_macro_f1": f1_score(y_test, forest.predict(X_test), average="macro"),
                    "per_class": {c: {"precision": precision_score(y_test == c, pred == c), "recall": recall_score(y_test == c, pred == c),
                                      "f1": f1_score(y_test == c, pred == c)} for c in classes}}
matrix = confusion_matrix(y_test, pred, labels=classes)
numbers["multi"]["confusion"] = matrix.tolist()

fig, ax = plt.subplots(figsize=(8.6, 6.8))
ax.imshow(matrix, cmap="Blues", vmin=0, vmax=250)
for i in range(4):
    for j in range(4):
        ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=24, fontweight="bold",
                color="white" if matrix[i, j] > 125 else INK)
names = ["earthquake", "explosion", "noise", "surface\nevent"]
ax.set_xticks(range(4), names)
ax.set_yticks(range(4), names)
ax.set(xlabel="Predicted by logistic regression", ylabel="Analyst label")
for side in ax.spines.values():
    side.set_visible(False)
save(fig, "confusion_logistic")

scores = multi.predict_proba(X_test)
truth = label_binarize(y_test, classes=classes)
fig, ax = plt.subplots(figsize=(8, 6.8))
numbers["multi"]["ovr_auc"] = {}
for j, (c, color) in enumerate(zip(classes, [BLUE, ORANGE, GREEN, RED])):
    fpr, tpr, _ = roc_curve(truth[:, j], scores[:, list(multi.classes_).index(c)])
    numbers["multi"]["ovr_auc"][c] = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=3, label=f"{c}: {auc(fpr, tpr):.3f}")
ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=2)
ax.set(xlabel="False-positive rate", ylabel="Recall", xlim=(0, 1), ylim=(0, 1.02))
ax.legend(frameon=False, title="One against the rest, AUC", fontsize=17, title_fontsize=17, loc="lower right")
save(fig, "ovr_roc_logistic")

# --- Multiclass with unequal classes: how the three averages separate ---------------
counts = {"noise": 1000, "earthquake": 500, "explosion": 100, "surface event": 50}
rows = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(y[y == c].index, size=n, replace=False) for c, n in counts.items()])
    pr = cross_val_predict(logistic(), X.loc[idx], y.loc[idx], cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    row = {"accuracy": accuracy_score(y.loc[idx], pr), "macro_f1": f1_score(y.loc[idx], pr, average="macro"),
           "micro_f1": f1_score(y.loc[idx], pr, average="micro"), "weighted_f1": f1_score(y.loc[idx], pr, average="weighted")}
    row.update({f"f1 {c}": f1_score(y.loc[idx] == c, pr == c) for c in counts})
    rows.append(row)
numbers["unequal"] = {"counts": counts, **pd.DataFrame(rows).mean().to_dict()}

# --- Logistic regression: score, sigmoid, loss ---------------------------------------
z = binary.decision_function(Xte)
fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.6))
rng = np.random.default_rng(1)
grid = np.linspace(-10, 10, 300)
left.plot(grid, 1 / (1 + np.exp(-grid)), color=INK, lw=3)
left.scatter(z[yte == 0], 0.0 + rng.uniform(-0.03, 0.03, (yte == 0).sum()), s=26, color=BLUE, alpha=0.6, label="earthquakes")
left.scatter(z[yte == 1], 1.0 + rng.uniform(-0.03, 0.03, (yte == 1).sum()), s=26, color=ORANGE, marker="s", alpha=0.6, label="explosions")
left.axhline(0.5, color=MUTED, ls="--", lw=2)
left.set(xlabel="Score $z = b + w_1 x_1 + \\dots + w_{61} x_{61}$", ylabel="Probability of explosion", xlim=(-10, 10), title="Sigmoid: score to probability")
left.legend(frameon=False, loc="center right", fontsize=16)
prob = np.linspace(0.005, 1, 300)
right.plot(prob, -np.log(prob), color=ORANGE, lw=3, label="true class: explosion")
right.plot(1 - prob, -np.log(prob), color=BLUE, lw=3, label="true class: earthquake")
right.set(xlabel="Stated probability of explosion", ylabel="Log-loss for one event", ylim=(0, 5.3), title="Confident errors cost the most")
right.legend(frameon=False, fontsize=16, loc="upper center")
save(fig, "sigmoid_logloss")
numbers["logloss_examples"] = {str(q): -np.log(q) for q in (0.99, 0.55, 0.45, 0.01)}

weights = pd.Series(binary[-1].coef_[0], index=X.columns)
top = weights.reindex(weights.abs().sort_values(ascending=False).index[:10]).iloc[::-1]
numbers["top_weights"] = top.iloc[::-1].round(2).to_dict()
numbers["feature_correlation_E_1_3_Fquart1"] = float(X["E_1_3"].corr(X["Fquart1"])) if {"E_1_3", "Fquart1"} <= set(X.columns) else None
fig, ax = plt.subplots(figsize=(8, 6.4))
ax.barh(top.index, top.to_numpy(), color=[ORANGE if v > 0 else BLUE for v in top], height=0.72)
ax.axvline(0, color=INK, lw=1.5)
ax.set(xlabel="Weight (per standard deviation)\nblue: toward earthquake, orange: toward explosion")
save(fig, "top_weights")

# --- Calibration: out-of-fold probabilities, logistic against a small deep forest -------
cv = StratifiedKFold(5, shuffle=True, random_state=0)
small_forest = RandomForestClassifier(n_estimators=10, random_state=0)
forest_p = cross_val_predict(small_forest, Xb, yb, method="predict_proba", cv=cv)[:, 1]
logit_p = cross_val_predict(logistic(), Xb, yb, method="predict_proba", cv=cv)[:, 1]
numbers["calibration"] = {"logistic_brier": brier_score_loss(yb, logit_p), "forest10_brier": brier_score_loss(yb, forest_p),
                          "logistic_accuracy": accuracy_score(yb, logit_p >= 0.5), "forest10_accuracy": accuracy_score(yb, forest_p >= 0.5)}
fig, ax = plt.subplots(figsize=(8, 6.6))
for name, pp, color, marker in [("logistic regression", logit_p, BLUE, "o"), ("forest of 10 trees", forest_p, RED, "s")]:
    observed, stated = calibration_curve(yb, pp, n_bins=10)
    ax.plot(stated, observed, color=color, marker=marker, ms=9, lw=3, label=name)
ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=2)
ax.set(xlabel="Stated probability of explosion", ylabel="Observed share of explosions", xlim=(0, 1), ylim=(0, 1))
ax.legend(frameon=False, loc="upper left")
save(fig, "reliability")

(FIGS / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
print(json.dumps(numbers, indent=0, default=lambda v: round(float(v), 3)))
