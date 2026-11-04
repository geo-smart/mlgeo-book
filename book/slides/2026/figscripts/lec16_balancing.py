"""Slide figures and numbers for lecture 16: balancing classes in training, testing and deployment.

Explosion (positive) against earthquake, Pacific Northwest seismic tables (Zenodo record 14025693).
Rare cases keep all 1,000 earthquakes and a random subset of explosions; every model is scored
out of fold at the proportion it was given, 5-fold cross-validation, mean of 10 random subsets.
Writes figures to figs/lec16_classification/ and the numbers quoted on slides to numbers_balancing.json.

Regenerate: pixi run python book/slides/2026/figscripts/lec16_balancing.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c"
FILES = {"1000_earthquakes_physical_features.csv": "md5:28129c8dd1b3e14f655d489577b841b5",
         "1000_explosion_physical_features.csv": "md5:af1342d32e163e961e043364136359b0"}
FIGS = Path(__file__).resolve().parent.parent / "figs" / "lec16_classification"
SHARES = [0.5, 0.2, 0.1, 0.05, 0.02]
SEEDS = range(10)
numbers = {}

frames = [pd.read_csv(pooch.retrieve(url=f"https://zenodo.org/api/records/14025693/files/{name}/content", known_hash=checksum,
                                     fname=name, path=pooch.os_cache("mlgeo")), index_col=0) for name, checksum in FILES.items()]
table = pd.concat(frames, ignore_index=True).dropna(axis=1)
X = table.drop(columns=["source", "serial_no"]).to_numpy()
y = (table["source"] == "explosion").astype(int).to_numpy()


def logistic(weights=None):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight=weights))


def synthetic_minority(X_pos, n, rng, k=5):
    """SMOTE: new samples on the line between a minority sample and one of its k nearest minority neighbors."""
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(X_pos))).fit(X_pos).kneighbors(X_pos, return_distance=False)[:, 1:]
    a = rng.integers(0, len(X_pos), n)
    b = neighbors[a, rng.integers(0, neighbors.shape[1], n)]
    return X_pos[a] + rng.random((n, 1)) * (X_pos[b] - X_pos[a])


def fit_strategies(X_train, y_train, X_test, rng):
    pos, neg = np.where(y_train == 1)[0], np.where(y_train == 0)[0]
    plain = logistic().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    out = {"none": plain, "class weights": logistic("balanced").fit(X_train, y_train).predict_proba(X_test)[:, 1]}
    keep = np.r_[pos, rng.choice(neg, len(pos), replace=False)]
    out["undersample"] = logistic().fit(X_train[keep], y_train[keep]).predict_proba(X_test)[:, 1]
    copies = np.r_[neg, pos, rng.choice(pos, len(neg) - len(pos), replace=True)] if len(neg) > len(pos) else np.r_[neg, pos]
    out["oversample"] = logistic().fit(X_train[copies], y_train[copies]).predict_proba(X_test)[:, 1]
    new = synthetic_minority(X_train[pos], max(len(neg) - len(pos), 0), rng)
    out["SMOTE"] = logistic().fit(np.r_[X_train, new], np.r_[y_train, np.ones(len(new), int)]).predict_proba(X_test)[:, 1]
    return out


def subset(share, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.r_[neg, rng.choice(pos, round(len(neg) * share / (1 - share)), replace=False)]


def score(y_true, p, threshold=0.5):
    yhat = p >= threshold
    return {"precision": precision_score(y_true, yhat, zero_division=0), "recall": recall_score(y_true, yhat), "f1": f1_score(y_true, yhat),
            "ap": average_precision_score(y_true, p), "brier": brier_score_loss(y_true, p), "mean_p": p.mean()}


rows = []
for share in SHARES:
    for seed in SEEDS:
        idx = subset(share, seed)
        Xs, ys, rng = X[idx], y[idx], np.random.default_rng(seed)
        oof = {}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
            for name, p in fit_strategies(Xs[tr], ys[tr], Xs[te], rng).items():
                oof.setdefault(name, np.zeros(len(ys)))[te] = p
        for name, p in oof.items():
            rows.append({"share": share, "seed": seed, "strategy": name, **score(ys, p)})
        if share == 0.05:
            rows.append({"share": share, "seed": seed, "strategy": "threshold 0.2", **score(ys, oof["none"], 0.2)})
            r = share / (1 - share)
            corrected = np.clip(oof["class weights"] * r / (oof["class weights"] * r + 1 - oof["class weights"]), 0, 1)
            rows.append({"share": share, "seed": seed, "strategy": "class weights, prior corrected", **score(ys, corrected)})
results = pd.DataFrame(rows).groupby(["share", "strategy"]).mean(numeric_only=True).drop(columns="seed")
numbers["strategies"] = {f"{s}|{name}": row.to_dict() for (s, name), row in results.iterrows()}

# --- The same balanced model deployed at other proportions ----------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=2026, stratify=y)
balanced = logistic().fit(X_tr, y_tr)
tn, fp, fn, tp = confusion_matrix(y_te, balanced.predict_proba(X_te)[:, 1] >= 0.5).ravel()
tpr, fpr = tp / (tp + fn), fp / (fp + tn)


def precision_at(share):
    return share * tpr / (share * tpr + (1 - share) * fpr)


numbers["deployment"] = {"recall": tpr, "false_positive_rate": fpr,
                         **{str(s): {"precision": precision_at(s), "false_alarms_per_caught": (1 - precision_at(s)) / precision_at(s)} for s in (0.5, 0.2, 0.05, 0.01, 0.0008)}}
rng = np.random.default_rng(0)
pos_te, neg_te = np.where(y_te == 1)[0], np.where(y_te == 0)[0]
p_te = balanced.predict_proba(X_te)[:, 1]
draws = [precision_score(np.r_[np.ones(13), np.zeros(len(neg_te))], np.r_[p_te[rng.choice(pos_te, 13, replace=False)], p_te[neg_te]] >= 0.5, zero_division=0) for _ in range(500)]
numbers["deployment"]["empirical_precision_at_5pct_250_earthquakes"] = float(np.mean(draws))

fig, ax = plt.subplots(figsize=(9, 6.2))
shares = np.logspace(np.log10(0.0008), np.log10(0.5), 200)
ax.plot(shares, precision_at(shares), color=RED, lw=3.5, label="precision")
ax.axhline(tpr, color=BLUE, lw=3, ls="--", label="recall")
for s, dx, label in [(0.5, -0.31, "tested on 50%"), (0.05, 0.0, "5%"), (0.01, 0.0, "1%"), (0.0008, 0.0, "Sitka's rarity,\n0.08%")]:
    ax.plot(s, precision_at(s), "o", color=RED, ms=12)
    ax.annotate(f"{label}: {precision_at(s):.2f}", (s, precision_at(s)), xytext=(s * 10 ** dx * 1.25, precision_at(s) + 0.07), fontsize=17, color=INK)
ax.set_xscale("log")
ax.set(xlabel="Share of explosions among the events in deployment", ylabel="Score of the same model", ylim=(0, 1.05), xlim=(0.0006, 0.7))
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.62))
fig.savefig(FIGS / "balance_deployment.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- When it matters: recall, precision, average precision against rarity --------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharex=True)
for ax, key, title in zip(axes, ["recall", "precision", "ap"], ["Recall at 0.5", "Precision at 0.5", "Average precision"]):
    for name, color, marker in [("none", BLUE, "o"), ("class weights", ORANGE, "s"), ("undersample", RED, "^")]:
        values = [results.loc[(s, name), key] for s in SHARES]
        ax.plot([100 * s for s in SHARES], values, color=color, marker=marker, ms=10, lw=3, label=name)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set(title=title, xlabel="Explosions in the data (%)", ylim=(0, 1))
axes[0].legend(frameon=False, fontsize=16, loc="lower right")
fig.tight_layout()
fig.savefig(FIGS / "balance_when.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Balancing before the split: copies in both sets -----------------------------------
leak = []
for seed in SEEDS:
    idx = subset(0.05, seed)
    Xs, ys, rng = X[idx], y[idx], np.random.default_rng(seed)
    pos, neg = np.where(ys == 1)[0], np.where(ys == 0)[0]
    copies = np.r_[neg, pos, rng.choice(pos, len(neg) - len(pos), replace=True)]
    Xo, yo = Xs[copies], ys[copies]
    Xa, Xb, ya, yb = train_test_split(Xo, yo, test_size=0.25, random_state=seed, stratify=yo)
    wrong = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2).fit(Xa, ya).predict_proba(Xb)[:, 1]
    Xa, Xb, ya, yb2 = train_test_split(Xs, ys, test_size=0.25, random_state=seed, stratify=ys)
    p2, n2 = np.where(ya == 1)[0], np.where(ya == 0)[0]
    tr = np.r_[n2, p2, rng.choice(p2, len(n2) - len(p2), replace=True)]
    right = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2).fit(Xa[tr], ya[tr]).predict_proba(Xb)[:, 1]
    leak.append({"before_precision": precision_score(yb, wrong >= 0.5), "before_recall": recall_score(yb, wrong >= 0.5), "before_f1": f1_score(yb, wrong >= 0.5),
                 "after_precision": precision_score(yb2, right >= 0.5, zero_division=0), "after_recall": recall_score(yb2, right >= 0.5), "after_f1": f1_score(yb2, right >= 0.5)})
numbers["leak_forest_5pct"] = pd.DataFrame(leak).mean().to_dict()

(FIGS / "numbers_balancing.json").write_text(json.dumps(numbers, indent=1, default=float))
show = results.loc[0.05][["precision", "recall", "f1", "ap", "brier", "mean_p"]].round(3)
print(show.to_string())
print(results.xs("none", level=1)[["precision", "recall", "ap"]].round(3).to_string())
print(results.xs("class weights", level=1)[["precision", "recall", "ap"]].round(3).to_string())
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in numbers["deployment"].items()}))
print({k: round(v, 3) for k, v in numbers["leak_forest_5pct"].items()})
