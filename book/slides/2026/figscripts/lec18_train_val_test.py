"""Lecture 18, Part 1 definitions: inputs and targets, training scores and overfitting, row order.

Data: GNSS station P395 (book/Chapter1-GettingStarted/data), and the Pacific Northwest
seismic source tables (Zenodo record 14025693) with the canonical split of notebook 3.5
(25% test, seed 2026, stratified). Writes figures and numbers_train_val_test.json to
figs/lec18_robust_training/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c"
GRAY = "#b8b1a4"
CLASS_COLORS = {"earthquake": "#0072B2", "explosion": "#D55E00", "noise": "#b8b1a4", "surface event": "#009E73"}
SEED, SPLIT_SEED = 42, 2026
ROOT = Path(__file__).resolve().parents[4]
FIGS = Path(__file__).resolve().parent.parent / "figs" / "lec18_robust_training"
FIGS.mkdir(parents=True, exist_ok=True)
numbers = {}


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- One sample of P395: inputs X and target Y ---------------------------------------------------------
gnss = pd.read_csv(ROOT / "book" / "Chapter1-GettingStarted" / "data" / "gps_P395_relative_position.csv")
east, year = gnss["east_m"].to_numpy() * 1000, gnss["decimal_year"].to_numpy()
i = int(np.searchsorted(year, 2016.0))
window = slice(i - 120, i + 75)
fig, ax = plt.subplots(figsize=(11, 6.6))
ax.plot(year[window], east[window], ".", color=GRAY, ms=6)
ax.axvspan(year[i - 60], year[i - 1], color=BLUE, alpha=0.15)
ax.axvspan(year[i], year[i + 29], color=ORANGE, alpha=0.25)
ax.plot(year[i - 60:i], east[i - 60:i], ".", color=BLUE, ms=7)
ax.hlines(east[i:i + 30].mean(), year[i], year[i + 29], color=RED, lw=4)
top = east[window].max()
ax.text(year[i - 30], top + 1.5, "X: six numbers\nfrom the past\n60 days", color=BLUE, ha="center", va="bottom", fontweight="bold", fontsize=18)
ax.text(year[i + 15], top + 1.5, "Y: mean\nof the next\n30 days", color=RED, ha="center", va="bottom", fontweight="bold", fontsize=18)
ax.set_ylim(east[window].min() - 2, top + 12)
ax.set(xlabel="Year", ylabel="East position (mm)")
ax.set_title("One sample of the P395 forecast", loc="left")
fig.tight_layout()
save(fig, "xy_p395_sample")

# --- Seismic source types ------------------------------------------------------------------------------
SEISMIC_FILES = {
    "1000_earthquakes_physical_features.csv": "md5:28129c8dd1b3e14f655d489577b841b5",
    "1000_explosion_physical_features.csv": "md5:af1342d32e163e961e043364136359b0",
    "1000_noise_physical_features.csv": "md5:16cdb992fed6cf6273d5624f5df905da",
    "1000_surface_physical_features.csv": "md5:9a2c2643030cf058704d68e130654e9d",
}
tables = [pd.read_csv(pooch.retrieve(url=f"https://zenodo.org/api/records/14025693/files/{name}/content", known_hash=checksum,
                                     fname=name, path=pooch.os_cache("mlgeo")), index_col=0)
          for name, checksum in SEISMIC_FILES.items()]
seismic = pd.concat(tables, ignore_index=True).dropna(axis=1)
X, y = seismic.drop(columns=["source", "serial_no"]).to_numpy(), seismic["source"].to_numpy()
classes = np.unique(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)


def forest():
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)


# Training scores: a forest, then one tree grown deeper and deeper (overfitting), depth chosen by validation.
fitted = forest().fit(X_train, y_train)
forest_train = f1_score(y_train, fitted.predict(X_train), average="macro")
forest_test = f1_score(y_test, fitted.predict(X_test), average="macro")
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]
folds5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
train_curve, val_curve = [], []
for depth in DEPTHS:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
    scores = cross_validate(tree, X_train, y_train, cv=folds5, scoring="f1_macro", return_train_score=True)
    train_curve.append(scores["train_score"].mean())
    val_curve.append(scores["test_score"].mean())
best = int(np.argmax(val_curve))
chosen = DecisionTreeClassifier(max_depth=DEPTHS[best], random_state=SEED).fit(X_train, y_train)
full = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
numbers["training_scores"] = {"forest_train_f1": forest_train, "forest_test_f1": forest_test,
                              "depths": [str(d) for d in DEPTHS], "tree_train_f1": train_curve, "tree_validation_f1": val_curve,
                              "chosen_depth": DEPTHS[best], "chosen_test_f1": f1_score(y_test, chosen.predict(X_test), average="macro"),
                              "full_depth": int(full.get_depth()), "full_test_f1": f1_score(y_test, full.predict(X_test), average="macro")}

fig, (left, right) = plt.subplots(1, 2, figsize=(21, 7), width_ratios=[1, 1.8])
left.bar([0, 1], [forest_train, forest_test], color=[GRAY, BLUE], width=0.6)
for x, v in zip([0, 1], [forest_train, forest_test]):
    left.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
left.set_xticks([0, 1], ["training\n(3,000 events)", "test\n(1,000 events)"])
left.set(ylim=(0, 1.12), ylabel="macro-F1")
left.set_title("Random forest, 300 trees", loc="left")
x = np.arange(len(DEPTHS))
right.plot(x, train_curve, "o-", color=GRAY, lw=3, ms=9)
right.plot(x, val_curve, "o-", color=ORANGE, lw=3, ms=9)
right.axvline(best, color=INK, ls="--", lw=1.5)
right.text(x[-1], train_curve[-1] - 0.035, "training", color=MUTED, ha="right", va="top", fontweight="bold")
right.text(x[-1], val_curve[-1] - 0.035, "validation (5 folds)", color=ORANGE, ha="right", va="top", fontweight="bold")
right.text(best + 0.2, 0.36, f"chosen by validation: depth {DEPTHS[best]}\ntest macro-F1 {numbers['training_scores']['chosen_test_f1']:.3f}, scored once", va="bottom")
right.annotate("", xy=(x[-3], train_curve[-3] - 0.01), xytext=(x[-3], val_curve[-3] + 0.01),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8))
right.text(x[-3] - 0.15, (train_curve[-3] + val_curve[-3]) / 2, "overfitting:\nthe gap grows", ha="right", va="center", fontsize=18)
right.set_xticks(x, [str(d) if d else "full" for d in DEPTHS])
right.set(xlabel="Maximum depth of one decision tree", ylabel="macro-F1", ylim=(0.3, 1.04))
right.set_title("One tree, deeper and deeper", loc="left")
fig.tight_layout(w_pad=3)
save(fig, "forest_vs_tree_depth")

# Row order: the files are stored by class, so the last 25% of rows is one class.
cut = int(0.75 * len(y))
index_model = forest().fit(X[:cut], y[:cut])
numbers["row_order"] = {"index_test_counts": pd.Series(y[cut:]).value_counts().to_dict(),
                        "index_train_counts": pd.Series(y[:cut]).value_counts().to_dict(),
                        "index_test_accuracy": float((index_model.predict(X[cut:]) == y[cut:]).mean())}
folds = cross_val_score(forest(), X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1_macro")
numbers["row_order"].update({"fold_f1": folds.tolist(), "fold_mean": folds.mean(), "fold_std": folds.std()})

fig, (left, right) = plt.subplots(1, 2, figsize=(21, 7), width_ratios=[1.5, 1])
bars = {"first 75% of rows\n(training)": y[:cut], "last 25% of rows\n(test)": y[cut:],
        "random, by class\n(training)": y_train, "random, by class\n(test)": y_test}
for k, (label, values) in enumerate(bars.items()):
    share = pd.Series(values).value_counts(normalize=True).reindex(classes, fill_value=0)
    bottom = 0
    for name in classes:
        left.bar(k, share[name], bottom=bottom, color=CLASS_COLORS[name], width=0.7, label=name if k == 2 else None)
        bottom += share[name]
left.set_xticks(range(4), list(bars))
left.tick_params(axis="x", labelsize=15)
left.set(ylabel="Share of samples", ylim=(0, 1))
fig.legend(*left.get_legend_handles_labels(), frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.32, -0.08), fontsize=17)
left.set_title("Class mix of each partition", loc="left")
right.plot(np.arange(1, 6), folds, "o", color=BLUE, ms=14)
right.axhline(folds.mean(), color=BLUE, lw=2)
right.text(5.4, folds.mean() - 0.002, f"mean\n{folds.mean():.3f}", color=BLUE, va="top", fontweight="bold")
right.set(xlim=(0.5, 6.6), ylim=(0.85, 0.91), xlabel="Fold", ylabel="Validation macro-F1")
right.set_xticks(range(1, 6))
right.set_title("5-fold cross-validation, forest", loc="left")
fig.tight_layout(w_pad=3)
save(fig, "row_order")

(FIGS / "numbers_train_val_test.json").write_text(json.dumps(numbers, indent=2, default=float))
print(json.dumps(numbers, indent=2, default=float))
