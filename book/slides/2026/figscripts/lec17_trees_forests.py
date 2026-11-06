"""Slide figures and numbers for lecture 17: trees and forests, classification then regression.

Classification: Pacific Northwest seismic source tables (Zenodo record 14025693, 1000 per
class, 61 waveform features), canonical leaderboard split of notebook 3.5 (random_state=2026,
stratified, 25% test). Regression: USCRN station WA Spokane 17 SSW (Turnbull), 2015 to 2024,
course snapshot in book/Chapter2-DataManipulation/snapshots/, train 2015-2020, test 2021-2024.
Writes the figures to figs/lec17_trees_forests/ and every number quoted on a slide to numbers.json.

Maps: CPC soil moisture and PREC/L precipitation (0.5 degree, 1991-2020), course snapshot, western United States.

Regenerate: pixi run python book/slides/2026/figscripts/lec17_trees_forests.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402  (backend must be set first)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import xarray as xr  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,  # noqa: E402
                             r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c"
SEISMIC_FILES = {
    "1000_earthquakes_physical_features.csv": "md5:28129c8dd1b3e14f655d489577b841b5",
    "1000_explosion_physical_features.csv": "md5:af1342d32e163e961e043364136359b0",
    "1000_noise_physical_features.csv": "md5:16cdb992fed6cf6273d5624f5df905da",
    "1000_surface_physical_features.csv": "md5:9a2c2643030cf058704d68e130654e9d",
}
ROOT = Path(__file__).resolve().parents[4]
SNAPSHOTS = ROOT / "book" / "Chapter2-DataManipulation" / "snapshots"
FIGS = Path(__file__).resolve().parent.parent / "figs" / "lec17_trees_forests"
FIGS.mkdir(parents=True, exist_ok=True)
numbers = {}


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


frames = []
for fname, checksum in SEISMIC_FILES.items():
    path = pooch.retrieve(url=f"https://zenodo.org/api/records/14025693/files/{fname}/content",
                          known_hash=checksum, fname=fname, path=pooch.os_cache("mlgeo"))
    frames.append(pd.read_csv(path, index_col=0))
seismic = pd.concat(frames, ignore_index=True).dropna(axis=1)
X = seismic.drop(columns=["source", "serial_no"])
y = seismic["source"]
classes = ["earthquake", "explosion", "noise", "surface event"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2026, stratify=y)
macro = lambda truth, pred: f1_score(truth, pred, average="macro")  # noqa: E731

# --- Binary: same split and model as lecture 16 ------------------------------------------------
two = y.isin(["earthquake", "explosion"])
Xb, yb = X[two], (y[two] == "explosion").astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, random_state=2026, stratify=yb)
binary_logistic = logistic().fit(Xtr, ytr)
binary_forest = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
p_log = binary_logistic.predict_proba(Xte)[:, 1]
p_for = binary_forest.predict_proba(Xte)[:, 1]

# --- Ten real test events, closest to chosen probabilities, so that every kind of error appears ---
targets = [(0.97, 1), (0.90, 1), (0.72, 0), (0.62, 1), (0.41, 0), (0.35, 1), (0.15, 0), (0.12, 1), (0.06, 0), (0.02, 0)]
truth = yte.to_numpy()
chosen = [int(np.argmin(np.abs(p_log - t) + 10 * (truth != label))) for t, label in targets]
ten_p, ten_y = np.round(p_log[chosen], 2), truth[chosen]
numbers["ten_events"] = [{"rank": i + 1, "probability": float(q), "analyst": "explosion" if v else "earthquake"}
                         for i, (q, v) in enumerate(zip(ten_p, ten_y))]


def rates(labels, probabilities, threshold):
    flagged = probabilities >= threshold
    return flagged[labels == 1].mean(), flagged[labels == 0].mean()


numbers["ten_rates"] = {str(t): dict(zip(("recall", "false_positive_rate"), map(float, rates(ten_y, ten_p, t)))) for t in (0.8, 0.5, 0.2)}
numbers["ten_auc"] = roc_auc_score(ten_y, ten_p)

fig, ax = plt.subplots(figsize=(7.4, 7.4))
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.grid(color="#c9c2b6", lw=1.3)
ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=2)
ax.text(0.60, 0.53, "no skill", color=MUTED, rotation=45, ha="center", va="center", fontsize=17)
ax.set(xlabel="False-positive rate", ylabel="Recall", xlim=(0, 1.02), ylim=(0, 1.02))
save(fig, "hand_roc_blank")

fpr_t, tpr_t, _ = roc_curve(ten_y, ten_p)
fpr_l, tpr_l, _ = roc_curve(yte, p_log)
fig, ax = plt.subplots(figsize=(8.2, 7.2))
ax.plot(fpr_l, tpr_l, color=BLUE, lw=3, label=f"all 500 test events: AUC {roc_auc_score(yte, p_log):.3f}")
ax.plot(fpr_t, tpr_t, color=RED, lw=3, drawstyle="default", marker="o", ms=5, label=f"your ten events: AUC {numbers['ten_auc']:.2f}")
for t in (0.8, 0.5, 0.2):
    r, f = rates(ten_y, ten_p, t)
    ax.plot(f, r, "o", ms=16, mfc="white", mec=RED, mew=3, zorder=5)
    ax.annotate(f"{t}", (f, r), xytext=(14, -22), textcoords="offset points", fontsize=19, color=RED, fontweight="bold")
ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=2)
ax.set(xlabel="False-positive rate", ylabel="Recall", xlim=(0, 1.02), ylim=(0, 1.02))
ax.legend(frameon=False, loc="lower right", fontsize=17)
save(fig, "hand_roc_reveal")

# --- Logistic regression and forest: same curve, different thresholds ----------------------------
numbers["binary"] = {}
fig, ax = plt.subplots(figsize=(8.2, 7.2))
fpr_f, tpr_f, _ = roc_curve(yte, p_for)
ax.plot(fpr_l, tpr_l, color=BLUE, lw=3, label=f"logistic regression: AUC {roc_auc_score(yte, p_log):.3f}")
ax.plot(fpr_f, tpr_f, color=GREEN, lw=3, label=f"forest, 300 trees: AUC {roc_auc_score(yte, p_for):.3f}")
for name, pp, color, marker in [("logistic", p_log, BLUE, "o"), ("forest", p_for, GREEN, "s")]:
    rows = []
    for t in (0.2, 0.5, 0.8):
        flagged = pp >= t
        tp, fp = int((flagged & (yte == 1)).sum()), int((flagged & (yte == 0)).sum())
        rows.append({"threshold": t, "caught": tp, "missed": int((yte == 1).sum()) - tp, "false_alarms": fp,
                     "recall": tp / int((yte == 1).sum()), "false_positive_rate": fp / int((yte == 0).sum()), "precision": tp / (tp + fp)})
        ax.plot(fp / int((yte == 0).sum()), tp / int((yte == 1).sum()), marker, ms=15, mfc="white", mec=color, mew=3, zorder=5)
    numbers["binary"][name] = {"thresholds": rows, "roc_auc": roc_auc_score(yte, pp),
                               "average_precision": average_precision_score(yte, pp), "brier": brier_score_loss(yte, pp)}
ax.set(xlabel="False-positive rate", ylabel="Recall", xlim=(0, 0.45), ylim=(0.55, 1.01))
ax.legend(frameon=False, loc="lower right", fontsize=17)
ax.text(0.44, 0.70, "markers: thresholds 0.8, 0.5, 0.2,\nleft to right", color=MUTED, fontsize=16, va="top", ha="right")
save(fig, "logistic_forest_roc")

# --- One tree: questions, boxes, purity, memorization ------------------------------------------------
# Two features a seismologist can name, so the tree can be drawn as lines through the events.
SOURCE_COLORS = {"earthquake": BLUE, "explosion": RED, "noise": MUTED, "surface event": ORANGE}
TWO = ["Fquart1", "KurtoSig"]
small_tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_train, y_train)
numbers["tree_depth2_macro_f1"] = macro(y_test, small_tree.predict(X_test))
three_questions = DecisionTreeClassifier(max_leaf_nodes=4, random_state=42).fit(X_train[TWO], y_train)
full_two = DecisionTreeClassifier(random_state=42).fit(X_train[TWO], y_train)
inner = np.flatnonzero(three_questions.tree_.children_left != -1)
# Nodes in the order the tree asks: spikiness, low frequencies, spikiness again.
q1, q2, q3 = [float(three_questions.tree_.threshold[i]) for i in inner]
numbers["two_feature_tree"] = {"features": TWO, "thresholds": [q1, q2, q3], "leaves_full": int(full_two.get_n_leaves()),
                               "train_three": macro(y_train, three_questions.predict(X_train[TWO])),
                               "test_three": macro(y_test, three_questions.predict(X_test[TWO])),
                               "train_full": macro(y_train, full_two.predict(X_train[TWO])),
                               "test_full": macro(y_test, full_two.predict(X_test[TWO]))}
XLIM, YLIM = (X_train["Fquart1"].quantile(0.002), X_train["Fquart1"].quantile(0.998)), (-1.0, X_train["KurtoSig"].quantile(0.995))
shown = X_train.assign(source=y_train).sample(1200, random_state=42)


def event_map(ax, events=True, size=11):
    ax.set_yscale("symlog", linthresh=2)
    if events:
        for source, color in SOURCE_COLORS.items():
            part = shown[shown["source"] == source]
            ax.scatter(part["Fquart1"], part["KurtoSig"], s=size, color=color, alpha=0.75, lw=0)
    ax.set(xlim=XLIM, ylim=YLIM, xticks=[], yticks=[], xlabel="more low-frequency  \u2190    \u2192  less", ylabel="smoother  \u2190    \u2192  spikier")
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)


def box_labels(ax, fontsize=20):
    ymid = lambda lo, hi: float(np.sqrt(max(lo, 1) * hi))  # noqa: E731  (middle of a box on the stretched axis)
    boxes = [(XLIM[0], XLIM[1], YLIM[0], q1, "noise"), (XLIM[0], q2, q1, YLIM[1], "explosion"),
             (q2, XLIM[1], q1, q3, "surface event"), (q2, XLIM[1], q3, YLIM[1], "earthquake")]
    for x0, x1, y0, y1, source in boxes:
        ax.fill_between([x0, x1], y0, y1, color=SOURCE_COLORS[source], alpha=0.16, lw=0, zorder=0)
        yy = 0.2 if source == "noise" else ymid(y0, y1)
        ax.text((x0 + x1) / 2, yy, source, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=SOURCE_COLORS[source],
                bbox=dict(fc="white", ec="none", alpha=0.85, boxstyle="round,pad=0.2"))


def question_lines(ax, numbered=True):
    for xs, ys, tag, where in [(XLIM, (q1, q1), "1", (XLIM[0] + 0.12, q1)), ((q2, q2), (q1, YLIM[1]), "2", (q2, YLIM[1] * 0.6)),
                               ((q2, XLIM[1]), (q3, q3), "3", (XLIM[1] - 0.12, q3))]:
        ax.plot(xs, ys, color=INK, lw=3)
        if numbered:
            ax.text(*where, tag, ha="center", va="center", fontsize=20, fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.3", fc=INK, ec="none"))


fig, ax = plt.subplots(figsize=(7.4, 6.6))
event_map(ax)
box_labels(ax)
question_lines(ax)
save(fig, "tree_boxes")


def node(ax, xy, text, face, ink=INK, weight="normal", size=19):
    ax.text(*xy, text, ha="center", va="center", fontsize=size, color=ink, fontweight=weight,
            bbox=dict(boxstyle="round,pad=0.5", fc=face, ec="none"))


def question_flow(steps, leaves, edges, name):
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.set(xlim=(-0.2, 10), ylim=(-1.4, 10))
    ax.axis("off")
    for start, end, word in edges:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.2), zorder=0)
        ax.text(start[0] + 0.55 * (end[0] - start[0]) + (-0.5 if end[0] < start[0] else 0.5), start[1] + 0.55 * (end[1] - start[1]), word,
                ha="center", va="center", fontsize=18, color=MUTED, style="italic")
    for xy, text in steps:
        node(ax, xy, text, "#efeae0")
    for xy, text, face, ink in leaves:
        node(ax, xy, text, face, ink=ink, weight="bold")
    save(fig, name)


Q1, Q2, Q3 = (3.4, 9.1), (5.0, 6.2), (6.6, 3.2)
question_flow([(Q1, "1  Is the waveform smooth?"), (Q2, "2  Is its energy mostly\nat low frequencies?"), (Q3, "3  Is it only\nmoderately spiky?")],
              [((1.2, 6.2), "noise", SOURCE_COLORS["noise"], "white"), ((2.9, 3.2), "explosion", SOURCE_COLORS["explosion"], "white"),
               ((4.4, -0.3), "surface\nevent", SOURCE_COLORS["surface event"], "white"), ((8.4, -0.3), "earthquake", SOURCE_COLORS["earthquake"], "white")],
              [(Q1, (1.2, 6.2), "yes"), (Q1, Q2, "no"), (Q2, (2.9, 3.2), "yes"), (Q2, Q3, "no"), (Q3, (4.4, -0.3), "yes"), (Q3, (8.4, -0.3), "no")], "tree_questions")

# Purity: a question that sorts the events, and one that leaves the larger group as mixed as before.
# Bar length is the number of events on that side of the line.
poor_cut = float(X_train["Fquart1"].quantile(0.9))
fig = plt.figure(figsize=(15.6, 6.5))
grid = fig.add_gridspec(2, 2, height_ratios=[4.6, 1.25], hspace=0.3, wspace=0.08)
order = list(SOURCE_COLORS)
for column, (title, side_a, side_b, draw) in enumerate([
        ("A question that changes little", X_train["Fquart1"] <= poor_cut, X_train["Fquart1"] > poor_cut, lambda a: a.plot([poor_cut] * 2, YLIM, color=INK, lw=3)),
        ("The question the tree keeps", X_train["KurtoSig"] <= q1, X_train["KurtoSig"] > q1, lambda a: a.plot(XLIM, [q1] * 2, color=INK, lw=3))]):
    ax = fig.add_subplot(grid[0, column])
    event_map(ax)
    draw(ax)
    ax.set_title(title, fontsize=22)
    if column == 1:
        ax.set_ylabel("")
    bars = fig.add_subplot(grid[1, column])
    for row, mask in enumerate((side_a, side_b)):
        left = 0
        counts = y_train[mask].value_counts()
        for source in order:
            bars.barh(1 - row, counts.get(source, 0), left=left, color=SOURCE_COLORS[source], height=0.76)
            left += counts.get(source, 0)
    bars.set(xlim=(0, len(y_train)), xticks=[], yticks=[1, 0], yticklabels=["one side", "other side"] if column == 0 else [])
    for side in ("left", "bottom"):
        bars.spines[side].set_visible(False)
    bars.tick_params(left=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=SOURCE_COLORS[source]) for source in order]
fig.legend(handles, order, loc="lower center", ncol=4, frameon=False, fontsize=19, bbox_to_anchor=(0.5, -0.06))
save(fig, "tree_purity")

# Memorizing: the same two features, three questions against every question the tree can ask.
xx, yy = np.meshgrid(np.linspace(*XLIM, 700), np.concatenate([np.linspace(YLIM[0], 2, 220), np.geomspace(2, YLIM[1], 480)]))
grid_events = pd.DataFrame({"Fquart1": xx.ravel(), "KurtoSig": yy.ravel()})
fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.7), sharey=True)
for ax, model, title in [(axes[0], three_questions, "Three questions"), (axes[1], full_two, "Asked until every box is pure")]:
    codes = np.searchsorted(sorted(SOURCE_COLORS), model.predict(grid_events)).reshape(xx.shape)
    event_map(ax, events=False)
    ax.pcolormesh(xx, yy, codes, cmap=matplotlib.colors.ListedColormap([SOURCE_COLORS[c] for c in sorted(SOURCE_COLORS)]), vmin=0, vmax=3, alpha=0.75, shading="auto", rasterized=True)
    ax.set_title(title, fontsize=22)
axes[1].set_ylabel("")
box_labels(axes[0], fontsize=16)
for ax in axes:
    ax.xaxis.label.set_size(17)
fig.subplots_adjust(wspace=0.05)
save(fig, "tree_memorize")

depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]
rows = []
for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    rows.append({"depth": depth, "depth_reached": int(tree.get_depth()), "leaves": int(tree.get_n_leaves()),
                 "train": macro(y_train, tree.predict(X_train)), "test": macro(y_test, tree.predict(X_test))})
numbers["tree_by_depth"] = rows
fig, ax = plt.subplots(figsize=(7.2, 8.0))
reached = [r["depth_reached"] for r in rows]
ax.plot(reached, [r["train"] for r in rows], color=MUTED, lw=3, marker="o", ms=9, label="training events")
ax.plot(reached, [r["test"] for r in rows], color=ORANGE, lw=3, marker="s", ms=9, label="test events")
ax.set(xlabel="Depth of the tree", ylabel="Macro-F1", ylim=(0.3, 1.03), xticks=[2, 5, 8, 12, 18])
ax.legend(frameon=False, loc="lower right")
save(fig, "tree_depth_curve")

# --- Many trees ----------------------------------------------------------------------------------------
rows = []
for n in (1, 2, 5, 10, 20, 50, 100, 200, 300, 500):
    forest = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1, oob_score=n >= 50).fit(X_train, y_train)
    rows.append({"trees": n, "test_macro_f1": macro(y_test, forest.predict(X_test)),
                 "oob_macro_f1": macro(y_train, forest.classes_[forest.oob_decision_function_.argmax(axis=1)]) if n >= 50 else None})
numbers["forest_by_trees"] = rows
log_multi = logistic().fit(X_train, y_train)
numbers["logistic_macro_f1"] = macro(y_test, log_multi.predict(X_test))
fig, ax = plt.subplots(figsize=(8.6, 6.4))
ax.plot([r["trees"] for r in rows], [r["test_macro_f1"] for r in rows], color=GREEN, lw=3, marker="o", ms=9, label="forest, test events")
oob = [r for r in rows if r["oob_macro_f1"] is not None]
ax.plot([r["trees"] for r in oob], [r["oob_macro_f1"] for r in oob], color=MUTED, lw=3, marker="s", ms=9, label="forest, out-of-bag events")
ax.axhline(numbers["logistic_macro_f1"], color=BLUE, ls="--", lw=2.5, label="logistic regression, test events")
ax.set_xscale("log")
ax.set_xticks([1, 10, 100, 500], ["1", "10", "100", "500"])
ax.set(xlabel="Number of trees", ylabel="Macro-F1", ylim=(0.74, 0.92))
ax.legend(frameon=False, loc="lower right", fontsize=17)
save(fig, "forest_by_trees")

# --- Four models on the leaderboard split ------------------------------------------------------------------
forest = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X_train, y_train)
deep_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
boost = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)
raw_logistic = LogisticRegression(max_iter=5000).fit(X_train, y_train)
models = {"one deep tree": deep_tree, "forest, 300 trees": forest, "boosted trees": boost, "logistic regression": log_multi}
numbers["models"] = {name: {"macro_f1": macro(y_test, m.predict(X_test)),
                            "recall": {c: float(recall_score(y_test == c, m.predict(X_test) == c)) for c in classes}}
                     for name, m in models.items()}
numbers["logistic_unscaled_macro_f1"] = macro(y_test, raw_logistic.predict(X_test))
numbers["forest_scaled_macro_f1"] = macro(y_test, make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)).fit(X_train, y_train).predict(X_test))

# --- Few training events: AUC of explosion against earthquake, mean of 10 random subsets ----------------------
sizes = [40, 100, 200, 500, 1000, 1500]
rows = []
for size in sizes:
    forest_auc, logistic_auc = [], []
    for seed in range(10):
        pick = np.random.default_rng(seed).choice(len(Xtr), size=size, replace=False)
        sub_X, sub_y = Xtr.iloc[pick], ytr.iloc[pick]
        forest_auc.append(roc_auc_score(yte, RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(sub_X, sub_y).predict_proba(Xte)[:, 1]))
        logistic_auc.append(roc_auc_score(yte, logistic().fit(sub_X, sub_y).predict_proba(Xte)[:, 1]))
    rows.append({"training_events": size, "forest_auc": np.mean(forest_auc), "logistic_auc": np.mean(logistic_auc),
                 "forest_sd": np.std(forest_auc), "logistic_sd": np.std(logistic_auc)})
numbers["few_training_events"] = rows
fig, ax = plt.subplots(figsize=(7.2, 5.8))
ax.errorbar(sizes, [r["forest_auc"] for r in rows], yerr=[r["forest_sd"] for r in rows], color=GREEN, lw=3, marker="o", ms=9, capsize=4, label="forest, 100 trees")
ax.errorbar(sizes, [r["logistic_auc"] for r in rows], yerr=[r["logistic_sd"] for r in rows], color=BLUE, lw=3, marker="s", ms=9, capsize=4, label="logistic regression")
ax.set_xscale("log")
ax.set_xticks([40, 100, 200, 500, 1500], ["40", "100", "200", "500", "1500"])
ax.set(xlabel="Training events", ylabel="ROC AUC, 500 test events")
ax.legend(frameon=False, loc="lower right")
save(fig, "few_training_events")

# --- Tree agreement as a warning: bins of the share of trees voting for the winner -----------------------------
votes = np.stack([tree.predict(X_test.to_numpy()) for tree in forest.estimators_])
agreement = np.stack([(votes == k).mean(axis=0) for k in range(len(forest.classes_))], axis=1).max(axis=1)
wrong = forest.predict(X_test) != y_test.to_numpy()
edges, labels = [0.25, 0.5, 0.7, 0.9, 1.001], ["below 50%", "50 to 70%", "70 to 90%", "above 90%"]
index = np.digitize(agreement, edges) - 1
numbers["agreement"] = [{"bin": labels[b], "events": int((index == b).sum()), "wrong": float(wrong[index == b].mean())} for b in range(4)]
fig, ax = plt.subplots(figsize=(8.2, 5.6))
ax.bar(labels, [100 * row["wrong"] for row in numbers["agreement"]], color=GREEN, width=0.7)
for i, row in enumerate(numbers["agreement"]):
    ax.text(i, 100 * row["wrong"] + 1, f"{100 * row['wrong']:.0f}%\n{row['events']} events", ha="center", va="bottom", fontsize=17)
ax.set(xlabel="Share of trees voting for the winning class", ylabel="Wrong (%)", ylim=(0, 70))
save(fig, "tree_agreement")

# --- Regression: soil moisture at 5 cm, Turnbull station -----------------------------------------------------
cols = (SNAPSHOTS / "uscrn_daily01_HEADERS.txt").read_text().splitlines()[1].split()
station = pd.read_csv(SNAPSHOTS / "uscrn_WA_Spokane_17_SSW_2015-2024.txt", sep=r"\s+", header=None, names=cols,
                      na_values=[-9999, -99, -9999.0, -99.0, -99.000])
station["date"] = pd.to_datetime(station["LST_DATE"].astype(str))
station = station.set_index("date").asfreq("D")
soil = pd.DataFrame({
    "moisture": station["SOIL_MOISTURE_5_DAILY"],
    "moisture_yesterday": station["SOIL_MOISTURE_5_DAILY"].shift(1),
    "moisture_10cm_yesterday": station["SOIL_MOISTURE_10_DAILY"].shift(1),
    "rain_today": station["P_DAILY_CALC"],
    "rain_yesterday": station["P_DAILY_CALC"].shift(1),
    "air_temperature": station["T_DAILY_MEAN"],
    "solar": station["SOLARAD_DAILY"],
    "day_of_year": station.index.dayofyear,
}).dropna()
FEATURES = ["moisture_yesterday", "moisture_10cm_yesterday", "rain_today", "rain_yesterday", "air_temperature", "solar"]
train, test = soil[:"2020-12-31"], soil["2021-01-01":]
climatology = train.groupby("day_of_year")["moisture"].mean().reindex(range(1, 367)).interpolate().rolling(15, center=True, min_periods=1).mean()
reg_forest = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(train[FEATURES], train["moisture"])
reg_linear = LinearRegression().fit(train[FEATURES], train["moisture"])
predictions = {"climatology": test["day_of_year"].map(climatology).to_numpy(), "yesterday's value": test["moisture_yesterday"].to_numpy(),
               "linear regression": reg_linear.predict(test[FEATURES]), "forest, 300 trees": reg_forest.predict(test[FEATURES])}
rainy = test["rain_today"].to_numpy() >= 5
numbers["soil"] = {"train_days": len(train), "test_days": len(test), "rainy_test_days": int(rainy.sum()),
                   "models": {name: {"mae": mean_absolute_error(test["moisture"], p), "r2": r2_score(test["moisture"], p),
                                     "mae_rainy": mean_absolute_error(test["moisture"][rainy], p[rainy]),
                                     "mae_dry": mean_absolute_error(test["moisture"][~rainy], p[~rainy])} for name, p in predictions.items()}}

window = test["2022-03-01":"2022-07-15"]
fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.6, 7.4), sharex=True, height_ratios=[3, 1])
top.plot(window.index, window["moisture"], color=INK, lw=3, label="observed")
top.plot(window.index, reg_forest.predict(window[FEATURES]), color=GREEN, lw=2.5, label="forest")
top.plot(window.index, window["day_of_year"].map(climatology), color=ORANGE, lw=2.5, label="climatology")
top.set(ylabel="Soil moisture, 5 cm\n(m³/m³)")
top.legend(frameon=False, ncol=3, loc="lower left", fontsize=17)
bottom.bar(window.index, window["rain_today"], color=BLUE, width=1)
bottom.set(ylabel="Rain\n(mm)")
bottom.xaxis.set_major_locator(mdates.MonthLocator())
bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
bottom.tick_params(axis="x", labelsize=14)
fig.align_ylabels()
save(fig, "soil_regression")

# --- Importance on real data: add yesterday's moisture again, in percent ---------------------------------------
importance = dict(zip(FEATURES, map(float, reg_forest.feature_importances_)))
perm = permutation_importance(reg_forest, test[FEATURES], test["moisture"], n_repeats=10, random_state=0, scoring="neg_mean_absolute_error")
twin_train = train[FEATURES].assign(moisture_yesterday_percent=100 * train["moisture_yesterday"])
twin_test = test[FEATURES].assign(moisture_yesterday_percent=100 * test["moisture_yesterday"])
twin_forest = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(twin_train, train["moisture"])
twin_importance = dict(zip(twin_train.columns, map(float, twin_forest.feature_importances_)))
numbers["importance"] = {"impurity": importance, "permutation_increase_in_mae": dict(zip(FEATURES, map(float, perm.importances_mean))),
                         "with_copy": twin_importance, "mae_without_copy": mean_absolute_error(test["moisture"], reg_forest.predict(test[FEATURES])),
                         "mae_with_copy": mean_absolute_error(test["moisture"], twin_forest.predict(twin_test))}
fig, ax = plt.subplots(figsize=(7.4, 4.8))
names = ["moisture yesterday", "moisture yesterday,\nin percent (a copy)", "all other features"]
without = [importance["moisture_yesterday"], 0, 1 - importance["moisture_yesterday"]]
with_copy = [twin_importance["moisture_yesterday"], twin_importance["moisture_yesterday_percent"],
             1 - twin_importance["moisture_yesterday"] - twin_importance["moisture_yesterday_percent"]]
y_pos = np.arange(3)
ax.barh(y_pos - 0.2, without, height=0.38, color=MUTED, label="original features")
ax.barh(y_pos + 0.2, with_copy, height=0.38, color=GREEN, label="with the copy added")
for i, (a, b) in enumerate(zip(without, with_copy)):
    ax.text(a + 0.012, i - 0.2, f"{a:.2f}", va="center", fontsize=17, color=MUTED)
    ax.text(b + 0.012, i + 0.2, f"{b:.2f}", va="center", fontsize=17, color=GREEN)
ax.set_yticks(y_pos, names)
ax.invert_yaxis()
ax.set(xlabel="Impurity importance (share of the total)", xlim=(0, 1.12))
ax.legend(frameon=False, loc="lower right", fontsize=17)
save(fig, "importance_copy")

# --- Extrapolation: ground surface temperature from air temperature, trained below a cut -----------------------
CUT = 28.0
surface = pd.DataFrame({"air": station["T_DAILY_MAX"], "ground": station["SUR_TEMP_DAILY_MAX"]}).dropna()
cool, hot = surface[surface["air"] < CUT], surface[surface["air"] >= CUT]
hold = cool.sample(frac=0.25, random_state=42)
fit = cool.drop(hold.index)
ex_forest = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(fit[["air"]], fit["ground"])
ex_linear = LinearRegression().fit(fit[["air"]], fit["ground"])
numbers["extrapolation"] = {"cut_air_c": CUT, "fit_days": len(fit), "held_out_cool_days": len(hold), "hot_days": len(hot),
                            "largest_ground_in_training": float(fit["ground"].max()), "largest_ground_on_hot_days": float(hot["ground"].max()),
                            "largest_forest_prediction_on_hot_days": float(ex_forest.predict(hot[["air"]]).max()),
                            "largest_linear_prediction_on_hot_days": float(ex_linear.predict(hot[["air"]]).max()),
                            "forest": {"mae_cool": mean_absolute_error(hold["ground"], ex_forest.predict(hold[["air"]])),
                                       "mae_hot": mean_absolute_error(hot["ground"], ex_forest.predict(hot[["air"]]))},
                            "linear": {"mae_cool": mean_absolute_error(hold["ground"], ex_linear.predict(hold[["air"]])),
                                       "mae_hot": mean_absolute_error(hot["ground"], ex_linear.predict(hot[["air"]]))}}
fig, ax = plt.subplots(figsize=(9.4, 6.4))
ax.scatter(fit["air"], fit["ground"], s=8, color="#b8b1a4", alpha=0.6, label="training days")
ax.scatter(hot["air"], hot["ground"], s=14, color=ORANGE, alpha=0.65, label="hot days, not used in training")
grid = pd.DataFrame({"air": np.linspace(surface["air"].min(), surface["air"].max(), 500)})
ax.plot(grid["air"], ex_forest.predict(grid), color=GREEN, lw=3, label="forest")
ax.plot(grid["air"], ex_linear.predict(grid), color=BLUE, lw=3, label="linear regression")
ax.axvline(CUT, color=INK, ls="--", lw=2)
ax.set(xlabel="Daily maximum air temperature (°C)", ylabel="Daily maximum ground surface\ntemperature (°C)")
ax.legend(frameon=False, loc="upper left", fontsize=17, markerscale=2.5)
save(fig, "extrapolation")

# --- The same tree for a number: daily change in soil moisture at Turnbull ------------------------------
# Two features with a physical reading, so the tree can again be drawn as lines through the days.
CHANGE = ["rain_today", "moisture_yesterday"]
change_train, change_test = train["moisture"] - train["moisture_yesterday"], test["moisture"] - test["moisture_yesterday"]
four_boxes = DecisionTreeRegressor(max_leaf_nodes=4, random_state=42).fit(train[CHANGE], change_train)
split_nodes = np.flatnonzero(four_boxes.tree_.children_left != -1)
assert [CHANGE[four_boxes.tree_.feature[i]] for i in split_nodes] == ["rain_today", "rain_today", "moisture_yesterday"]
rain_some, rain_lots, wet_soil = [float(four_boxes.tree_.threshold[i]) for i in split_nodes]
leaf_mean = lambda rain, moisture: float(four_boxes.predict(pd.DataFrame({"rain_today": [rain], "moisture_yesterday": [moisture]}))[0])  # noqa: E731
MEANS = {"dry": leaf_mean(0, 0.2), "downpour": leaf_mean(rain_lots + 1, 0.2),
         "moderate_dry_soil": leaf_mean(rain_some + 1, wet_soil - 0.01), "moderate_wet_soil": leaf_mean(rain_some + 1, wet_soil + 0.01)}
leaf_days = pd.Series(four_boxes.apply(train[CHANGE])).value_counts().to_dict()
change_r2 = lambda model: r2_score(change_test, model.predict(test[CHANGE]))  # noqa: E731
numbers["regression_tree"] = {"target": "daily change in 5 cm soil moisture, m3/m3", "cuts": {"rain_mm": [rain_some, rain_lots], "moisture_yesterday": wet_soil},
                              "leaf_means": MEANS, "days_per_leaf": sorted(int(v) for v in leaf_days.values()), "test_r2_four_boxes": change_r2(four_boxes)}
full_change = DecisionTreeRegressor(random_state=42).fit(train[CHANGE], change_train)
numbers["regression_tree"].update(leaves_full=int(full_change.get_n_leaves()), train_r2_full=r2_score(change_train, full_change.predict(train[CHANGE])), test_r2_full=change_r2(full_change))
numbers["regression_tree"]["test_r2_forest_by_leaf_size"] = {
    str(size): change_r2(RandomForestRegressor(n_estimators=300, min_samples_leaf=size, random_state=42, n_jobs=-1).fit(train[CHANGE], change_train)) for size in (1, 5, 20, 50)}

CHANGE_CMAP = plt.get_cmap("BrBG")
CHANGE_NORM = matplotlib.colors.TwoSlopeNorm(vmin=-0.02, vcenter=0, vmax=0.05)
signed = lambda value: f"{value:+.3f}".replace("0.", ".").replace("-", "\u2212")  # noqa: E731
RAIN_MAX, MOIST = float(train["rain_today"].max()), (0.0, float(train["moisture_yesterday"].max()) + 0.01)
fig, ax = plt.subplots(figsize=(7.4, 6.6))
ax.set_xscale("function", functions=(np.sqrt, np.square))
ax.scatter(train["rain_today"], train["moisture_yesterday"], c=change_train, cmap=CHANGE_CMAP, norm=CHANGE_NORM, s=16, lw=0.2, edgecolor="#8a8378")
for x0, x1, y0, y1, key, ytext in [(0, rain_some, *MOIST, "dry", 0.33), (rain_lots, RAIN_MAX * 1.05, *MOIST, "downpour", 0.33),
                                   (rain_some, rain_lots, MOIST[0], wet_soil, "moderate_dry_soil", 0.1), (rain_some, rain_lots, wet_soil, MOIST[1], "moderate_wet_soil", 0.33)]:
    ax.fill_between([x0, x1], y0, y1, color=CHANGE_CMAP(CHANGE_NORM(MEANS[key])), alpha=0.28, lw=0, zorder=0)
    xtext = ((np.sqrt(x0) + np.sqrt(x1)) / 2) ** 2
    ax.text(xtext, ytext, signed(MEANS[key]), ha="center", va="center", fontsize=20, fontweight="bold", color=INK, bbox=dict(fc="white", ec="none", alpha=0.85, boxstyle="round,pad=0.2"))
for xs, ys, tag, where in [((rain_some, rain_some), MOIST, "1", (rain_some, 0.2)), ((rain_lots, rain_lots), MOIST, "2", (rain_lots, 0.2)),
                           ((rain_some, rain_lots), (wet_soil, wet_soil), "3", (((np.sqrt(rain_some) + np.sqrt(rain_lots)) / 2) ** 2, wet_soil))]:
    ax.plot(xs, ys, color=INK, lw=3)
    ax.text(*where, tag, ha="center", va="center", fontsize=20, fontweight="bold", color="white", bbox=dict(boxstyle="circle,pad=0.3", fc=INK, ec="none"))
ax.set(xlim=(0, RAIN_MAX * 1.05), ylim=MOIST, xticks=[], yticks=[], xlabel="no rain today  \u2190    \u2192  more rain", ylabel="drier soil yesterday  \u2190    \u2192  wetter")
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
for side in ("top", "right"):
    ax.spines[side].set_visible(True)
save(fig, "regression_boxes")


def leaf(value, words):
    face = CHANGE_CMAP(CHANGE_NORM(value))
    return f"{words}\n{signed(value)}", face, "white" if abs(CHANGE_NORM(value) - 0.5) > 0.3 else INK


question_flow([(Q1, "1  Did it rain much?"), (Q2, "2  A downpour?"), (Q3, "3  Was the soil\nalready wet?")],
              [((1.0, 6.2), *leaf(MEANS["dry"], "slow drying")), ((2.9, 3.2), *leaf(MEANS["downpour"], "large rise")),
               ((4.4, -0.3), *leaf(MEANS["moderate_wet_soil"], "small rise")), ((8.4, -0.3), *leaf(MEANS["moderate_dry_soil"], "larger rise"))],
              [(Q1, (1.0, 6.2), "no"), (Q1, Q2, "yes"), (Q2, (2.9, 3.2), "yes"), (Q2, Q3, "no"), (Q3, (4.4, -0.3), "yes"), (Q3, (8.4, -0.3), "no")], "regression_questions")

# Stairs: one feature, so the prediction itself can be drawn.
past, future = surface[:"2020-12-31"], surface["2021-01-01":]
line_x = pd.DataFrame({"air": np.linspace(past["air"].min(), past["air"].max(), 1200)})
stair_models = [("One question", DecisionTreeRegressor(max_depth=1, random_state=42)), ("Three levels", DecisionTreeRegressor(max_depth=3, random_state=42)),
                ("No limit", DecisionTreeRegressor(random_state=42)), ("300 trees", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))]
fig, axes = plt.subplots(1, 4, figsize=(17, 7), sharey=True)
numbers["stairs_test_mae_c"] = {}
for ax, (title, model) in zip(axes, stair_models):
    model.fit(past[["air"]], past["ground"])
    ax.scatter(past["air"], past["ground"], s=5, color="#b8b1a4", alpha=0.45, lw=0)
    ax.plot(line_x["air"], model.predict(line_x), color=GREEN if "trees" in title else RED, lw=2.4, drawstyle="steps-post" if "trees" not in title else "default")
    ax.set(title=title, xticks=[], yticks=[], xlabel="air temperature  \u2192")
    numbers["stairs_test_mae_c"][title] = mean_absolute_error(future["ground"], model.predict(future[["air"]]))
axes[0].set_ylabel("ground temperature  \u2192")
fig.subplots_adjust(wspace=0.04)
save(fig, "regression_stairs")

# --- Forests on a map: annual soil moisture, western United States, CPC grid ----------------------------
# Leaky-bucket soil moisture and the precipitation that drives it (Fan and van den Dool, 2004), 1991-2020 means.
GRID = xr.open_dataset(SNAPSHOTS / "cpc_soilw_precl_westUS_1991-2020.nc")
GRID_STATIC = xr.open_dataset(SNAPSHOTS / "cpc_westUS_static_1991-2020.nc")
monthly = GRID.groupby("time.month").mean("time")
lon2d, lat2d = np.meshgrid(GRID["lon"].values - 360, GRID["lat"].values)
cells = pd.DataFrame({"soil": GRID["soilw"].mean("time").values.ravel(), "lon": lon2d.ravel(), "lat": lat2d.ravel(),
                      "precip": GRID["precip"].mean("time").values.ravel(),
                      "precip_winter": monthly["precip"].sel(month=[12, 1, 2]).mean("month").values.ravel(),
                      "precip_summer": monthly["precip"].sel(month=[6, 7, 8]).mean("month").values.ravel(),
                      "air": GRID_STATIC["air_clim"].mean("month").values.ravel(), "air_july": GRID_STATIC["air_clim"].sel(month=7).values.ravel(),
                      "air_january": GRID_STATIC["air_clim"].sel(month=1).values.ravel()})
map_shape = lon2d.shape
ocean = cells["soil"].isna().to_numpy()
cells = cells.dropna().reset_index(drop=True)
FEATURE_SETS = {"coordinates": ["lon", "lat"], "climate": ["precip", "precip_winter", "precip_summer", "air", "air_july", "air_january"]}
target = cells["soil"].to_numpy()


def map_forest(columns, trees=300, seed=42, n_jobs=-1):
    return RandomForestRegressor(n_estimators=trees, max_features=1.0 if len(columns) == 2 else 1 / 3, random_state=seed, n_jobs=n_jobs)


lat_r, lon_r = np.radians(cells["lat"].to_numpy()), np.radians(cells["lon"].to_numpy())
distance_km = 2 * 6371 * np.arcsin(np.sqrt(np.sin((lat_r[:, None] - lat_r) / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r) * np.sin((lon_r[:, None] - lon_r) / 2) ** 2))

# Random folds, and out of bag: neighbors on both sides of the line.
numbers["map"] = {"cells": len(cells), "random_10fold_r2": {}, "out_of_bag_r2": {}}
for name, columns in FEATURE_SETS.items():
    predicted = np.zeros(len(cells))
    for fold_train, fold_test in KFold(10, shuffle=True, random_state=42).split(cells):
        predicted[fold_test] = map_forest(columns).fit(cells.loc[fold_train, columns], target[fold_train]).predict(cells.loc[fold_test, columns])
    numbers["map"]["random_10fold_r2"][name] = r2_score(target, predicted)
    numbers["map"]["out_of_bag_r2"][name] = map_forest(columns).set_params(oob_score=True).fit(cells[columns], target).oob_score_

# Buffered leave-one-out: drop every training cell closer than the radius (as Ploton et al., 2020, Fig. 5).
RADII = [0, 100, 200, 300, 500, 750, 1000]
tested = np.random.default_rng(42).choice(len(cells), 250, replace=False)


def buffered(i, radius, columns):
    keep = np.flatnonzero((distance_km[i] > radius) & (np.arange(len(cells)) != i))
    return map_forest(columns, trees=80, seed=int(i), n_jobs=1).fit(cells.loc[keep, columns].to_numpy(), target[keep]).predict(cells.loc[[i], columns].to_numpy())[0]


buffer_r2 = {name: [r2_score(target[tested], Parallel(n_jobs=-1)(delayed(buffered)(i, radius, columns) for i in tested)) for radius in RADII] for name, columns in FEATURE_SETS.items()}
numbers["map"]["buffer_radii_km"], numbers["map"]["buffer_r2"] = RADII, buffer_r2

# How far the field itself stays correlated: mean product of standardized values in distance bins.
BINS = [0, 100, 200, 300, 400, 600, 800, 1000, 1400]
standard = (target - target.mean()) / target.std()
upper = np.triu_indices(len(cells), 1)
pairs_d, pairs_z = distance_km[upper], standard[upper[0]] * standard[upper[1]]
bin_mid = [(a + b) / 2 for a, b in zip(BINS[:-1], BINS[1:])]
correlation = [float(pairs_z[(pairs_d > a) & (pairs_d <= b)].mean()) for a, b in zip(BINS[:-1], BINS[1:])]
numbers["map"]["correlogram"] = {"bin_mid_km": bin_mid, "correlation": correlation}
fig, ax = plt.subplots(figsize=(9.6, 5.8))
ax.plot(RADII, buffer_r2["climate"], color=GREEN, lw=3.5, marker="o", ms=10, label="forest on climate")
ax.plot(RADII, buffer_r2["coordinates"], color=RED, lw=3.5, marker="s", ms=10, label="forest on coordinates")
ax.plot(bin_mid, correlation, color=MUTED, lw=2.5, ls="--", marker=".", ms=10, label="correlation of the field itself")
ax.axhline(0, color=INK, lw=1)
ax.set(xlabel="Distance from the nearest training cell (km)", ylabel="R\u00b2 of the prediction", ylim=(-0.7, 1.02), xlim=(-30, 1050))
ax.legend(frameon=False, loc="lower left", fontsize=17)
save(fig, "map_distance")

# One region held out: the same comparison as a picture.
BOX = (-122, -114, 40, 46)
in_box = ((cells["lon"] > BOX[0]) & (cells["lon"] < BOX[1]) & (cells["lat"] > BOX[2]) & (cells["lat"] < BOX[3])).to_numpy()
fills = {}
numbers["map"]["held_out_region"] = {"box": BOX, "cells": int(in_box.sum())}
for name, columns in FEATURE_SETS.items():
    model = map_forest(columns).fit(cells.loc[~in_box, columns], target[~in_box])
    fill = target.copy()
    fill[in_box] = model.predict(cells.loc[in_box, columns])
    fills[name] = fill
    numbers["map"]["held_out_region"][name] = r2_score(target[in_box], fill[in_box])
numbers["map"]["other_regions_r2"] = {}
for other in [(-121, -113, 41, 47), (-124, -118, 41, 47), (-115, -109, 37, 43), (-120, -112, 36, 42)]:
    mask = ((cells["lon"] > other[0]) & (cells["lon"] < other[1]) & (cells["lat"] > other[2]) & (cells["lat"] < other[3])).to_numpy()
    numbers["map"]["other_regions_r2"][str(other)] = {name: r2_score(target[mask], map_forest(columns).fit(cells.loc[~mask, columns], target[~mask]).predict(cells.loc[mask, columns])) for name, columns in FEATURE_SETS.items()}
fig, axes = plt.subplots(1, 3, figsize=(13.4, 6.6), sharey=True)
image = None
for ax, values, title in [(axes[0], target, "CPC field"), (axes[1], fills["coordinates"], "On coordinates"), (axes[2], fills["climate"], "On climate")]:
    field = np.full(map_shape[0] * map_shape[1], np.nan)
    field[~ocean] = values
    image = ax.pcolormesh(lon2d, lat2d, field.reshape(map_shape), cmap="YlGnBu", vmin=0, vmax=600, shading="nearest", rasterized=True)
    ax.add_patch(plt.Rectangle((BOX[0], BOX[2]), BOX[1] - BOX[0], BOX[3] - BOX[2], fill=False, ec=RED, lw=3.5))
    ax.set(title=title, xlabel="Longitude", aspect=1.25, facecolor="#e8e4dc", xticks=[-124, -118, -112])
axes[0].set_ylabel("Latitude")
fig.colorbar(image, ax=axes, shrink=0.86, pad=0.012, label="Annual mean soil moisture (mm)")
save(fig, "map_held_out_region")

# --- Forest settings: 5-fold on the training set only --------------------------------------------------
numbers["settings_classifier_cv_macro_f1"] = {}
folds = StratifiedKFold(5, shuffle=True, random_state=1)
for label, settings in [("default", {}), ("50 trees", {"n_estimators": 50}), ("1000 trees", {"n_estimators": 1000}), ("max_features log2", {"max_features": "log2"}),
                        ("max_features 0.5", {"max_features": 0.5}), ("all features", {"max_features": 1.0}), ("min leaf 5", {"min_samples_leaf": 5}),
                        ("min leaf 20", {"min_samples_leaf": 20}), ("depth 8", {"max_depth": 8}), ("depth 3", {"max_depth": 3})]:
    scores = cross_val_score(RandomForestClassifier(**{"n_estimators": 300, "random_state": 42, "n_jobs": -1, **settings}), X_train, y_train, cv=folds, scoring="f1_macro")
    numbers["settings_classifier_cv_macro_f1"][label] = {"mean": scores.mean(), "std": scores.std()}
numbers["settings_regressor_cv_mae"] = {}
for label, settings in [("default, all features", {}), ("one third of features", {"max_features": 1 / 3})]:
    scores = -cross_val_score(RandomForestRegressor(**{"n_estimators": 300, "random_state": 42, "n_jobs": -1, **settings}), train[FEATURES], train["moisture"], cv=TimeSeriesSplit(5), scoring="neg_mean_absolute_error")
    numbers["settings_regressor_cv_mae"][label] = {"mean": scores.mean(), "std": scores.std()}

(FIGS / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
print(json.dumps(numbers, indent=0, default=lambda v: round(float(v), 3)))
