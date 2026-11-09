"""Slide figures and numbers for lecture 18: robust training, ensembles, automated search.

Time and search: GNSS station P395, daily east position, NGL (course file book/Chapter1-GettingStarted/data/
gps_P395_relative_position.csv); 60 days of history predict the mean of the next 30 days, one sample every 5 days.
Region: CPC soil moisture and PREC/L precipitation (0.5 degree, 1991-2020 means), western United States, course snapshot.
Event: a small synthetic ground-motion table with a planted per-event term, same generator as notebook 3.8.
Ensembles: Pacific Northwest seismic source tables (Zenodo record 14025693), canonical split of notebook 3.5, and
USCRN station WA Spokane 17 SSW (2015 to 2024, course snapshot), air and ground temperature, train to 2021.
Settings match notebooks 3.8, 3.9 and 3.10, which print the same numbers.
Writes the figures to figs/lec18_robust_training/ and every number quoted on a slide to numbers.json.

Regenerate: pixi run python book/slides/2026/figscripts/lec18_scores_ensembles_automl.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import xarray as xr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import f1_score, mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit, cross_val_predict, cross_val_score, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c"
GRAY = "#b8b1a4"
SEISMIC_FILES = {
    "1000_earthquakes_physical_features.csv": "md5:28129c8dd1b3e14f655d489577b841b5",
    "1000_explosion_physical_features.csv": "md5:af1342d32e163e961e043364136359b0",
    "1000_noise_physical_features.csv": "md5:16cdb992fed6cf6273d5624f5df905da",
    "1000_surface_physical_features.csv": "md5:9a2c2643030cf058704d68e130654e9d",
}
SEED, SPLIT_SEED = 42, 2026
HISTORY, HORIZON, STRIDE = 60, 30, 5
GAP = HORIZON // STRIDE   # samples between training and validation, so target windows do not overlap
ROOT = Path(__file__).resolve().parents[4]
SNAPSHOTS = ROOT / "book" / "Chapter2-DataManipulation" / "snapshots"
FIGS = Path(__file__).resolve().parent.parent / "figs" / "lec18_robust_training"
FIGS.mkdir(parents=True, exist_ok=True)
numbers = {}


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def mae_cv(model, X, y, cv):
    return -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")


# --- Time: P395 ---------------------------------------------------------------------------------------
gnss = pd.read_csv(ROOT / "book" / "Chapter1-GettingStarted" / "data" / "gps_P395_relative_position.csv")
east = gnss["east_m"].to_numpy() * 1000
year = gnss["decimal_year"].to_numpy()
rows = []
for i in range(HISTORY, len(east) - HORIZON, STRIDE):
    past = east[i - HISTORY:i]
    rows.append({"mean_60d": past.mean(), "mean_10d": east[i - 10:i].mean(), "last": east[i - 1],
                 "slope_60d": np.polyfit(np.arange(HISTORY), past, 1)[0],
                 "season_sin": np.sin(2 * np.pi * year[i]), "season_cos": np.cos(2 * np.pi * year[i]),
                 "target": east[i:i + HORIZON].mean(), "year": year[i]})
samples = pd.DataFrame(rows)
FEATURES = ["mean_60d", "mean_10d", "last", "slope_60d", "season_sin", "season_cos"]
X_time, y_time = samples[FEATURES].to_numpy(), samples["target"].to_numpy()
last_value = samples["last"].to_numpy()
forest_time = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
shuffled, future = KFold(5, shuffle=True, random_state=SEED), TimeSeriesSplit(5, gap=GAP)
shuffled_scores, future_scores = mae_cv(forest_time, X_time, y_time, shuffled), mae_cv(forest_time, X_time, y_time, future)
persistence = [mean_absolute_error(y_time[va], last_value[va]) for _, va in future.split(X_time)]
line_scores = mae_cv(LinearRegression(), X_time, y_time, future)
numbers["time"] = {"samples": len(samples), "first_year": float(year[0]), "last_year": float(year[-1]),
                   "forest_shuffled_mae": shuffled_scores.mean(), "forest_future_mae": future_scores.mean(), "forest_future_std": future_scores.std(),
                   "persistence_future_mae": float(np.mean(persistence)), "linear_future_mae": line_scores.mean(),
                   "forest_shuffled_r2": cross_val_score(forest_time, X_time, y_time, cv=shuffled).mean(),
                   "forest_future_r2": cross_val_score(forest_time, X_time, y_time, cv=future).mean()}

tr_s, va_s = list(shuffled.split(X_time))[0]
tr_f, va_f = list(future.split(X_time))[-1]
fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
for ax, (tr, va), title in [(axes[0], (tr_s, va_s), f"Shuffled: validation days sit between training days. Forest error {shuffled_scores.mean():.2f} mm"),
                            (axes[1], (tr_f, va_f), f"Train on the past, validate on the future. Forest error {future_scores.mean():.1f} mm")]:
    predicted = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1).fit(X_time[tr], y_time[tr]).predict(X_time[va])
    ax.scatter(samples["year"].iloc[tr], y_time[tr], s=9, color=GRAY, label="training samples")
    ax.scatter(samples["year"].iloc[va], y_time[va], s=16, color=BLUE, label="validation samples, observed")
    ax.scatter(samples["year"].iloc[va], predicted, s=16, color=ORANGE, marker="x", label="forest prediction")
    ax.set_title(title, loc="left", fontsize=21)
    ax.set_ylabel("East (mm)")
axes[0].legend(frameon=False, loc="lower left", markerscale=1.8, fontsize=17)
axes[1].set_xticks(range(2006, 2028, 4))
axes[1].set_xlabel("Year")
fig.tight_layout(h_pad=0.8)
save(fig, "p395_two_splits")

fit_last, val_last = list(future.split(X_time))[-1]
forest_last = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1).fit(X_time[fit_last], y_time[fit_last]).predict(X_time[val_last])
line_last = LinearRegression().fit(X_time[fit_last], y_time[fit_last]).predict(X_time[val_last])
years_last = samples["year"].iloc[val_last]
fig, ax = plt.subplots(figsize=(10, 7.6))
ax.plot(years_last, forest_last, color=ORANGE, lw=3)
ax.plot(years_last, last_value[val_last], color=GREEN, lw=2.2)
ax.plot(years_last, line_last, color=RED, lw=2.2)
ax.plot(years_last, y_time[val_last], color=BLUE, lw=4, alpha=0.45)
ax.text(years_last.iloc[-1], forest_last[-1] + 2, "Forest", color=ORANGE, ha="right", fontweight="bold")
ax.text(years_last.iloc[len(years_last) // 2], -113, "Observed (blue), persistence (green)\nand linear regression (red) overlap", color=MUTED, fontsize=18)
ax.set(xlabel="Year", ylabel="East (mm)")
ax.set_title("Last validation fold", loc="left")
fig.tight_layout()
save(fig, "p395_baselines")

# --- Region: CPC soil moisture ------------------------------------------------------------------------
GRID = xr.open_dataset(SNAPSHOTS / "cpc_soilw_precl_westUS_1991-2020.nc")
STATIC = xr.open_dataset(SNAPSHOTS / "cpc_westUS_static_1991-2020.nc")
monthly = GRID.groupby("time.month").mean("time")
lon2d, lat2d = np.meshgrid(GRID["lon"].values - 360, GRID["lat"].values)
cells = pd.DataFrame({"soil": GRID["soilw"].mean("time").values.ravel(), "lon": lon2d.ravel(), "lat": lat2d.ravel(),
                      "precip": GRID["precip"].mean("time").values.ravel(),
                      "precip_winter": monthly["precip"].sel(month=[12, 1, 2]).mean("month").values.ravel(),
                      "precip_summer": monthly["precip"].sel(month=[6, 7, 8]).mean("month").values.ravel(),
                      "air": STATIC["air_clim"].mean("month").values.ravel(), "air_july": STATIC["air_clim"].sel(month=7).values.ravel(),
                      "air_january": STATIC["air_clim"].sel(month=1).values.ravel()}).dropna().reset_index(drop=True)
COORDINATES, CLIMATE = ["lon", "lat"], ["precip", "precip_winter", "precip_summer", "air", "air_july", "air_january"]
soil = cells["soil"].to_numpy()


def block_labels(table, size):
    return np.floor(table["lon"] / size) * 1000 + np.floor(table["lat"] / size)


def pooled_r2(columns, cv, groups=None):
    forest = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    return r2_score(soil, cross_val_predict(forest, cells[columns], soil, cv=cv, groups=groups))


SIZES = [2, 4, 6]
region = {"cells": len(cells), "random": {}, "blocks": {}}
for name, columns in [("coordinates", COORDINATES), ("climate", CLIMATE)]:
    region["random"][name] = pooled_r2(columns, KFold(5, shuffle=True, random_state=SEED))
    region["blocks"][name] = {size: pooled_r2(columns, GroupKFold(5), block_labels(cells, size)) for size in SIZES}
region["block_counts"] = {size: int(block_labels(cells, size).nunique()) for size in SIZES}
numbers["region"] = region

fold_of = np.zeros(len(cells), dtype=int)
for fold, (_, va) in enumerate(GroupKFold(5).split(cells, groups=block_labels(cells, 4))):
    fold_of[va] = fold
fig, (left, middle) = plt.subplots(1, 2, figsize=(21, 7.4), width_ratios=[1, 1.9])
left.scatter(cells["lon"], cells["lat"], c=fold_of, cmap="Set2", s=150, marker="s", linewidths=0)
for edge in np.arange(-128, -108, 4):
    left.axvline(edge, color="white", lw=2.5)
for edge in np.arange(28, 52, 4):
    left.axhline(edge, color="white", lw=2.5)
left.set(xlim=(cells["lon"].min() - 0.5, cells["lon"].max() + 0.5), ylim=(cells["lat"].min() - 0.5, cells["lat"].max() + 0.5),
         xlabel="Longitude", ylabel="Latitude")
left.set_title("4° × 4° blocks, colored by fold", loc="left")
left.set_aspect(1.25)
positions = ["Random\ncells", "2° blocks", "4° blocks", "6° blocks"]
for name, color in [("climate", GREEN), ("coordinates", RED)]:
    values = [region["random"][name]] + [region["blocks"][name][s] for s in SIZES]
    middle.plot(positions, values, "o-", color=color, lw=3, ms=11)
    middle.text(3.2, values[-1], {"climate": "RF from\nclimate only", "coordinates": "RF from\nlatitude and\nlongitude only"}[name], color=color, va="center", fontsize=16, fontweight="bold")
    for i, v in enumerate(values):
        middle.text(i + (0.14 if i == 0 and name == "climate" else 0), v + (0.035 if name == "climate" else -0.065), f"{v:.2f}", ha="center", color=color, fontsize=17)
middle.set(ylabel="R², held-out cells", ylim=(0, 1.02), xlim=(-0.3, 4.3))
middle.set_title("Annual soil moisture, 992 cells", loc="left")
fig.tight_layout()
save(fig, "map_blocks")

# --- Event: synthetic ground motion, planted per-event term -------------------------------------------
rng = np.random.default_rng(SEED)
n_events, n_stations = 60, 15
station_x, station_y = rng.uniform(0, 200, n_stations), rng.uniform(0, 200, n_stations)
station_term = rng.normal(0, 0.10, n_stations)
records = []
for event in range(n_events):
    magnitude = rng.uniform(4, 7)
    x, y = rng.uniform(0, 200), rng.uniform(0, 200)
    event_term = rng.normal(0, 0.30)
    for station in range(n_stations):
        distance = np.hypot(x - station_x[station], y - station_y[station]) + 5
        log_pga = 0.6 * magnitude - 1.5 * np.log10(distance) + event_term + station_term[station] + rng.normal(0, 0.15)
        records.append({"event": event, "magnitude": magnitude, "distance_km": distance, "log_pga": log_pga})
motions = pd.DataFrame(records)
forest_event = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
X_event, y_event = motions[["magnitude", "distance_km"]], motions["log_pga"]
numbers["event"] = {"random_r2": cross_val_score(forest_event, X_event, y_event, cv=KFold(5, shuffle=True, random_state=SEED)).mean(),
                    "grouped_r2": cross_val_score(forest_event, X_event, y_event, cv=GroupKFold(5), groups=motions["event"]).mean()}

# --- Ensembles: seismic source types ------------------------------------------------------------------
tables = []
for name, checksum in SEISMIC_FILES.items():
    path = pooch.retrieve(url=f"https://zenodo.org/api/records/14025693/files/{name}/content", known_hash=checksum, fname=name, path=pooch.os_cache("mlgeo"))
    tables.append(pd.read_csv(path, index_col=0))
seismic = pd.concat(tables, ignore_index=True).dropna(axis=1)
X, y = seismic.drop(columns=["source", "serial_no"]), seismic["source"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y)
members = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
           "forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
           "boosted": HistGradientBoostingClassifier(random_state=SEED)}
truth = y_test.to_numpy()
macro = lambda a, b: f1_score(a, b, average="macro")
probabilities, out_of_fold = {}, {}
for name, model in members.items():
    probabilities[name] = model.fit(X_train, y_train).predict_proba(X_test)
    out_of_fold[name] = cross_val_predict(model, X_train, y_train, cv=3, method="predict_proba")
classes = members["forest"].classes_
labels = {name: classes[p.argmax(axis=1)] for name, p in probabilities.items()}
wrong = {name: labels[name] != truth for name in members}
average = np.mean(list(probabilities.values()), axis=0)
stack = LogisticRegression(max_iter=5000).fit(np.hstack(list(out_of_fold.values())), y_train)
stacked = stack.predict(np.hstack(list(probabilities.values())))
agree = (labels["logistic"] == labels["forest"]) & (labels["forest"] == labels["boosted"])
numbers["ensemble"] = {"f1": {name: macro(truth, labels[name]) for name in members},
                       "average_f1": macro(truth, classes[average.argmax(axis=1)]), "stack_f1": macro(truth, stacked),
                       "wrong": {name: int(wrong[name].sum()) for name in members},
                       "all_three_wrong": int((wrong["logistic"] & wrong["forest"] & wrong["boosted"]).sum()),
                       "agree_events": int(agree.sum()), "agree_error": float(wrong["forest"][agree].mean()),
                       "disagree_events": int((~agree).sum()), "disagree_average_error": float((classes[average.argmax(axis=1)] != truth)[~agree].mean())}
e = numbers["ensemble"]
fig, (left, right, third) = plt.subplots(1, 3, figsize=(15, 6.6), width_ratios=[1.4, 1, 0.75])
names = ["Logistic\nregression", "Forest", "Boosted\ntrees", "Equal\naverage", "Stack"]
values = [e["f1"]["logistic"], e["f1"]["forest"], e["f1"]["boosted"], e["average_f1"], e["stack_f1"]]
bars = left.bar(names, values, color=[GRAY, GRAY, GRAY, BLUE, BLUE])
for bar, v in zip(bars, values):
    left.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=18)
left.set(ylim=(0.85, 0.915), ylabel="1,000 test events")
left.set_title("Macro-F1", loc="left")
left.tick_params(axis="x", labelsize=14)
shared = e["all_three_wrong"]
y_pos = np.arange(3)[::-1]
for pos, name, label in zip(y_pos, members, ["Logistic regression", "Forest", "Boosted trees"]):
    right.barh(pos, shared, color=RED)
    right.barh(pos, e["wrong"][name] - shared, left=shared, color=GRAY)
    right.text(e["wrong"][name] + 1.5, pos, str(e["wrong"][name]), va="center", fontsize=18)
right.set_yticks(y_pos, ["Logistic", "Forest", "Boosted"])
right.set(xlabel="Test events misclassified", xlim=(0, 140))
right.set_title(f"{shared} fool all three", loc="left", color=RED)
rates = [e["agree_error"] * 100, e["disagree_average_error"] * 100]
third.bar([f"Agree\n{e['agree_events']} events", f"Disagree\n{e['disagree_events']} events"], rates, color=[GRAY, BLUE])
for i, v in enumerate(rates):
    third.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=18)
third.set(ylabel="Average is wrong", ylim=(0, 50))
third.set_title("Disagreement", loc="left")
third.tick_params(axis="x", labelsize=14)
fig.tight_layout()
save(fig, "ensemble_scores")

# --- Boosting by hand: USCRN air and ground temperature ----------------------------------------------
columns = (SNAPSHOTS / "uscrn_daily01_HEADERS.txt").read_text().splitlines()[1].split()
station = pd.read_csv(SNAPSHOTS / "uscrn_WA_Spokane_17_SSW_2015-2024.txt", sep=r"\s+", names=columns, dtype={"LST_DATE": str},
                      na_values=[-9999, -99, -9999.0, -99.0, -99.000])
station["date"] = pd.to_datetime(station["LST_DATE"])
surface = station.set_index("date")[["T_DAILY_MAX", "SUR_TEMP_DAILY_MAX"]].dropna()
surface.columns = ["air", "ground"]
fit, held = surface[:"2021"], surface["2022":]
RATE, DEPTH, ROUNDS = 0.3, 2, 300
fit_pred, held_pred = np.full(len(fit), fit["ground"].mean()), np.full(len(held), fit["ground"].mean())
line_x = pd.DataFrame({"air": np.linspace(surface["air"].min(), surface["air"].max(), 500)})
line_pred = np.full(len(line_x), fit["ground"].mean())
curve, snapshots = [], {}
for step in range(1, ROUNDS + 1):
    tree = DecisionTreeRegressor(max_depth=DEPTH, random_state=SEED).fit(fit[["air"]], fit["ground"] - fit_pred)
    fit_pred += RATE * tree.predict(fit[["air"]])
    held_pred += RATE * tree.predict(held[["air"]])
    line_pred += RATE * tree.predict(line_x)
    curve.append((step, mean_absolute_error(fit["ground"], fit_pred), mean_absolute_error(held["ground"], held_pred)))
    if step in (1, 3, 10, 300):
        snapshots[step] = line_pred.copy()
curve = np.array(curve)
best = int(curve[curve[:, 2].argmin(), 0])
numbers["boost"] = {"fit_days": len(fit), "held_days": len(held), "start_mae": mean_absolute_error(held["ground"], np.full(len(held), fit["ground"].mean())),
                    "rounds": {str(r): {"fit": curve[r - 1, 1], "held": curve[r - 1, 2]} for r in (1, 3, 10, 300)}, "best_round": best, "best_held": curve[best - 1, 2],
                    "forest_held": mean_absolute_error(held["ground"], RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(fit[["air"]], fit["ground"]).predict(held[["air"]])),
                    "line_held": mean_absolute_error(held["ground"], LinearRegression().fit(fit[["air"]], fit["ground"]).predict(held[["air"]]))}
fig = plt.figure(figsize=(15, 7.4))
grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
for spot, step in zip([(0, 0), (0, 1), (1, 0), (1, 1)], (1, 3, 10, 300)):
    ax = fig.add_subplot(grid[spot])
    ax.scatter(fit["air"], fit["ground"], s=3, color=GRAY)
    ax.plot(line_x["air"], snapshots[step], color=BLUE, lw=2.6)
    ax.set_title(f"{step} tree{'s' if step > 1 else ''}", loc="left", fontsize=20)
    ax.set(ylim=(-25, 65))
    if spot[1] == 0:
        ax.set_ylabel("Ground (°C)", fontsize=17)
    if spot[0] == 1:
        ax.set_xlabel("Air (°C)", fontsize=17)
    ax.tick_params(labelsize=14)
ax = fig.add_subplot(grid[:, 2])
ax.plot(curve[:, 0], curve[:, 1], color=GRAY, lw=3)
ax.plot(curve[:, 0], curve[:, 2], color=BLUE, lw=3)
ax.set_xscale("log")
ax.set(xlabel="Trees added", ylabel="Error (°C)", ylim=(3, 12))
ax.text(6, 7.2, "Held-out years\n2022 to 2024", color=BLUE, fontsize=17, fontweight="bold", va="top")
ax.text(40, 3.35, "Training years", color=MUTED, fontsize=17, fontweight="bold")
ax.set_title("Each tree fits the residuals", loc="left", fontsize=20)
fig.tight_layout()
save(fig, "boosting_by_hand")

# --- Automated search: P395, first 80% searched, last 20% kept for one final test --------------------
cut = int(0.8 * len(X_time))
X_search, y_search, X_final, y_final = X_time[:cut], y_time[:cut], X_time[cut + GAP:], y_time[cut + GAP:]
candidates = {"Linear": LinearRegression(), "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
              "Nearest 3": make_pipeline(StandardScaler(), KNeighborsRegressor(3)), "Nearest 30": make_pipeline(StandardScaler(), KNeighborsRegressor(30)),
              "Forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
              "Forest, leaf 20": RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=SEED, n_jobs=-1),
              "Boosted": HistGradientBoostingRegressor(random_state=SEED), "Boosted, slow": HistGradientBoostingRegressor(learning_rate=0.03, random_state=SEED)}
board = pd.DataFrame({"shuffled": {name: mae_cv(m, X_search, y_search, shuffled).mean() for name, m in candidates.items()},
                      "future": {name: mae_cv(m, X_search, y_search, TimeSeriesSplit(5, gap=GAP)).mean() for name, m in candidates.items()}})
winners = {split: board[split].idxmin() for split in board}
final = {split: mean_absolute_error(y_final, candidates[w].fit(X_search, y_search).predict(X_final)) for split, w in winners.items()}
numbers["automl"] = {"board": board.to_dict(), "winners": winners, "final_mae": final, "final_persistence": mean_absolute_error(y_final, X_final[:, FEATURES.index("last")]),
                     "search_samples": len(X_search), "final_samples": len(X_final)}
fig, ax = plt.subplots(figsize=(15, 7.6))
trees = {"Forest": ORANGE, "Forest, leaf 20": ORANGE, "Boosted": RED, "Boosted, slow": RED, "Nearest 3": MUTED, "Nearest 30": MUTED, "Linear": BLUE, "Ridge": BLUE}
for name, row in board.iterrows():
    ax.plot([0, 1], [row["shuffled"], row["future"]], "o-", color=trees[name], lw=2.8, ms=10)
right_side = board["future"].sort_values()
placed = []
for name, value in right_side.items():
    spot = value * 1.0
    while placed and spot / placed[-1] < 1.189:
        spot = placed[-1] * 1.19
    placed.append(spot)
    ax.text(1.03, spot, f"{value:.1f}  {name}", color=trees[name], va="center", fontsize=18, fontweight="bold" if name == winners["future"] else "normal")
left_side = board["shuffled"].sort_values()
placed = []
for name, value in left_side.items():
    spot = value
    while placed and spot / placed[-1] < 1.189:
        spot = placed[-1] * 1.19
    placed.append(spot)
    ax.text(-0.03, spot, f"{value:.2f}  {name}", color=trees[name], ha="right", va="center", fontsize=18, fontweight="bold" if name == winners["shuffled"] else "normal")
ax.set_yscale("log")
ax.set_ylim(0.4, 42)
ax.set_xlim(-0.42, 1.42)
ax.set_xticks([0, 1], ["Ranked on shuffled folds", "Ranked on future folds"], fontsize=20, fontweight="bold")
ax.set_yticks([])
ax.minorticks_off()
ax.spines["left"].set_visible(False)
ax.set_ylabel("Forecast error (mm, log scale)")
fig.tight_layout()
save(fig, "automl_two_rankings")

# --- Search optimism: best of many settings on 300 events --------------------------------------------
few = np.random.default_rng(1).choice(len(X_train), 300, replace=False)
X_few, y_few = X_train.iloc[few], y_train.iloc[few]
draw = np.random.default_rng(SEED)
scores = []
for _ in range(30):
    kind = draw.choice(["logistic", "forest", "nearest"])
    if kind == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=10 ** draw.uniform(-3, 2), max_iter=5000))
    elif kind == "forest":
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=int(draw.choice([1, 2, 5, 10, 20])), max_features=float(draw.choice([0.1, 0.33, 1.0])), random_state=SEED, n_jobs=-1)
    else:
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(int(draw.choice([1, 3, 5, 10, 20, 40]))))
    scores.append((cross_val_score(model, X_few, y_few, cv=5, scoring="f1_macro").mean(), macro(truth, model.fit(X_few, y_few).predict(X_test))))
scores = np.array(scores)
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.scatter(scores[:, 0], scores[:, 1], s=110, color=GRAY)
top = scores[:, 0].argmax()
ax.scatter(*scores[top], s=260, color=RED, zorder=3)
ax.annotate(f"Chosen by the search:\n{scores[top, 0]:.3f} cross-validated, {scores[top, 1]:.3f} on test", scores[top], xytext=(scores[top, 0] - 0.13, scores[top, 1] + 0.012),
            color=RED, fontsize=19, fontweight="bold", arrowprops={"arrowstyle": "-", "color": RED})
limits = [scores.min() - 0.01, scores.max() + 0.01]
ax.plot(limits, limits, color=MUTED, ls="--")
ax.text(limits[0] + 0.006, limits[0] + 0.012, "same score on both", color=MUTED, rotation=26, fontsize=17)
ax.set(xlim=limits, ylim=limits, xlabel="Cross-validated macro-F1, 300 training events", ylabel="Macro-F1, 1,000 test events")
fig.tight_layout()
save(fig, "search_optimism")
numbers["optimism"] = {"tries": len(scores), "best_cv": scores[:, 0].max(), "test_of_best": scores[scores[:, 0].argmax(), 1], "best_test": scores[:, 1].max()}

# --- A script an agent could write: drops a class, scales before the split, scores on training ------
two_thirds = seismic[seismic["source"] != "surface event"]
scaled = StandardScaler().fit_transform(two_thirds.drop(columns=["source", "serial_no"]))
a_train, a_test, b_train, b_test = train_test_split(scaled, two_thirds["source"], test_size=0.25, random_state=7)
flawed = {name: model.fit(a_train, b_train).score(a_train, b_train) for name, model in
          {"logistic": LogisticRegression(max_iter=5000), "forest": RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)}.items()}
repaired = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1))
numbers["audit"] = {"flawed_training_accuracy": flawed, "repaired_cv_macro_f1": cross_val_score(repaired, X_train, y_train, cv=5, scoring="f1_macro").mean(),
                    "repaired_test_macro_f1": macro(truth, repaired.fit(X_train, y_train).predict(X_test))}

# --- Practices: what each training choice changes, scored on the outcome that matters ---------------
# P395: six forest settings and one change of target, scored on shuffled folds (what an unchecked report
# would claim) and on future folds (the forecast). Same samples, folds and settings as notebook 3.8, Part 1.
SETTINGS = [{"max_depth": depth, "min_samples_leaf": leaf} for depth in (3, 12, None) for leaf in (1, 20)]
tuning = pd.DataFrame([{**setting,
                        "shuffled": mae_cv(RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, **setting), X_time, y_time, shuffled).mean(),
                        "future": mae_cv(RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, **setting), X_time, y_time, future).mean()}
                       for setting in SETTINGS])
X_change = np.column_stack([samples["mean_60d"] - last_value, samples["mean_10d"] - last_value,
                            samples[["slope_60d", "season_sin", "season_cos"]].to_numpy()])
y_change = y_time - last_value


def change_mae(split):
    """Error in position (mm) of a forest that predicts the change from the last observed day."""
    model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    return np.mean([mean_absolute_error(y_time[va], last_value[va] + model.fit(X_change[tr], y_change[tr]).predict(X_change[va]))
                    for tr, va in split.split(X_change)])


by_shuffled, by_future = tuning.loc[tuning["shuffled"].idxmin()], tuning.loc[tuning["future"].idxmin()]
numbers["practice_time"] = {"settings": tuning.to_dict("records"),
                            "pick_by_shuffled": {"shuffled": by_shuffled["shuffled"], "future": by_shuffled["future"]},
                            "pick_by_future": {"shuffled": by_future["shuffled"], "future": by_future["future"]},
                            "future_range": [tuning["future"].min(), tuning["future"].max()],
                            "change_shuffled": change_mae(shuffled), "change_future": change_mae(future)}
t = numbers["practice_time"]

fig, ax = plt.subplots(figsize=(15, 6.8))
persist = numbers["time"]["persistence_future_mae"]
rows = [("Forecast future position\n(6 random forests)", tuning["shuffled"].to_numpy(), tuning["future"].to_numpy(), RED),
        ("Forecast change in position\n(1 random forest)", [t["change_shuffled"]], [t["change_future"]], GREEN),
        ("Forecast future position\n(linear regression)", [], [numbers["time"]["linear_future_mae"]], BLUE)]
for i, (label, reported, forecast, color) in enumerate(rows):
    yy = len(rows) - 1 - i
    if len(reported):
        ax.plot([min(reported), max(forecast)], [yy, yy], color=GRAY, lw=2, zorder=1)
        ax.scatter(reported, [yy] * len(reported), s=230, facecolors="white", edgecolors=color, linewidths=3, zorder=2)
        ax.text(min(reported), yy - 0.33, f"{min(reported):.2f}" + (f" to {max(reported):.1f}" if len(reported) > 1 else ""),
                ha="center", color=MUTED, fontsize=17)
    ax.scatter(forecast, [yy] * len(forecast), s=260, color=color, zorder=3)
    best = min(forecast)
    ax.text(best, yy + 0.22, (f"{best:.1f} to {max(forecast):.1f}" if len(forecast) > 1 else f"{best:.2f}"),
            ha="left" if len(forecast) > 1 else "center", color=color, fontsize=19, fontweight="bold")
ax.axvline(persist, color=INK, ls="--", lw=2)
ax.text(persist * 1.05, 2.55, f"persistence: forecast = last\nobserved position, {persist:.2f} mm", color=INK, fontsize=17, va="top")
ax.set_xscale("log")
ax.set_xticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=19)
ax.set(xlabel="Mean absolute error of the 30-day position (mm), log scale", ylim=(-0.6, len(rows) - 0.45), xlim=(0.3, 40))
ax.scatter([], [], s=200, facecolors="white", edgecolors=MUTED, linewidths=3, label="shuffled folds: what an unchecked report claims")
ax.scatter([], [], s=200, color=MUTED, label="future folds: the forecast")
ax.legend(frameon=False, loc="lower right", fontsize=17)
fig.tight_layout()
save(fig, "practice_p395")

# Soil map: the same six settings, coordinates or climate, scored on random cells and on 6 degree blocks.
blocks6 = block_labels(cells, 6)


def soil_r2(columns, cv, groups=None, **setting):
    model = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, **setting)   # as in the map figure
    return r2_score(soil, cross_val_predict(model, cells[columns], soil, cv=cv, groups=groups))


soil_practice = {}
for name, columns in [("coordinates", COORDINATES), ("climate", CLIMATE)]:
    random_scores = [soil_r2(columns, KFold(5, shuffle=True, random_state=SEED), **s) for s in SETTINGS]
    block_scores = [soil_r2(columns, GroupKFold(5), blocks6, **s) for s in SETTINGS]
    soil_practice[name] = {"random_range": [min(random_scores), max(random_scores)], "block_range": [min(block_scores), max(block_scores)],
                           "block_of_random_pick": block_scores[int(np.argmax(random_scores))], "best_block": max(block_scores)}
numbers["practice_region"] = soil_practice

# Seismic source types, canonical split: tuning, amount of data, convergence.
grid_forest = {}
for leaf in (1, 3, 10):
    for features in ("sqrt", 0.5):
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, max_features=features, random_state=SEED, n_jobs=-1)
        grid_forest[(leaf, features)] = (cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro").mean(), model)
best_key = max(grid_forest, key=lambda k: grid_forest[k][0])
smaller = {}
for n in (300, 1000):
    X_small, _, y_small, _ = train_test_split(X_train, y_train, train_size=n, random_state=SEED, stratify=y_train)
    smaller[n] = macro(truth, RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_small, y_small).predict(X_test))
numbers["practice_seismic"] = {"forest_default": e["f1"]["forest"],
                               "forest_tuned": macro(truth, grid_forest[best_key][1].fit(X_train, y_train).predict(X_test)),
                               "tuned_setting": list(best_key), "train_300": smaller[300], "train_1000": smaller[1000],
                               "train_all": len(X_train), "logistic_scaled": e["f1"]["logistic"],
                               "logistic_raw_100_iterations": macro(truth, LogisticRegression(max_iter=100).fit(X_train, y_train).predict(X_test))}

(FIGS / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
print(json.dumps(numbers, indent=1, default=float))
