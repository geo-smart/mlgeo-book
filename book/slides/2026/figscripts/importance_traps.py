"""Slide figure for lecture 17: two importance rankings, one trap.

Left panel: notebook 3.7's duplicated-feature demonstration — impurity
importance splits temp_1's credit with its Celsius twin (0.93 -> 0.49 + 0.44)
while the held-out error does not move. Right panel: permutation importance
on the held-out samples. Re-plotted at 20pt from the exact notebook pipeline
(same generator, same seeds); headline numbers asserted.

Regenerate: pixi run python book/slides/2026/figscripts/importance_traps.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

TAG = ("Synthetic Seattle-like daily max temperature, 2012–2019 "
       "(notebook 3.7 generator — synthetic)")


def make_daily_temps(start="2012-01-01", end="2019-12-31", seed=42):
    """Verbatim from notebook 3.7 so the numbers match the book."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    day_of_year = dates.dayofyear.to_numpy()
    climatology = 62.0 - 15.0 * np.cos(2 * np.pi * (day_of_year - 203) / 365.25)
    trend = 0.05 * np.arange(len(dates)) / 365.25
    noise = np.zeros(len(dates))
    for i in range(1, len(dates)):
        noise[i] = 0.65 * noise[i - 1] + rng.normal(0.0, 3.0)
    df = pd.DataFrame({
        "date": dates,
        "average": np.round(climatology, 1),
        "actual": np.round(climatology + trend + noise, 1),
    })
    df["temp_1"] = df["actual"].shift(1)
    df["temp_2"] = df["actual"].shift(2)
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df.dropna().reset_index(drop=True)


df = make_daily_temps()
features = ["temp_1", "temp_2", "average", "month", "day", "doy_sin", "doy_cos"]
X, y = df[features], df["actual"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
rf_mae = mean_absolute_error(y_test, rf.predict(X_test))
imp = pd.Series(rf.feature_importances_, index=features)

X_train_dup, X_test_dup = X_train.copy(), X_test.copy()
X_train_dup["temp_1_C"] = (X_train_dup["temp_1"] - 32) * 5 / 9
X_test_dup["temp_1_C"] = (X_test_dup["temp_1"] - 32) * 5 / 9
rf_dup = RandomForestRegressor(n_estimators=300, random_state=42).fit(
    X_train_dup, y_train)
dup_mae = mean_absolute_error(y_test, rf_dup.predict(X_test_dup))
imp_dup = pd.Series(rf_dup.feature_importances_, index=X_train_dup.columns)

perm = permutation_importance(
    rf, X_test, y_test, n_repeats=20, random_state=42,
    scoring="neg_mean_absolute_error")

assert f"{imp['temp_1']:.2f}" == "0.93", imp["temp_1"]
assert f"{imp_dup['temp_1']:.2f}" == "0.49", imp_dup["temp_1"]
assert f"{imp_dup['temp_1_C']:.2f}" == "0.44", imp_dup["temp_1_C"]
assert f"{rf_mae:.2f}" == "2.55" and f"{dup_mae:.2f}" == "2.55", (rf_mae, dup_mae)
print(f"impurity temp_1 {imp['temp_1']:.3f} -> {imp_dup['temp_1']:.3f} "
      f"+ {imp_dup['temp_1_C']:.3f};  test MAE {rf_mae:.2f} -> {dup_mae:.2f} F")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.6))

# Left: impurity credit before/after adding the unit-converted twin.
show = ["temp_1", "temp_1_C", "average", "temp_2", "day", "doy_cos", "doy_sin",
        "month"]
pos = np.arange(len(show))
before = [imp.get(f, 0.0) for f in show]
after = [imp_dup.get(f, 0.0) for f in show]
h = 0.38
ax1.barh(pos + h / 2, before, height=h, color="#1f77b4", label="7 features")
ax1.barh(pos - h / 2, after, height=h, color="#ff7f0e",
         label="+ duplicate (°C copy of temp_1)")
ax1.set_yticks(pos, ["yesterday (°F)", "yesterday (°C)  — same data",
                     "climatology", "two days ago", "day", "cos(day of year)",
                     "sin(day of year)", "month"])
ax1.invert_yaxis()
ax1.set_xlabel("Impurity importance (training)")
ax1.set_title("Credit splits between identical twins", loc="left")
ax1.annotate("0.93", xy=(before[0], pos[0] + h / 2), xytext=(before[0] + 0.01,
             pos[0] + h / 2), va="center", fontsize=19, fontweight="bold")
ax1.annotate("0.49 + 0.44", xy=(after[1], pos[1] - h / 2),
             xytext=(after[1] + 0.01, pos[1] - h / 2), va="center",
             fontsize=19, fontweight="bold", color="#b3402a")
ax1.legend(loc="lower right", fontsize=16)
ax1.text(0.98, 0.55, f"held-out error unchanged:\nMAE {rf_mae:.2f} °F either way",
         transform=ax1.transAxes, ha="right", fontsize=18, color="#116b66",
         fontweight="bold")

# Right: permutation importance on held-out samples.
nice = {"temp_1": "yesterday (°F)", "temp_2": "two days ago",
        "average": "climatology", "month": "month", "day": "day",
        "doy_sin": "sin(day of year)", "doy_cos": "cos(day of year)"}
order = np.argsort(perm.importances_mean)
ax2.barh([nice[f] for f in np.array(features)[order]],
         perm.importances_mean[order],
         xerr=perm.importances_std[order], color="#1f77b4")
ax2.set_xlabel("Increase in held-out MAE (°F) when shuffled")
ax2.set_title("What the model relies on for new data", loc="left")

fig.suptitle(TAG, x=0.01, ha="left", fontsize=18, fontweight="normal",
             color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = (Path(__file__).resolve().parent.parent / "figs"
       / "3.7_randomForest_regression" / "importance_traps_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
