"""Self-checks and the session summary for lab_s07_land_surface_model.ipynb (Session 7, land-surface track).

``check(qid, answer)`` prints "correct", a hint (never the answer) or "not answered yet", and remembers the
result. The "Your turn" cells, where you write a few lines of analysis, are checked the same way, and
``use_or_reference`` lets the cells after them run on a reference result when yours is missing or has the wrong
form. ``score_findings`` scores your Part 5 findings against your twin and remembers the count.
``print_summary(NETID)`` prints the block the instructor reads on Canvas.

Expected values are computed from the course copies in snapshots/ each time a check runs; no answer is typed
into this file. The worked solutions call ``check_worked``, which prints the same way but records nothing, so
opening a solution never changes your summary.
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

import lab_s07_lsm_synthetic as lab
from lab_s07_lsm_synthetic import GRID_FILE, NOT_ANSWERED, W_MAX, open_grid, spokane_anomalies, spokane_table

TRACK = "lsm"
PARTS = {
    0: ["q0_peak_month"],
    1: ["q1_same_bytes", "q1_demo", "q1_cell_km", "q1_gauges", "q1_raw_or_derived"],
    2: ["q2_limits", "p2_table", "p2_driest", "q2_extreme", "q2_sentinel"],
    3: ["p3_series", "q3_interval", "q3_frozen", "p3_monthly"],
    4: ["q4_lat_order", "q4_cell", "p4_clim", "q4_region", "q4_bad_cell"],
    5: ["q5a", "p5_counts", "q5b", "q5c", "q5d"],
    6: ["q6_sign", "p6_anom", "p6_leads", "q6_lag", "q6_neff", "q6_meaning"],
    7: ["q7_pipeline", "q7_task", "p7_split", "q7_leak"],
    8: ["q8a", "q8"],
}
# Questions with a right answer; the others (lists, findings, free text) count as answered or not.
SELF_CHECKED = [q for p in (0, 1, 2, 3, 4, 6, 7) for q in PARTS[p]] + ["p5_counts"]

# The Your-turn checks added on 14 October (afternoon). A copy of the earlier notebook never runs them; after
# git pull it imports this module, and its summary counts the earlier parts and checks, so a finished copy reads
# as it did before the update (see _is_new_version).
YOUR_TURN_QIDS = ("p2_table", "p2_driest", "p3_series", "p3_monthly", "p4_clim", "p5_counts", "p6_anom",
                  "p6_leads", "p7_split")
PARTS_EARLIER = {p: [q for q in qs if q not in YOUR_TURN_QIDS] for p, qs in PARTS.items()}
SELF_CHECKED_EARLIER = [q for q in SELF_CHECKED if q not in YOUR_TURN_QIDS]
# Set in the first code cell of the updated notebook only.
VERSION_MARKER = "LAB_VERSION"
# Names that only the updated notebook defines (Your-turn placeholders and given cells).
_NEW_VERSION_NAMES = ("n_missing", "precip_total", "sm5_monthly", "n_valid", "rows_per_month", "rate_test",
                      "my_clean_example", "r_lead")

_ANSWERED, _PASSED = {}, {}
_PASSED_EARLIER = {}   # the earlier rule, for the two checks whose tolerance was tightened on 14 October
_STATE = {"found": None, "derived": None, "wrap": None}


# --------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------
def _ok(msg="correct"):
    print(msg)
    return True


def _hint(msg):
    print("not yet. Hint: " + msg)
    return False


def _words(s):
    return len(str(s or "").split())


def _first_word(s):
    """The first word of an answer, lower case, without punctuation: "Derived: model output" gives "derived"."""
    w = re.findall(r"[a-z0-9-]+", str(s).lower())
    return w[0] if w else ""


def _letter(s):
    """A one-letter choice ("A", "a", "A.", "(A)", "C) because ..."); "Any of them" is not a choice."""
    m = re.match(r"\s*\(?([A-Da-d])(?:[).:,]|\s|$)", str(s))
    return m.group(1).upper() if m else ""


def _blank(a):
    return a is None or a is ... or (isinstance(a, (str, list, dict, tuple)) and len(a) == 0) or (
        isinstance(a, str) and not a.strip())


# --------------------------------------------------------------------------------------------------
# Expected values, computed from the course copies
# --------------------------------------------------------------------------------------------------
def _peak_month():
    t = spokane_table()
    m = pd.to_datetime(t["month"]).dt.month
    return int(t.groupby(m)["soilw_mm"].mean().idxmax())


def _driest_month():
    t = spokane_table()
    return t.loc[t["soilw_mm"].idxmin(), "month"]


def _cell_km():
    c_lat = float(spokane_table()["cell_lat"].iloc[0])
    return 111.2 * 0.5, 111.2 * 0.5 * np.cos(np.deg2rad(c_lat))


def _frozen_gap_pct():
    u = lab.uscrn_daily().loc["2015":"2020"]
    miss = u["SOIL_MOISTURE_5_DAILY"].isna()
    m = u.index.month
    return 100 * miss[np.isin(m, [12, 1, 2])].mean(), 100 * miss[np.isin(m, [6, 7, 8])].mean()


def _best_lag():
    a = spokane_anomalies()
    r = [a["soilw_mm"].corr(a["precip_mm_per_day"].shift(k)) for k in range(4)]
    return int(np.argmax(r)), r


def _neff():
    a = spokane_anomalies()
    x, y = a["soilw_mm"], a["precip_mm_per_day"]
    r1x, r1y = x.autocorr(1), y.autocorr(1)
    return len(x) * (1 - r1x * r1y) / (1 + r1x * r1y)


def _real_test_rate():
    y, _ = lab.real_drought_labels()
    return float(100 * y.sel(time=slice("2011", "2020")).mean())


def _box_range(var):
    g = open_grid(GRID_FILE)[var]
    return float(g.min()), float(g.max())


# --------------------------------------------------------------------------------------------------
# One function per question: True (correct), False (a hint was printed) or None (not answered)
# --------------------------------------------------------------------------------------------------
def _c_q0(a):
    p = _peak_month()
    d = min(abs(int(a) - p), 12 - abs(int(a) - p))
    if d <= 1:
        return _ok("correct" if d == 0 else "close enough: the peak is within one month of your choice")
    if int(a) in (6, 7, 8, 9):
        return _hint("summer is when evaporation empties the bucket. Until which month does winter "
                     "precipitation exceed the evaporation that cold air allows?")
    return _hint("average soilw_mm by calendar month (groupby month, mean) and read the maximum.")


def _c_q1_same(a):
    if not isinstance(a, bool):
        return _hint("answer True or False.")
    if a == lab._LAST.get("same_bytes", True):
        return _ok()
    return _hint("compare the two 'sha256' lines of the table: identical bytes give identical fingerprints.")


def _c_q1_demo(a):
    if a.get("size_changes") is False and a.get("fingerprint_changes") is True:
        return _ok()
    if a.get("size_changes"):
        return _hint("a float32 value takes 4 bytes whatever the number. Compare the two sizes printed.")
    return _hint("compare the two fingerprints printed by the demonstration cell.")


def _c_q1_km(a):
    ns, ew = _cell_km()
    if abs(a[0] - ns) < 3 and abs(a[1] - ew) < 3:
        return _ok()
    if abs(a[1] - ns) < 3:
        return _hint("a degree of longitude shrinks with latitude: multiply by cos(latitude).")
    return _hint("one degree of latitude is about 111.2 km; the cell is 0.5 degree on a side.")


def _c_q1_raw(a):
    if _first_word(a) == "derived":
        return _ok()
    return _hint("no instrument measured soil water here. Which numbers went into the model, and what did "
                 "the model do with them?")


def _c_q1_gauges(a):
    v0, v1 = lab.gauge_totals()
    if abs(a[0] - v0) <= 0.005 * v0 + 0.5 and abs(a[1] - v1) <= 0.005 * v1 + 0.5:
        return _ok()
    return _hint("average static['gauges_total'] over the months of 1991-1996, then over 1997-2020; both "
                 "periods include their first and last year (year <= 1996, year >= 1997).")


def _c_q2_limits(a):
    lo, hi = a["soilw_mm"]
    msgs = []
    if lo < 0 or lo > 1e-6 or hi < W_MAX - 1 or hi > 1000:
        msgs.append("soil water: the bucket cannot be emptier than empty or fuller than its holding capacity "
                    "(Part 0). Write limits that hold for any cell, not the values you expect at this one.")
    plo, phi = a["precip_mm_per_day"]
    p_min, p_max = _box_range("precip")
    if plo < 0 or plo > p_min:
        msgs.append("precipitation: it cannot be negative, and a dry month has none at all.")
    if phi < p_max:
        msgs.append("precipitation: your upper limit removes real wet months somewhere in the box. Compare it "
                    "with the largest value on the whole grid (grid['precip'].max()), not only at Spokane.")
    elif phi > 656:
        msgs.append("precipitation: a month cannot average more than its wettest day, and the wettest 24 hours on "
                    "record in California, Oregon, Washington, Idaho or Nevada is 656 mm (California, January 1943; "
                    "NCEI State Climate Extremes Committee). A limit above that catches nothing.")
    if msgs:
        return _hint(" ".join(msgs))
    return _ok("correct: your limits bracket every value and follow the physics")


def _c_q2_extreme(a):
    m, kind = a
    if str(m)[:7] != _driest_month():
        return _hint("find the row with the smallest soilw_mm (idxmin) and give its month as 'YYYY-MM'.")
    if _first_word(kind) != "real":
        return _hint("look at the precipitation in the months before it. Is a dry bucket after a dry summer "
                     "physically possible? Is the value inside the limits you wrote?")
    return _ok()


def _c_q2_sentinel(a):
    return _ok() if abs(float(a) + 99.0) < 1e-6 else _hint(
        "the USCRN readme (snapshots/uscrn_daily01_readme.txt, note C) gives the code for 7-character fields "
        "with three decimals.")


def _c_q3_interval(a):
    return _ok() if tuple(sorted(a)) == (28, 31) else _hint(
        "steps should be numbers of days: dates.diff().dt.days, then .min() and .max().")


def _c_q3_frozen(a):
    w, s = _frozen_gap_pct()
    if abs(a[0] - w) <= 2 and abs(a[1] - s) <= 2:
        return _ok()
    return _hint("in u (from lab.uscrn_daily()) the -99 codes are already NaN: count the days where "
                 "u['SOIL_MOISTURE_5_DAILY'].isna() in December-February and in June-August of 2015-2020, divide "
                 "by the number of days in those months and multiply by 100.")


def _c_q4_order(a):
    s = str(a).strip().lower()
    ok = _first_word(s).startswith("desc") or s.startswith("north to south") or s.startswith("from north")
    return _ok() if ok else _hint("print grid.lat.values[:3]: does latitude grow or shrink with the index?")


def _c_q4_cell(a):
    t = spokane_table()
    la, lo = float(t["cell_lat"].iloc[0]), float(t["cell_lon_east"].iloc[0])
    if abs(a[0] - la) < 0.01 and abs(a[1] - lo) < 0.01:
        return _ok()
    if abs(a[1] - (lo - 360)) < 0.01:
        return _hint("right cell, wrong convention: the file stores longitude in degrees east from 0 to 360.")
    return _hint("use grid.sel(lat=..., lon=..., method='nearest') with the station's longitude converted to 0-360.")


def _c_q4_region(a):
    return _ok() if _letter(a) == "A" else _hint(
        "map the seasonal range (calendar-month maximum minus minimum of the 1991-2020 climatology). Where does "
        "most precipitation fall in winter and almost none in summer?")


def _c_q4_bad(a):
    g = open_grid(GRID_FILE)["soilw"]
    bad = (g.max("time") < 1.0).to_series()
    cells = [(round(float(la), 2), round(float(lo), 2)) for la, lo in bad[bad].index]
    la, lo = round(float(a[0]), 2), round(float(a[1]) % 360, 2)
    if (la, lo) in cells:
        return _ok()
    return _hint("map the 1991-2020 maximum of soil water per cell (grid.soilw.max('time')), look for land cells "
                 "that never hold even 1 mm, and compare with their precipitation.")


def _c_q5a(a):
    if len(a) >= 3 and all(_words(x) >= 3 for x in a):
        return _ok("recorded. Keep this list next to you while you hunt")
    return _hint("write at least three, each in a few words: what fails, and where in the chain (gauge, "
                 "analysis, model run, file writing).")


def _c_q5c(a):
    d_truth, d_file, d_clean = (float(x) for x in a)
    print(f"drought months 2011-2020: truth {d_truth:.1f} %, file as received {d_file:.1f} %, "
          f"after your cleaning {d_clean:.1f} %")
    if abs(d_clean - d_file) < 1e-9:
        print("my_clean: " + NOT_ANSWERED + " (your cleaning has not changed the drought frequency)")
        _STATE["derived"] = (d_truth, d_file, None)
        return None
    _STATE["derived"] = (d_truth, d_file, d_clean)
    closer = abs(d_clean - d_truth) < abs(d_file - d_truth)
    print(f"distance to the truth: {abs(d_file - d_truth):.1f} points before cleaning, {abs(d_clean - d_truth):.1f} "
          f"after ({'closer' if closer else 'further'}). Part (d) asks why.")
    return True


def _c_text(a):
    if _words(a) >= 20:
        return _ok("recorded")
    return _hint("a few sentences, at least 20 words.")


def _c_q6_sign(a):
    return _ok() if int(np.sign(a)) == 1 else _hint(
        "more rain this month puts more water in the bucket. Which sign does that give?")


def _c_q6_lag(a):
    k, _ = _best_lag()
    return _ok() if int(a) == k else _hint(
        "compute the anomaly correlation for shifts of 0, 1, 2 and 3 months with "
        "a['soilw_mm'].corr(a['precip_mm_per_day'].shift(k)) and take the largest.")


def _c_q6_neff(a):
    v = _neff()
    s = spokane_anomalies()["soilw_mm"]
    r = s.autocorr(1)
    v2 = len(s) * (1 - r * r) / (1 + r * r)
    if abs(float(a[0]) - v) <= 0.1 * v and abs(float(a[1]) - v2) <= 0.1 * v2:
        return _ok()
    return _hint("N_eff = N (1 - r1x r1y) / (1 + r1x r1y), with r1 the lag-one autocorrelation of each anomaly "
                 "series and N = 360. For the second number use the soil's r1 for both series.")


def _c_q6_meaning(a):
    return _ok() if _letter(a) == "C" else _hint(
        "the soil water in this file is computed from the precipitation in this file. What does a correlation "
        "between the input and the output of a model measure?")


PIPELINE_LEARNED = r"anomal|percentile|threshold|climatolog|normali[sz]|standardi[sz]|z-score|quantile"
PIPELINE_BALANCE = r"balanc|weight|resampl|oversampl|undersampl|smote"


def _c_q7_pipeline(a):
    s = [str(x).lower() for x in a]
    if len(s) < 6:
        return _hint("at least six steps, from opening the raw file to the table a model learns from.")
    split = next((i for i, x in enumerate(s) if re.search(r"\bsplit", x)), None)
    learned = [i for i, x in enumerate(s) if re.search(PIPELINE_LEARNED, x)]
    balance = [i for i, x in enumerate(s) if re.search(PIPELINE_BALANCE, x)]
    if split is None:
        return _hint("where do you separate training and test data, and by what (time, space)?")
    if learned and min(learned) < split:
        return _hint("anomalies, percentiles and drought thresholds are learned from the data. Computed before the "
                     "split, they carry the test years into training. Where must they go, and from which rows?")
    if balance and min(balance) < split:
        return _hint("balancing or weighting before the split lets test information into training.")
    if not balance:
        return _hint("add the step where you deal with the rare class or the rare extremes, and place it.")
    return _ok("correct order on three points (a split exists; anomalies, thresholds and balancing come after "
               "it). Compare the rest with the worked solution")


def _c_q7_task(a):
    task, rate = a
    r = _real_test_rate()
    ok_task = _first_word(task).startswith("class")
    if not ok_task and _first_word(task).startswith("regress"):
        print("regression is also defensible (emulating the model's soil water); the guiding question asks about "
              "drought months, so this notebook follows classification.")
        ok_task = True
    if not ok_task:
        return _hint("the target is 'drought month or not'. Which task has a yes or no target?")
    rate = float(rate)
    if abs(rate - r) <= 0.3:
        return _ok()
    if abs(100 * rate - r) <= 0.3:
        return _hint("the mean of 0/1 labels is a fraction; give the rate in percent (multiply by 100).")
    y, _ = lab.real_drought_labels()
    g = open_grid(GRID_FILE)
    ok = (xr.open_dataset(lab.STATIC_FILE)["landperc"] >= 50) & (g["soilw"].max("time") >= 1)
    if abs(rate - float(100 * y.where(ok).sel(time=slice("2011", "2020")).mean())) <= 0.3:
        return _hint("that is the rate over cells with at least 50 % land. The question asks for all model land "
                     "cells: use labels as they are, without the land-percent filter.")
    return _hint("compute the mean of the drought labels over all model land cells in 2011-2020, in percent.")


def _c_q7_leak(a):
    return _ok() if _letter(a) == "A" else _hint(
        "neighbouring months and neighbouring cells carry almost the same soil water. Which split puts "
        "near-copies of test rows in training?")


def _c_wrap(a):
    _STATE["wrap"] = [str(x) for x in a]
    if len(a) == 3 and all(_words(x) >= 20 for x in a):
        return _ok("recorded")
    return _hint("three answers, two to four sentences each (at least 20 words).")


# --------------------------------------------------------------------------------------------------
# "Your turn" cells: the reference results, their expected form, and the checks
# --------------------------------------------------------------------------------------------------
USING_REFERENCE = "Using the reference result so you can continue; compare with yours."
TABLE_COLUMNS = ("month", "cell_lat", "cell_lon_east", "soilw_mm", "precip_mm_per_day", "soilw_ltm_1991_2020_mm")
ANOM_COLUMNS = ["soilw_mm", "precip_mm_per_day"]


def _ref_series():
    t = spokane_table()
    dates = pd.to_datetime(t["month"])
    return dates, t["precip_mm_per_day"] * dates.dt.days_in_month


def _ref_clim():
    clim = open_grid(GRID_FILE)["soilw"].groupby("time.month").mean()
    return clim, clim.max("month") - clim.min("month")


def _ref_split(yy, test_blocks):
    year, inside = yy["time"].dt.year, yy["block"].isin(test_blocks)
    train, test = yy[(year <= 2010) & ~inside], yy[(year >= 2011) & inside]
    return train, test, float((train["drought"] == 0).sum() / train["drought"].sum())


def _is_table(x, cols, n=None):
    return isinstance(x, pd.DataFrame) and all(c in x.columns for c in cols) and (n is None or len(x) == n)


def _form_series(x):
    dates, total = x
    return (isinstance(dates, pd.Series) and pd.api.types.is_datetime64_any_dtype(dates) and len(dates) == 360
            and len(total) == 360 and np.isfinite(np.asarray(total, dtype=float)).all())


def _form_clim(x):
    clim, srange = x
    return (isinstance(clim, xr.DataArray) and set(clim.dims) == {"month", "lat", "lon"}
            and clim.sizes["month"] == 12 and isinstance(srange, xr.DataArray) and set(srange.dims) == {"lat", "lon"})


def _form_split(x):
    train, test, w1 = x
    return (_is_table(train, ["drought", "time", "block"]) and _is_table(test, ["drought", "time", "block"])
            and len(train) > 0 and len(test) > 0 and np.isfinite(float(w1)) and float(w1) > 0)


_FORM = {
    "t": lambda x: _is_table(x, TABLE_COLUMNS, 360),
    "p3_series": _form_series,
    "p4_clim": _form_clim,
    "a": lambda x: _is_table(x, ANOM_COLUMNS, 360),
    "p7_split": _form_split,
}
_REFERENCE = {
    "t": lambda: spokane_table(),
    "p3_series": _ref_series,
    "p4_clim": _ref_clim,
    "a": lambda: spokane_anomalies()[ANOM_COLUMNS],
    "p7_split": _ref_split,
}


def use_or_reference(value, name, *context):
    """Return your result when it has the expected form; otherwise say so and return the reference result."""
    try:
        ok = bool(_FORM[name](value))
    except Exception:  # noqa: BLE001  None, a wrong type or a wrong shape all mean "use the reference"
        ok = False
    if ok:
        return value
    print(USING_REFERENCE)
    return _REFERENCE[name](*context)


def _c_p2_table(a):
    t, n = a
    ref = spokane_table()
    if not isinstance(t, pd.DataFrame):
        return _hint("t should be the DataFrame that pd.read_csv returns.")
    if t.shape != ref.shape:
        return _hint(f"your t has shape {t.shape}; read the whole file with pd.read_csv(lab.TABLE_FILE) and no "
                     "other arguments.")
    if not isinstance(n, pd.Series):
        return _hint("n_missing should be a Series with one count per column: t.isna() marks the missing "
                     "values and .sum() counts them in each column.")
    if not n.reindex(ref.columns).equals(ref.isna().sum()):
        return _hint("count per column on the table as read: t.isna().sum().")
    return _ok()


def _c_p2_driest(a):
    if str(a)[:7] == _driest_month():
        return _ok()
    return _hint("t['soilw_mm'].idxmin() gives the row label of the smallest value; read the 'month' column at "
                 "that row with t.loc[i, 'month'].")


def _c_p3_series(a):
    dates, total = a
    rd, rp = _ref_series()
    if not isinstance(dates, pd.Series) or not pd.api.types.is_datetime64_any_dtype(dates):
        return _hint("dates should be a Series of datetimes: pd.to_datetime(t['month']).")
    if len(dates) != len(rd) or not (dates.values == rd.values).all():
        return _hint("one date per row of t, in the same order.")
    if total is None or len(total) != len(rp):
        return _hint("precip_total needs one value per month: the rate (mm/day) times the days in that month.")
    if not np.allclose(np.asarray(total, dtype=float), rp.values, rtol=1e-6):
        return _hint("multiply each month's rate by its own number of days, dates.dt.days_in_month, not by a "
                     "fixed number.")
    return _ok()


def _c_p3_monthly(a, u):
    monthly, n = a
    col = u["SOIL_MOISTURE_5_DAILY"]
    r_mean, r_n = col.resample("MS").mean(), col.resample("MS").count()
    if not isinstance(monthly, pd.Series) or len(monthly) != len(r_mean):
        return _hint("sm5_monthly needs one value per calendar month of u: resample('MS') groups the days by "
                     "month start, then .mean().")
    if not np.allclose(np.asarray(monthly, dtype=float), r_mean.values, equal_nan=True, atol=1e-6):
        return _hint("average the 5 cm column (SOIL_MOISTURE_5_DAILY) of u, the table with the codes already NaN.")
    if not isinstance(n, pd.Series) or len(n) != len(r_n) or not (np.asarray(n) == r_n.values).all():
        return _hint("n_valid: the same resample with .count(), which counts the days that have a value.")
    return _ok()


def _c_p4_clim(a):
    clim, srange = a
    rc, rs = _ref_clim()
    if not isinstance(clim, xr.DataArray) or "month" not in clim.dims or clim.sizes["month"] != 12:
        return _hint("clim should hold 12 maps, one per calendar month: grid['soilw'].groupby('time.month').mean().")
    if set(clim.dims) != {"month", "lat", "lon"}:
        return _hint("average over time only, so that lat and lon stay.")
    if not np.allclose(clim.transpose(*rc.dims).values, rc.values, equal_nan=True, atol=1e-3):
        return _hint("average each calendar month over all 30 years of grid['soilw'].")
    if not isinstance(srange, xr.DataArray) or set(srange.dims) != {"lat", "lon"}:
        return _hint("srange is one map: reduce clim over the 'month' dimension.")
    if not np.allclose(srange.transpose(*rs.dims).values, rs.values, equal_nan=True, atol=1e-3):
        return _hint("the seasonal range at each cell is the largest of the 12 monthly means minus the smallest.")
    return _ok()


def _c_p5_counts(a, d):
    rows, dups, med = a
    dd = d.copy()
    dd["time"] = pd.to_datetime(dd["time"])
    r_rows = dd.groupby("time").size()
    if not isinstance(rows, pd.Series) or len(rows) != len(r_rows) or not (rows.values == r_rows.values).all():
        return _hint("rows_per_month: group the rows of d by 'time' and count them with .size().")
    n_dup = int(dd.duplicated(["time", "lat", "lon"], keep=False).sum())
    if not isinstance(dups, pd.DataFrame) or len(dups) != n_dup:
        return _hint("dups: d.duplicated(['time', 'lat', 'lon'], keep=False) is True for every copy of a repeated "
                     "row; select those rows of d.")
    r_med = dd.groupby("time")["soil_moisture_mm"].median()
    if not isinstance(med, pd.Series) or len(med) != len(r_med) or not np.allclose(
            np.asarray(med, dtype=float), r_med.values, equal_nan=True):
        return _hint("med: group by 'time' and take the median of 'soil_moisture_mm'.")
    return _ok("correct. Now read them against what you expect of a monthly grid")


def _c_p6_anom(a):
    ref = spokane_anomalies()[ANOM_COLUMNS]
    if not _is_table(a, ANOM_COLUMNS):
        return _hint("a should be a DataFrame with the columns soilw_mm and precip_mm_per_day.")
    if len(a) != len(ref):
        return _hint("one row per month of t (360 rows).")
    if np.allclose(a[ANOM_COLUMNS].values, ref.values, atol=1e-6):
        return _ok()
    t = spokane_table()[ANOM_COLUMNS]
    if np.allclose(a[ANOM_COLUMNS].values, (t - t.mean()).values, atol=1e-6):
        return _hint("you removed one overall mean. Remove the mean of each calendar month instead: group by the "
                     "month number, 1 to 12.")
    return _hint("from each value subtract the mean of all values in the same calendar month: "
                 "t[cols].groupby(cal).transform('mean') gives that mean on every row.")


def _c_p6_leads(a):
    _, ref = _best_lag()
    r = [float(x) for x in a]
    if len(r) != 4:
        return _hint("four correlations, for leads of 0, 1, 2 and 3 months.")
    if np.allclose(r, ref, atol=0.005):
        return _ok()
    an = spokane_anomalies()
    back = [an["soilw_mm"].corr(an["precip_mm_per_day"].shift(-k)) for k in range(4)]
    if np.allclose(r, back, atol=0.005):
        return _hint("the shift goes the other way: .shift(k) with k = 0, 1, 2, 3 puts the precipitation of k "
                     "months earlier on each row.")
    return _hint("correlate the anomalies a['soilw_mm'] and a['precip_mm_per_day'].shift(k), not the raw series.")


def _c_p7_split(a, yy, test_blocks):
    train, test, w1 = a
    rt, rs, rw = _ref_split(yy, test_blocks)
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        return _hint("train and test should be DataFrames: rows of yy selected with a True/False condition.")
    if len(train) != len(rt):
        return _hint("train: rows with a year up to 2010 AND a block not in test_blocks (combine with & and ~).")
    if len(test) != len(rs):
        return _hint("test: rows with a year from 2011 AND a block in test_blocks.")
    if abs(float(w1) - rw) > 1e-6 * rw:
        if abs(float(w1) - 1 / rw) < 1e-6:
            return _hint("the weight should raise the rare class: non-drought rows divided by drought rows.")
        return _hint("count drought (1) and non-drought (0) rows in train only, then divide.")
    return _ok()


_CHECKS = {
    "q0_peak_month": _c_q0, "q1_same_bytes": _c_q1_same, "q1_demo": _c_q1_demo, "q1_cell_km": _c_q1_km,
    "q1_raw_or_derived": _c_q1_raw, "q1_gauges": _c_q1_gauges, "q2_limits": _c_q2_limits,
    "q2_extreme": _c_q2_extreme, "q2_sentinel": _c_q2_sentinel, "q3_interval": _c_q3_interval,
    "q3_frozen": _c_q3_frozen, "q4_lat_order": _c_q4_order, "q4_cell": _c_q4_cell, "q4_region": _c_q4_region,
    "q4_bad_cell": _c_q4_bad, "q5a": _c_q5a, "q5c": _c_q5c, "q5d": _c_text, "q6_sign": _c_q6_sign,
    "q6_lag": _c_q6_lag, "q6_neff": _c_q6_neff, "q6_meaning": _c_q6_meaning, "q7_pipeline": _c_q7_pipeline,
    "q7_task": _c_q7_task, "q7_leak": _c_q7_leak, "q8": _c_text, "wrap": _c_wrap,
    "p2_table": _c_p2_table, "p2_driest": _c_p2_driest, "p3_series": _c_p3_series, "p3_monthly": _c_p3_monthly, "p4_clim": _c_p4_clim,
    "p5_counts": _c_p5_counts, "p6_anom": _c_p6_anom, "p6_leads": _c_p6_leads, "p7_split": _c_p7_split,
}


def _earlier_q1_gauges(a):
    v0, v1 = lab.gauge_totals()
    return abs(a[0] - v0) <= 0.05 * v0 + 1 and abs(a[1] - v1) <= 0.05 * v1 + 1


def _earlier_q7_task(a):
    task, rate = a
    w = _first_word(task)
    return (w.startswith("class") or w.startswith("regress")) and abs(float(rate) - _real_test_rate()) <= 1.0


# The rules of the earlier version, kept silently: they only count in the summary of an earlier copy.
_EARLIER_RULES = {"q1_gauges": _earlier_q1_gauges, "q7_task": _earlier_q7_task}


def _run(qid, answer, *context):
    if _blank(answer) or (isinstance(answer, tuple) and all(x is None for x in answer)):
        print(NOT_ANSWERED)
        return None
    try:
        return _CHECKS[qid](answer, *context)
    except Exception as e:  # noqa: BLE001  a malformed answer should give a hint, not a traceback
        return _hint(f"the answer has an unexpected form ({type(e).__name__}: {e}). Look at the example "
                     "format in the cell.")


def check(qid, answer, *context):
    """Self-check for question `qid`: prints 'correct', a hint or 'not answered yet', and records the result.

    `context` passes what a Your-turn check needs besides your result (your twin's table, the test blocks).
    """
    res = _run(qid, answer, *context)
    _ANSWERED[qid] = res is not None
    _PASSED[qid] = bool(res)
    if qid in _EARLIER_RULES:
        try:
            _PASSED_EARLIER[qid] = res is not None and bool(_EARLIER_RULES[qid](answer))
        except Exception:  # noqa: BLE001  a malformed answer did not pass the earlier rule either
            _PASSED_EARLIER[qid] = False
    if qid == "wrap" and _blank(answer):
        _STATE["wrap"] = None


def check_worked(qid, answer, *context):
    """The same check for the worked solutions: prints, records nothing."""
    saved = dict(_STATE)
    _run(qid, answer, *context)
    _STATE.update(saved)


def score_findings(findings, twin, reveal=False, who="you"):
    """Score findings against the twin (lab_s07_lsm_synthetic.score_findings) and record the result.

    who="you" records Part 5 (b); any other name records Part 8 (the AI tool's findings).
    """
    qid = "q5b" if who == "you" else "q8a"
    if not findings:
        print(NOT_ANSWERED)
        _ANSWERED[qid] = False
        return
    lab.score_findings(findings, twin, reveal=reveal, who=who)
    _ANSWERED[qid] = True
    if qid == "q5b":
        _STATE["found"] = (lab.count_found(findings, twin), len(twin._planted))


# --------------------------------------------------------------------------------------------------
# The summary the instructor reads
# --------------------------------------------------------------------------------------------------
def _short(s, n=300):
    s = " ".join(str(s or "").split())
    return s[:n] if s else "not answered"


def _is_new_version(ns):
    """True for the updated notebook: its first cell sets LAB_VERSION, it defines a Your-turn name, or it ran
    a Your-turn check. A copy of the earlier notebook does none of these."""
    return (ns.get(VERSION_MARKER) is not None or any(k in ns for k in _NEW_VERSION_NAMES)
            or any(q in _ANSWERED for q in YOUR_TURN_QIDS))


def _caller_globals():
    try:
        return sys._getframe(2).f_globals   # the notebook that called print_summary
    except (AttributeError, ValueError):
        return {}


def print_summary(netid=""):
    """Print the Session 7 summary block from the answers and self-checks run so far."""
    netid = str(netid or "").strip()
    new = _is_new_version(_caller_globals())
    parts_def, checked = (PARTS, SELF_CHECKED) if new else (PARTS_EARLIER, SELF_CHECKED_EARLIER)
    parts = " | ".join(f"P{p} {sum(_ANSWERED.get(q, False) for q in qs)}/{len(qs)}" for p, qs in parts_def.items())
    passed = sum((_PASSED if new else {**_PASSED, **_PASSED_EARLIER}).get(q, False) for q in checked)
    seed = lab.seed_from_netid(netid)
    found = _STATE["found"]
    found_s = f"{found[0]} of {found[1]}" if found else "not answered"
    der = _STATE["derived"]
    if der is None:
        der_s = "not answered"
    else:
        der_s = " / ".join("not answered" if x is None else f"{x:.1f}" for x in der) + " %"
    wrap = _STATE["wrap"] or ["", "", ""]
    wrap = (list(wrap) + ["", "", ""])[:3]
    lines = [
        "==== SESSION 7 SUMMARY ====",
        f"track: {TRACK}",
        f"netid: {netid or 'not answered'}",
        f"parts answered: {parts}",
        f"self-checks passed: {passed} of {len(checked)}",
    ] + ([] if new else ["notebook version: earlier (before 14 October, afternoon); parts and self-checks "
                         "counted as in that version"]) + [
        f"planted problems found: {found_s} (seed {seed})",
        f"derived quantity (truth / as given / after cleaning): {der_s}",
    ] + [f"reflection {i + 1}: {_short(w)}" for i, w in enumerate(wrap)] + ["==== END SUMMARY ===="]
    print("\n".join(lines))
