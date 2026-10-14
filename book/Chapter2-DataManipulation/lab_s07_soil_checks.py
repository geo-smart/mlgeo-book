"""Self-checks for lab_s07_soil_moisture.ipynb (Session 7, soil moisture track).

Each check prints one of three things:
  "not answered yet"            the answer still holds its placeholder (...) or an empty string;
  "correct"                     with one line on why it matters;
  "not yet. Hint: ..."          a pointer to what to look at, never the answer.
print_summary() reruns every check on the student's answers and prints the block the instructor grades.

Expected values are computed here from the course copy of the data
(snapshots/uscrn_WA_Spokane_17_SSW_2015-2024.txt) each time a check runs, so no answer is
typed into this file. Values quoted from documents (sensor counts, intervals) are taken from
the provider pages cited in the notebook.
"""
from __future__ import annotations

import contextlib
import io
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from lab_s07_soil_synthetic import (DEPTHS_CM, PROBLEM_TYPES, REAL_FILE, check_findings, rain_response,
                                    read_uscrn_daily, score_findings)

# Quoted from https://www.ncei.noaa.gov/access/crn/measurements.html (retrieved 2026-10-14):
# "a total of 15 Stevens Water Monitoring Systems, Inc., Hydra Probe II (SDI-12) units are placed in the
#  ground in three plots at five depths" and "All 15 probes are interrogated every two minutes and their
#  measurements averaged over 5-minute periods for output purposes."
DOC_PROBES_PER_DEPTH = 15 // 5
DOC_READING_INTERVAL_MIN = 2
# Daily file README note J: "an average of the day's hourly soil measurements".
DOC_FILE_INTERVAL_HOURS = 24


def _blank(x):
    if x is ... or x is None:
        return True
    if isinstance(x, str) and not x.strip():
        return True
    if isinstance(x, (list, tuple, dict)) and len(x) == 0:
        return True
    return False


_STATUS = {"last": None}                  # outcome of the last check run: "todo", "ok" or "hint"


def _todo():
    _STATUS["last"] = "todo"
    print("not answered yet")


def _ok(why=""):
    _STATUS["last"] = "ok"
    print("correct" + (f". {why}" if why else ""))


def _hint(where):
    _STATUS["last"] = "hint"
    print(f"not yet. Hint: {where}")


_NEGATION = {"not", "no", "never", "without", "cannot", "can't", "didn't", "isn't", "wasn't", "doesn't"}


def _mentions(text, words):
    """True when one of `words` appears in `text` with no negation among the three words before it."""
    t = text.lower()
    for w in words:
        for m in re.finditer(re.escape(w), t):
            before = [x.strip(".,;:()") for x in t[:m.start()].split()[-3:]]
            if not _NEGATION & set(before):
                return True
    return False


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def real():
    """The course copy, read with missing codes replaced by NaN (cached)."""
    return read_uscrn_daily()


def _gaps(series):
    m = series.isna()
    runs = (m != m.shift()).cumsum()
    out = [(grp.index[0], len(grp)) for _, grp in series[m].groupby(runs[m])]
    return sorted(out, key=lambda g: -g[1])


# ----------------------------------------------------------------------------- Part 0
def check_prediction(wettest_month, driest_month):
    if _blank(wettest_month) or _blank(driest_month):
        return _todo()
    for m in (wettest_month, driest_month):
        if _num(m) is None or not 1 <= _num(m) <= 12:
            return _hint("months as numbers from 1 (January) to 12 (December)")
    _STATUS["last"] = "ok"
    print(f"Recorded: wettest {int(wettest_month)}, driest {int(driest_month)}. Part 3 compares it with the data.")


# ----------------------------------------------------------------------------- Part 1
def check_n_rows(n_days):
    if _blank(n_days):
        return _todo()
    with open(REAL_FILE, "rb") as f:
        n_lines = sum(1 for _ in f)
    calendar = len(pd.date_range("2015-01-01", "2024-12-31"))
    if _num(n_days) == calendar == n_lines:
        _ok("One line per day, no day missing and none repeated. A file that ends early or repeats a day fails here.")
    else:
        _hint("count the days from 2015-01-01 to 2024-12-31 by hand; remember which years are leap years")


def check_download(my_copy_matches, comparison):
    if comparison is None:
        print("No download to compare (offline or skipped). You are working from the course copy, which is fine.")
        return
    if my_copy_matches is ...:
        return _todo()
    truth = bool(comparison.loc["same fingerprint"].iloc[0])
    if bool(my_copy_matches) == truth:
        _ok("Same fingerprint means the same bytes." if truth else
            "Different fingerprint: now check sizes, rows and which years differ before deciding why.")
    else:
        _hint("the row 'same fingerprint' of the comparison table")


def check_native(native):
    if _blank(native) or any(v is ... for v in native.values()):
        return _todo()
    wrong = []
    if _num(native.get("probes_per_depth")) != DOC_PROBES_PER_DEPTH:
        wrong.append("probes_per_depth: the measurements page says how many probes sit in how many plots")
    if _num(native.get("reading_interval_minutes")) != DOC_READING_INTERVAL_MIN:
        wrong.append("reading_interval_minutes: how often each probe is interrogated, before any averaging")
    if _num(native.get("file_interval_hours")) != DOC_FILE_INTERVAL_HOURS:
        wrong.append("file_interval_hours: README note J says what one value in this file averages")
    if str(native.get("raw_or_derived", "")).strip().lower() != "derived":
        wrong.append("raw_or_derived: count the steps between the dielectric reading and the number in the file")
    if wrong:
        for w in wrong:
            _hint(w)
    else:
        _ok("Each daily value is an average over three probes and many readings, converted from a dielectric "
            "measurement by a general equation: a derived product.")


def check_processing_text(text):
    if _blank(text):
        return _todo()
    t = text.lower()
    steps = {"conversion from dielectric": ("dielectric", "calibrat", "convert", "equation"),
             "averaging": ("averag", "mean"),
             "quality control": ("flag", "quality", "qc", "removed", "missing"),
             "combining probes": ("three probes", "3 probes", "probes", "plots")}
    found = [k for k, words in steps.items() if _mentions(t, words)]
    if len(found) >= 3:
        _ok(f"You named: {', '.join(found)}.")
    else:
        _hint("the quotes in Part 1: what turns a radio-wave reading into m3/m3, how many readings make one "
              "value, and what happens to a reading the provider flags")


# ----------------------------------------------------------------------------- Part 2
def check_row_meaning(text):
    if _blank(text):
        return _todo()
    t = text.lower()
    if ("day" in t) and any(w in t for w in ("station", "site", "turnbull", "spokane", "one place")):
        _ok("One row is one Local Standard Time day at one station.")
    else:
        _hint("what fields 1 and 2 are (README section 5), and whether the day is a UTC day or a local day")


def check_units(units, source):
    if _blank(units) or _blank(source):
        return _todo()
    u = units.replace(" ", "").lower()
    ok_u = u in ("m^3/m^3", "m3/m3", "m³/m³", "m^3m^-3", "m3m-3", "cm3/cm3", "cm^3/cm^3", "cm³/cm³")
    ok_s = any(w in source.lower() for w in ("header", "readme"))
    if ok_u and ok_s:
        _ok("A volume of water per volume of soil, a fraction between 0 and the soil's pore space.")
    elif not ok_u:
        _hint("the third line of uscrn_daily01_HEADERS.txt under SOIL_MOISTURE_5_DAILY")
    else:
        _hint("which provider file states the units: name it")


def check_missing_codes(sm_code, t_code):
    if _blank(sm_code) or _blank(t_code):
        return _todo()
    raw = pd.read_csv(REAL_FILE, sep=r"\s+", header=None)
    sm_ok = _num(sm_code) == raw[18].min()
    t_ok = _num(t_code) == raw[5].min()
    if sm_ok and t_ok:
        _ok("README note C: the lowest number the field's width can hold. Read as a number, it is an error.")
    else:
        _hint("README note C, and the field widths and decimals of fields 6 and 19")


def check_limits(limits):
    if _blank(limits) or any(v is ... or (isinstance(v, (tuple, list)) and any(x is ... for x in v))
                             for v in limits.values()):
        return _todo()
    problems = []
    lo, hi = limits.get("SOIL_MOISTURE_5_DAILY", (None, None))
    if _num(lo) is None or _num(lo) < 0:
        problems.append("soil moisture lower limit: can a soil hold less than no water?")
    if _num(hi) is None or _num(hi) >= 1:
        problems.append("soil moisture upper limit: water can only fill the pores, not the grains")
    plo, phi = limits.get("P_DAILY_CALC", (None, None))
    if _num(plo) != 0:
        problems.append("precipitation lower limit: what is the least a gauge can collect in a day?")
    if _num(phi) is None or _num(phi) <= 0:
        problems.append("precipitation upper limit: a positive number of mm")
    tlo, thi = limits.get("T_DAILY_MAX", (None, None))
    if _num(tlo) is None or _num(thi) is None or not _num(tlo) < _num(thi):
        problems.append("temperature limits: lower below upper")
    if problems:
        for p in problems:
            _hint(p)
    else:
        _ok("Recorded. Now count the values outside your limits.")


def check_extremes(hottest_is_real, sm5_min_as_read_is_real):
    if hottest_is_real is ... or sm5_min_as_read_is_real is ...:
        return _todo()
    msgs = []
    if hottest_is_real is not True:
        msgs.append("the hottest day: do the neighbouring days and the region's weather that week agree?")
    if sm5_min_as_read_is_real is not False:
        msgs.append("the minimum of SOIL_MOISTURE_5_DAILY as read, next to README note C")
    if msgs:
        for m in msgs:
            _hint(m)
    else:
        _ok("The hottest day sits in a run of hot days during a documented heat wave; -99.000 is the missing-value code.")


# ----------------------------------------------------------------------------- Part 3
def check_interval(days):
    if _blank(days):
        return _todo()
    step = real().index.to_series().diff().dropna().unique()
    if len(step) == 1 and _num(days) == step[0] / pd.Timedelta(days=1):
        _ok("Every row is one day after the previous one; gaps are blank values, not missing rows.")
    else:
        _hint("the unique differences between consecutive dates")


def check_gaps(n_missing, longest_gap_days):
    if _blank(n_missing) or _blank(longest_gap_days):
        return _todo()
    s = real()["SOIL_MOISTURE_5_DAILY"]
    ok1 = _num(n_missing) == int(s.isna().sum())
    ok2 = _num(longest_gap_days) == _gaps(s)[0][1]
    if ok1 and ok2:
        _ok("Now ask when the gaps happen.")
    elif not ok1:
        _hint("count blanks in the 5 cm column after replacing the missing-value code")
    else:
        _hint("runs of consecutive blank days; the longest run, in days")


def check_gap_causes(winter_gaps, gap_2023):
    if _blank(winter_gaps) or _blank(gap_2023):
        return _todo()
    msgs = []
    if not _mentions(str(winter_gaps), ("frozen", "freez")):
        msgs.append("the 5 cm soil temperature on the days the 5 cm moisture is blank")
    if not _mentions(str(gap_2023), ("outage", "logger", "station", "power", "maintenance")):
        msgs.append("in November and December 2023, which columns are blank, which are not, and CRX_VN")
    if msgs:
        for m in msgs:
            _hint(m)
    else:
        _ok("Two causes, two different fixes: a frozen probe is a limit of the method; an outage is lost data.")


def check_prediction_vs_data(wettest_month, driest_month):
    if _blank(wettest_month) or _blank(driest_month):
        return _todo()
    s = real()["SOIL_MOISTURE_5_DAILY"]
    clim = s.groupby(s.index.month).mean()
    near = 0.01                                    # m3/m3: months closer than this are not told apart here
    wet = sorted(int(m) for m in clim.index[clim >= clim.max() - near])
    dry = sorted(int(m) for m in clim.index[clim <= clim.min() + near])
    if int(wettest_month) in wet and int(driest_month) in dry:
        _ok(f"Your Part 0 prediction matches the monthly means. Months {wet} all lie within {near} m3/m3 of the "
            "wettest: a late-winter plateau, not one wettest month. And those winter means rest only on the "
            "unfrozen days, because frozen days are blank.")
    else:
        print("Your Part 0 prediction differs from the monthly means of the 5 cm series. Plot those means, "
              "with how many days each one rests on, and write one sentence on what you had not expected.")


# ----------------------------------------------------------------------------- Part 4
def check_axes(depth_spacing_uniform, time_step_days):
    if depth_spacing_uniform is ... or _blank(time_step_days):
        return _todo()
    uniform = len(set(np.diff(DEPTHS_CM))) == 1
    if bool(depth_spacing_uniform) == uniform and _num(time_step_days) == 1:
        _ok("Five probes at 5, 10, 20, 50 and 100 cm: an image drawn with equal rows would stretch the top.")
    else:
        _hint("the differences between the depth coordinates, and the time step from Part 3")


def arrival_dates(year=2021, rise=0.05):
    """First day after 1 September when each depth is `rise` m3/m3 above its August-September minimum."""
    out = {}
    for z in DEPTHS_CM:
        s = real()[f"SOIL_MOISTURE_{z}_DAILY"][f"{year}-08-01":f"{year + 1}-03-31"]
        base = s[:f"{year}-09-30"].min()
        hit = s[f"{year}-09-01":][s[f"{year}-09-01":] > base + rise]
        out[z] = hit.index[0] if len(hit) else pd.NaT
    return pd.Series(out, name="arrival")


def check_arrival(delay_5_to_100_days, year=2021):
    if _blank(delay_5_to_100_days):
        return _todo()
    a = arrival_dates(year)
    expected = (a[100] - a[5]).days
    if abs(_num(delay_5_to_100_days) - expected) <= 1:
        other = {y: (arrival_dates(y)[100] - arrival_dates(y)[5]).days for y in (2016, 2017)}
        _ok(f"In {year} the front took weeks to reach 1 m. Other autumns differ: 2016 took {other[2016]} days, "
            f"and in 2017 ({other[2017]} days from 5 cm) the 20, 50 and 100 cm probes wetted on the same day, "
            "during one storm. Run arrival_dates(2016) and arrival_dates(2017).")
    else:
        _hint(f"run arrival_dates({year}) and subtract the 5 cm date from the 100 cm date")


def check_variability(day_to_day_change_decreases_with_depth):
    if day_to_day_change_decreases_with_depth is ...:
        return _todo()
    ch = [real()[f"SOIL_MOISTURE_{z}_DAILY"].diff().abs().mean() for z in DEPTHS_CM]
    truth = all(a > b for a, b in zip(ch, ch[1:]))
    if bool(day_to_day_change_decreases_with_depth) == truth:
        _ok("Soil above a probe smooths the rain signal it receives.")
    else:
        _hint("the mean absolute day-to-day change at each depth, printed above")


# ----------------------------------------------------------------------------- Part 5
_FAMILIES = {
    "spike": ("spike", "jump", "glitch", "lightning", "interference", "noise"),
    "drift": ("drift", "aging", "ageing", "trend", "slow"),
    "stuck": ("stuck", "flat", "repeat", "frozen value", "constant"),
    "frozen_read_as_dry": ("frozen", "freez", "ice", "winter"),
    "missing_code_left_in": ("-99", "sentinel", "missing code", "missing-value", "fill value", "9999"),
    "gauge_outage_as_zero": ("gauge", "zero rain", "0 mm", "clogg", "outage"),
    "rain_one_day_late": ("shift", "offset", "late", "utc", "time zone", "timezone", "one day"),
    "probes_swapped": ("swap", "mislabel", "wrong depth", "exchanged", "switched"),
    "duplicated_days": ("duplicate", "twice", "repeated row", "repeated day"),
    "units_percent": ("percent", "%", "units", "unit change", "scale"),
}


def check_expected_problems(text):
    if _blank(text):
        return _todo()
    t = text.lower()
    named = [k for k, words in _FAMILIES.items() if any(w in t for w in words)]
    print(f"You named {len(named)} of the ten kinds this generator can plant.")
    if len(named) < 4:
        _hint("think through one year of the station's life: winter at 5 cm, a technician's visit, the "
              "logger's clock, the rain gauge in a storm, and the step that writes the table")
    else:
        _ok("Now hunt for them in your twin.")


def check_three_values(r_truth, r_contaminated, r_cleaned, twin, clean_fn):
    if any(_blank(x) for x in (r_truth, r_contaminated, r_cleaned)):
        return _todo()
    exp_truth = rain_response(twin.truth)[0]
    exp_cont = rain_response(twin.observed)[0]
    exp_clean = rain_response(clean_fn(twin.observed.copy()))[0]
    msgs = []
    if _num(r_truth) is None or not np.isclose(_num(r_truth), exp_truth, rtol=0.02):
        msgs.append(f"r_truth: rain_response(twin.truth)[0], the slope of the 5 cm change on rain, per 10 mm")
    if _num(r_contaminated) is None or not np.isclose(_num(r_contaminated), exp_cont, rtol=0.02):
        msgs.append("r_contaminated: rain_response(twin.observed)[0], on the table exactly as delivered")
    if _num(r_cleaned) is None or not np.isclose(_num(r_cleaned), exp_clean, rtol=0.02, equal_nan=True):
        msgs.append("r_cleaned: rain_response(clean(twin.observed))[0], after your cleaning")
    if msgs:
        for m in msgs:
            _hint(m)
        return
    err_c = abs(exp_cont - exp_truth) / abs(exp_truth) * 100
    err_k = abs(exp_clean - exp_truth) / abs(exp_truth) * 100
    _STATUS["last"] = "ok"
    print(f"correct. Off from the truth: as delivered {err_c:.0f}%, cleaned {err_k:.0f}%.")
    if err_k > 10:
        print("Your cleaning leaves more than 10% error: which of your planted problems touch the 5 cm column "
              "or the precipitation column?")


def check_reflection(text, min_words=40):
    if _blank(text):
        return _todo()
    n = len(text.split())
    named = [k for k in PROBLEM_TYPES if k.replace("_", " ") in text.lower() or k in text.lower()]
    if n < min_words:
        _hint(f"write at least {min_words} words; you have {n}")
    elif not named:
        _hint("name the problems you discuss with the names in PROBLEM_TYPES")
    else:
        _ok(f"{n} words, problems named: {', '.join(named)}.")


# ----------------------------------------------------------------------------- Part 6
def check_signs(rain_vs_change_sign, temp_vs_moisture_sign):
    if _blank(rain_vs_change_sign) or _blank(temp_vs_moisture_sign):
        return _todo()
    if str(rain_vs_change_sign).strip() == "+" and str(temp_vs_moisture_sign).strip() == "-":
        _ok("Recorded. Compare it with the correlations below.")
    else:
        _hint("what rain does to the water content of the top 5 cm that day; and whether warm months are "
              "wet or dry at this site (Part 3)")


def lag_table(df=None, lags=range(-2, 4)):
    """Pearson and Spearman correlation of daily precipitation on day t with the 5 cm change on day t + lag.

    Uses every day on which the 5 cm soil is above 0 C on day t + lag and the day before, with or without
    precipitation; most of these days have none, and they all share the lowest rank in Spearman's correlation.
    """
    from scipy import stats
    d = real() if df is None else df
    p = d["P_DAILY_CALC"]
    ch = d["SOIL_MOISTURE_5_DAILY"].diff()
    warm = (d["SOIL_TEMP_5_DAILY"] > 0) & (d["SOIL_TEMP_5_DAILY"].shift(1) > 0)
    rows = []
    for lag in lags:
        y = ch.shift(-lag)
        ok = warm.shift(-lag, fill_value=False) & p.notna() & y.notna()
        rows.append({"lag_days": lag, "n_days": int(ok.sum()),
                     "pearson": stats.pearsonr(p[ok], y[ok])[0], "spearman": stats.spearmanr(p[ok], y[ok])[0]})
    return pd.DataFrame(rows).set_index("lag_days")


def check_lag(lag_of_max, second_lag):
    if _blank(lag_of_max) or _blank(second_lag):
        return _todo()
    t = lag_table()["pearson"]
    order = t.sort_values(ascending=False).index
    if _num(lag_of_max) == order[0] and _num(second_lag) == order[1]:
        _ok("Rain that falls late in a Local Standard Time day shows up in the next day's average.")
    else:
        _hint("the pearson column of the lag table: the largest value, then the second largest")


def check_seasonal_trap(r_raw, r_anomaly, r_raw_expected, r_anom_expected):
    if _blank(r_raw) or _blank(r_anomaly):
        return _todo()
    if np.isclose(_num(r_raw), r_raw_expected, atol=0.01) and np.isclose(_num(r_anomaly), r_anom_expected, atol=0.01):
        _ok("Most of the raw correlation is the shared seasonal cycle, not day-to-day coupling.")
    else:
        _hint("Pearson r of temperature and 5 cm moisture before and after removing the day-of-year means")


def check_neff(n_eff, expected):
    if _blank(n_eff):
        return _todo()
    if _num(n_eff) is not None and abs(_num(n_eff) - expected) <= 0.05 * expected:
        _ok("Thousands of days, but only a few hundred independent pieces of information.")
    else:
        _hint("n_eff = n (1 - r1 r2) / (1 + r1 r2) with the lag-1 autocorrelations of the two anomaly series")


def check_significance(differs_from_zero, r_anomaly, n_eff_expected):
    if differs_from_zero is ... or _blank(differs_from_zero):
        return _todo()
    truth = abs(r_anomaly) > 2 / np.sqrt(n_eff_expected)
    if bool(differs_from_zero) == truth:
        _ok(f"|r| = {abs(r_anomaly):.2f} against 2/sqrt(n_eff) = {2 / np.sqrt(n_eff_expected):.2f}: "
            + ("the day-to-day coupling is real but small." if truth else
               "once the season is removed, a warm day says nothing about that day's soil moisture."))
    else:
        _hint("compare |r| of the anomalies with 2 / sqrt(n_eff)")


# ----------------------------------------------------------------------------- Part 7
def check_pipeline(steps):
    if _blank(steps) or all(_blank(s) for s in steps):
        return _todo()
    text = [str(s).lower() for s in steps]

    def first(*words):
        for i, s in enumerate(text):
            if any(w in s for w in words):
                return i
        return None
    found = {"missing codes": first("-99", "missing code", "sentinel", "nan"),
             "frozen days": first("frozen", "soil temperature", "st_5"),
             "selection": first("select", "warm season", "keep", "period", "subset"),
             "split by time": first("split"),
             "balancing": first("balanc", "weight", "resampl", "stratif")}
    msgs = [f"a step about {k}" for k, v in found.items() if v is None]
    if found["split by time"] is not None and found["balancing"] is not None \
            and found["balancing"] < found["split by time"]:
        msgs.append("the order: balancing before the split lets the test set shape the training set")
    if len(steps) < 6:
        msgs.append(f"at least six steps (you have {len(steps)})")
    if msgs:
        for m in msgs:
            _hint(m)
    else:
        _ok("Every step has a place; check that each also has a reason.")


def check_task(task, target):
    if _blank(task) or _blank(target):
        return _todo()
    if str(task).strip().lower() not in ("regression", "classification", "forecasting", "detection", "clustering"):
        _hint("one of: regression, classification, forecasting, detection, clustering")
    elif len(str(target).split()) < 4:
        _hint("name the target: which column, at which depth, on which day relative to the inputs")
    else:
        _ok("Several tasks fit this question; the target you named decides which.")


def check_balance(answers, train_fraction_expected):
    if _blank(answers) or any(v is ... for v in answers.values()):
        return _todo()
    msgs = []
    if "time" not in str(answers.get("split_by", "")).lower():
        msgs.append("split_by: neighbouring days share weather; what keeps them out of both sets?")
    b = str(answers.get("balance_on", "")).strip().lower()
    if "train" not in b or any(w in b for w in ("test", "valid", "all", "both", "and")):
        msgs.append("balance_on: which set may the balancing look at? Only one of them.")
    metric = str(answers.get("metric_for_rare_class", "")).lower()
    if not any(w in metric for w in ("recall", "precision", "f1", "pr", "average precision")):
        msgs.append(f"metric_for_rare_class: with about {1 - train_fraction_expected:.0%} of training days in one "
                    "class, what does accuracy reward?")
    f = _num(answers.get("dry_fraction_train"))
    if f is None or abs(f - train_fraction_expected) > 0.005:
        msgs.append("dry_fraction_train: the share of training days in the dry class, as a fraction")
    if msgs:
        for m in msgs:
            _hint(m)
    else:
        _ok("And the base rate differs between the splits: a model trained on wetter years meets drier ones, "
            "and the 2023 outage removed wet-season days from the test years.")


# ----------------------------------------------------------------------------- Part 8
def check_ai_tool(ai_findings, twin, ai_response=...):
    if _blank(ai_findings):
        return _todo()
    check_findings(ai_findings, twin, show_missed=False, who="The AI tool", offer_missed=False)
    if _num(ai_response) is not None:
        truth = rain_response(twin.truth)[0]
        print(f"The tool's cleaned response is {abs(_num(ai_response) - truth) / abs(truth):.0%} from the truth.")
    t = twin.truth
    extremes = {"largest rain day": t.loc[t["precip_mm"].idxmax(), "date"],
                "hottest day": t.loc[t["t_max_c"].idxmax(), "date"]}
    for name, day in extremes.items():
        day = pd.Timestamp(day)
        for f in ai_findings:
            try:
                a, b = pd.Timestamp(f.get("start")), pd.Timestamp(f.get("end", f.get("start")))
            except (ValueError, TypeError):
                continue
            if a - pd.Timedelta(days=1) <= day <= b + pd.Timedelta(days=1) and f.get("column") in (
                    "precip_mm", "t_max_c", "any", "all"):
                print(f"The tool flagged the twin's {name} ({day.date()}), which is real in the truth.")


# ----------------------------------------------------------------------------- Wrap-up
def check_wrapup(answers):
    if _blank(answers) or any(_blank(a) for a in answers):
        return _todo()
    short = [i + 1 for i, a in enumerate(answers)
             if len(str(a).split()) < 20 or not any(ch.isdigit() for ch in str(a))]
    if short:
        _hint(f"answers {short}: two to four sentences (at least 20 words) each, with one number or one date from your work")
    else:
        _ok("Now run the summary cell at the end.")


# ----------------------------------------------------------------------------- Summary
# The answer variables of each part, grouped by question. A question counts as answered when all its
# variables hold something other than the placeholder.
PARTS = {
    "P0": [("predict_wettest_month", "predict_driest_month")],
    "P1": [("n_days_expected",), ("my_copy_matches",), ("native",), ("processing_steps",)],
    "P2": [("row_meaning",), ("sm_units", "units_source"), ("sm_missing_code", "t_missing_code"), ("limits",),
           ("hottest_is_real", "sm5_min_as_read_is_real")],
    "P3": [("sampling_interval_days",), ("n_missing_sm5", "longest_gap_days"),
           ("winter_gaps_cause", "gap_2023_cause"), ("prediction_note",)],
    "P4": [("depth_spacing_uniform", "time_step_days"), ("delay_5_to_100_days",), ("change_decreases_with_depth",)],
    "P5": [("expected_problems",), ("findings",), ("r_truth", "r_contaminated", "r_cleaned"), ("reflection_part5",)],
    "P6": [("rain_vs_change_sign", "temp_vs_moisture_sign"), ("lag_of_max", "second_lag"), ("r_raw", "r_anomaly"),
           ("n_eff", "anomaly_r_differs_from_zero")],
    "P7": [("pipeline_steps",), ("ml_task", "ml_target"), ("balance",)],
    "P8": [("ai_findings",), ("ai_comparison",)],
}


def _filled(x):
    """False for a placeholder: ..., None, an empty string, an empty list, or a container holding one."""
    if _blank(x):
        return False
    if isinstance(x, dict):
        return all(_filled(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_filled(v) for v in x) and not any(v is ... for v in x)
    return True


def _self_checks(ns):
    """(label, function) for every graded self-check, called on the student's own answers."""
    g = ns.get

    def neff_expected():
        r1, r2 = g("Ta").autocorr(1), g("Sa").autocorr(1)
        return int(g("both_a").sum()) * (1 - r1 * r2) / (1 + r1 * r2)
    return [
        ("1.1", lambda: check_n_rows(g("n_days_expected", ...))),
        ("1.3", lambda: check_native(g("native", ...))),
        ("1.4", lambda: check_processing_text(g("processing_steps", ""))),
        ("2.1", lambda: check_row_meaning(g("row_meaning", ""))),
        ("2.2", lambda: check_units(g("sm_units", ""), g("units_source", ""))),
        ("2.3", lambda: check_missing_codes(g("sm_missing_code", ...), g("t_missing_code", ...))),
        ("2.4", lambda: check_limits(g("limits", ...))),
        ("2.5", lambda: check_extremes(g("hottest_is_real", ...), g("sm5_min_as_read_is_real", ...))),
        ("3.1", lambda: check_interval(g("sampling_interval_days", ...))),
        ("3.2", lambda: check_gaps(g("n_missing_sm5", ...), g("longest_gap_days", ...))),
        ("3.3", lambda: check_gap_causes(g("winter_gaps_cause", ""), g("gap_2023_cause", ""))),
        ("4.1", lambda: check_axes(g("depth_spacing_uniform", ...), g("time_step_days", ...))),
        ("4.2", lambda: check_arrival(g("delay_5_to_100_days", ...))),
        ("4.3", lambda: check_variability(g("change_decreases_with_depth", ...))),
        ("5a", lambda: check_expected_problems(g("expected_problems", ""))),
        ("5c", lambda: check_three_values(g("r_truth", ...), g("r_contaminated", ...), g("r_cleaned", ...),
                                          g("twin"), g("clean"))),
        ("5d", lambda: check_reflection(g("reflection_part5", ""))),
        ("6.1", lambda: check_signs(g("rain_vs_change_sign", ""), g("temp_vs_moisture_sign", ""))),
        ("6.2", lambda: check_lag(g("lag_of_max", ...), g("second_lag", ...))),
        ("6.3", lambda: check_seasonal_trap(g("r_raw", ...), g("r_anomaly", ...), g("r_raw_true"), g("r_anom_true"))),
        ("6.4", lambda: check_neff(g("n_eff", ...), neff_expected())),
        ("6.4b", lambda: check_significance(g("anomaly_r_differs_from_zero", ...), g("r_anom_true"),
                                            neff_expected())),
        ("7.1", lambda: check_pipeline(g("pipeline_steps", []))),
        ("7.2", lambda: check_task(g("ml_task", ""), g("ml_target", ""))),
        ("7.3", lambda: check_balance(g("balance", ...), float(g("y_train").mean()))),
        ("wrap-up", lambda: check_wrapup(g("wrapup", []))),
    ]


def _short(text, n=300):
    t = " ".join(str(text).split())
    return t[:n] if t else "not answered"


def print_summary(ns):
    """Print the Session 7 summary block from the notebook's variables (pass globals())."""
    g = ns.get
    parts = []
    for p, questions in PARTS.items():
        # no download to compare: the question about your copy does not apply and is left out of the count
        asked = [q for q in questions if not (q == ("my_copy_matches",) and g("comparison") is None)]
        done = sum(all(_filled(g(v, ...)) for v in q) for q in asked)
        parts.append(f"{p} {done}/{len(asked)}")

    checks = _self_checks(ns)
    passed = 0
    for _, fn in checks:
        _STATUS["last"] = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
        except Exception:                                   # a missing variable or a cell not run counts as not passed
            _STATUS["last"] = None
        passed += _STATUS["last"] == "ok"

    twin, findings = g("twin"), g("findings")
    seed = twin.seed if twin is not None else "not run"
    if twin is not None and _filled(findings):
        n_found = score_findings(findings, twin)[0]
        planted = f"{n_found} of {twin.n_planted} (seed {seed})"
    else:
        planted = f"not answered (seed {seed})"

    def val(name):
        x = _num(g(name, ...)) if _filled(g(name, ...)) else None
        return f"{x:.4f}" if x is not None else "not answered"

    wrap = list(g("wrapup", [])) + ["", "", ""]
    lines = ["==== SESSION 7 SUMMARY ====",
             "track: soil",
             f"netid: {str(g('NETID') or '').strip() or 'not answered'}",
             "parts answered: " + " | ".join(parts),
             f"self-checks passed: {passed} of {len(checks)}",
             f"planted problems found: {planted}",
             "derived quantity (truth / as given / after cleaning): "
             f"{val('r_truth')} / {val('r_contaminated')} / {val('r_cleaned')} m3/m3 per 10 mm",
             f"reflection 1: {_short(wrap[0])}",
             f"reflection 2: {_short(wrap[1])}",
             f"reflection 3: {_short(wrap[2])}",
             "==== END SUMMARY ===="]
    print("\n".join(lines))
