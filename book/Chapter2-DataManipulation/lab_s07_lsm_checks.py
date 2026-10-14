"""Self-checks and the session summary for lab_s07_land_surface_model.ipynb (Session 7, land-surface track).

``check(qid, answer)`` prints "correct", a hint (never the answer) or "not answered yet", and remembers the
result. ``score_findings`` scores your Part 5 findings against your twin and remembers the count.
``print_summary(NETID)`` prints the block the instructor reads on Canvas.

Expected values are computed from the course copies in snapshots/ each time a check runs; no answer is typed
into this file. The worked solutions call ``check_worked``, which prints the same way but records nothing, so
opening a solution never changes your summary.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import lab_s07_lsm_synthetic as lab
from lab_s07_lsm_synthetic import GRID_FILE, NOT_ANSWERED, W_MAX, open_grid, spokane_anomalies, spokane_table

TRACK = "lsm"
PARTS = {
    0: ["q0_peak_month"],
    1: ["q1_same_bytes", "q1_demo", "q1_cell_km", "q1_gauges", "q1_raw_or_derived"],
    2: ["q2_limits", "q2_extreme", "q2_sentinel"],
    3: ["q3_interval", "q3_frozen"],
    4: ["q4_lat_order", "q4_cell", "q4_region", "q4_bad_cell"],
    5: ["q5a", "q5b", "q5c", "q5d"],
    6: ["q6_sign", "q6_lag", "q6_neff", "q6_meaning"],
    7: ["q7_pipeline", "q7_task", "q7_leak"],
    8: ["q8a", "q8"],
}
# Questions with a right answer; the others (lists, findings, free text) count as answered or not.
SELF_CHECKED = [q for p in (0, 1, 2, 3, 4, 6, 7) for q in PARTS[p]]

_ANSWERED, _PASSED = {}, {}
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
    if abs(a[0] - v0) <= 0.05 * v0 + 1 and abs(a[1] - v1) <= 0.05 * v1 + 1:
        return _ok()
    return _hint("average static['gauges_total'] over the months of 1991-1996, then over 1997-2020.")


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
        "take the differences between consecutive dates in days (np.diff on the dates), then min and max.")


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
    if abs(float(rate) - r) > 1.0:
        return _hint("compute the mean of the drought labels over land cells in 2011-2020, in percent.")
    return _ok()


def _c_q7_leak(a):
    return _ok() if _letter(a) == "A" else _hint(
        "neighbouring months and neighbouring cells carry almost the same soil water. Which split puts "
        "near-copies of test rows in training?")


def _c_wrap(a):
    _STATE["wrap"] = [str(x) for x in a]
    if len(a) == 3 and all(_words(x) >= 20 for x in a):
        return _ok("recorded")
    return _hint("three answers, two to four sentences each (at least 20 words).")


_CHECKS = {
    "q0_peak_month": _c_q0, "q1_same_bytes": _c_q1_same, "q1_demo": _c_q1_demo, "q1_cell_km": _c_q1_km,
    "q1_raw_or_derived": _c_q1_raw, "q1_gauges": _c_q1_gauges, "q2_limits": _c_q2_limits,
    "q2_extreme": _c_q2_extreme, "q2_sentinel": _c_q2_sentinel, "q3_interval": _c_q3_interval,
    "q3_frozen": _c_q3_frozen, "q4_lat_order": _c_q4_order, "q4_cell": _c_q4_cell, "q4_region": _c_q4_region,
    "q4_bad_cell": _c_q4_bad, "q5a": _c_q5a, "q5c": _c_q5c, "q5d": _c_text, "q6_sign": _c_q6_sign,
    "q6_lag": _c_q6_lag, "q6_neff": _c_q6_neff, "q6_meaning": _c_q6_meaning, "q7_pipeline": _c_q7_pipeline,
    "q7_task": _c_q7_task, "q7_leak": _c_q7_leak, "q8": _c_text, "wrap": _c_wrap,
}


def _run(qid, answer):
    if _blank(answer):
        print(NOT_ANSWERED)
        return None
    try:
        return _CHECKS[qid](answer)
    except Exception as e:  # noqa: BLE001  a malformed answer should give a hint, not a traceback
        return _hint(f"the answer has an unexpected form ({type(e).__name__}: {e}). Look at the example "
                     "format in the cell.")


def check(qid, answer):
    """Self-check for question `qid`: prints 'correct', a hint or 'not answered yet', and records the result."""
    res = _run(qid, answer)
    _ANSWERED[qid] = res is not None
    _PASSED[qid] = bool(res)
    if qid == "wrap" and _blank(answer):
        _STATE["wrap"] = None


def check_worked(qid, answer):
    """The same check for the worked solutions: prints, records nothing."""
    saved = dict(_STATE)
    _run(qid, answer)
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


def print_summary(netid=""):
    """Print the Session 7 summary block from the answers and self-checks run so far."""
    netid = str(netid or "").strip()
    parts = " | ".join(f"P{p} {sum(_ANSWERED.get(q, False) for q in qs)}/{len(qs)}" for p, qs in PARTS.items())
    passed = sum(_PASSED.get(q, False) for q in SELF_CHECKED)
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
        f"self-checks passed: {passed} of {len(SELF_CHECKED)}",
        f"planted problems found: {found_s} (seed {seed})",
        f"derived quantity (truth / as given / after cleaning): {der_s}",
    ] + [f"reflection {i + 1}: {_short(w)}" for i, w in enumerate(wrap)] + ["==== END SUMMARY ===="]
    print("\n".join(lines))
