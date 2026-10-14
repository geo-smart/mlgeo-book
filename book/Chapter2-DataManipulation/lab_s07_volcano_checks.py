"""Self-checks and the end-of-session summary for the Session 7 volcano lab (Mount St Helens, 2004 to 2008).

Each check prints `correct`, a hint that says where to look (never the answer), or `not answered yet`,
and remembers its last result so that `print_summary` can report it. The expected values are computed
here from the course copies in snapshots/, never typed in. Nothing here downloads anything.

"Your turn" cells end with `your_turn(key, globals())`: it checks what the student wrote and, if a result is
missing (None) or has the wrong kind or shape, puts the reference result in its place so that the cells
after it still run, and says so in one line.
Run from book/Chapter2-DataManipulation.
"""
from __future__ import annotations

import json
import numbers
import re
from pathlib import Path

import numpy as np
import pandas as pd

_SNAP = Path("snapshots")
_cache = {}
_LOG = {}        # question id -> "passed", "not yet", "not answered" or "recorded" (last run of each check)

# Self-checks that give a verdict, by part. Their number is the m in "self-checks passed: n of m".
GRADED = ["1.rows", "1.same", "1.updated", "1.nyquist", "1.vclip", "1.kind",
          "2.outside", "2.magtypes", "2.missing", "2.error", "2.largest",
          "3.dt", "3.gap", "3.extreme", "3.window", "3.rate", "3.clock", "3.burst",
          "4.df", "4.spectra", "4.fpeak", "4.low", "4.line",
          "5.shift", "5.clean", "5d",
          "6.pearsonA", "6.disp", "6.spearmanB", "6.trapA", "6.trapB", "6.says",
          "7.steps", "7.task", "7.frac", "7.split", "7.acc", "7.test", "7.big", "7.hist", "7.balance",
          "8", "w1", "w2", "w3"]

# Answer variables of each part, as named in the notebook. A name in <...> is a check instead: a "Your turn"
# result counts as answered when its check ran on the student's own result ("<clean>" is the cleaning in Part 5c).
PARTS = {
    "P0": ["q0_rate_guess"],
    "P1": ["q1_rows", "q1_same", "q1_updated", "q1_nyquist", "q1_vclip", "q1_kind"],
    "P2": ["LIMITS", "<2.outside>", "q2_magtypes", "q2_missing", "q2_error_id", "q2_largest"],
    "P3": ["q3_dt", "q3_gap_min", "q3_n_extreme", "<3.window>", "<3.rate>", "<3.clock>", "q3_burst"],
    "P4": ["q4_df", "<4.spectra>", "q4_fpeak", "q4_low", "q4_line"],
    "P5": ["q5a_expect", "findings", "<5.shift>", "<clean>", "q5d"],
    "P6": ["q6_sign", "<6.pearsonA>", "<6.disp>", "<6.spearmanB>", "q6_trapA", "q6_trapB", "q6_says"],
    "P7": ["q7_steps", "q7_task", "q7_target", "<7.frac>", "<7.split>", "q7_acc", "q7_test_pos", "<7.big>", "<7.hist>",
           "q7_balance"],
    "P8": ["tool_findings", "q8"],
}

# The notebook before the 14 October update had no "Your turn" cells: it asked q3_rate, q6_pearsonA, q6_spearmanB,
# q7_frac and q7_big as typed values and graded 37 checks. A copy of that version imports this module after
# git pull; its summary counts that version's checks and parts, so a finished copy reads as it did before.
GRADED_EARLIER = ["1.rows", "1.same", "1.updated", "1.nyquist", "1.vclip", "1.kind",
                  "2.magtypes", "2.missing", "2.error", "2.largest",
                  "3.dt", "3.gap", "3.extreme", "3.rate", "3.burst",
                  "4.df", "4.fpeak", "4.low", "4.line",
                  "5.clean", "5d",
                  "6.pearsonA", "6.spearmanB", "6.trapA", "6.trapB", "6.says",
                  "7.steps", "7.task", "7.frac", "7.acc", "7.test", "7.big", "7.balance",
                  "8", "w1", "w2", "w3"]
PARTS_EARLIER = {
    "P0": ["q0_rate_guess"],
    "P1": ["q1_rows", "q1_same", "q1_updated", "q1_nyquist", "q1_vclip", "q1_kind"],
    "P2": ["LIMITS", "q2_magtypes", "q2_missing", "q2_error_id", "q2_largest"],
    "P3": ["q3_dt", "q3_gap_min", "q3_n_extreme", "q3_rate", "q3_burst"],
    "P4": ["q4_df", "q4_fpeak", "q4_low", "q4_line"],
    "P5": ["q5a_expect", "findings", "<clean>", "q5d"],
    "P6": ["q6_sign", "q6_pearsonA", "q6_spearmanB", "q6_trapA", "q6_trapB", "q6_says"],
    "P7": ["q7_steps", "q7_task", "q7_target", "q7_frac", "q7_acc", "q7_test_pos", "q7_big", "q7_balance"],
    "P8": ["tool_findings", "q8"],
}
# Names that only the updated notebook defines: your_turn is imported in its first code cell, the others are
# placeholders of its Your-turn cells. The earlier notebook has none of them.
_NEW_VERSION_NAMES = ("your_turn", "per10", "JUMP_MINUTE", "SHIFT_S")


def _is_new_version(ns):
    return any(k in ns for k in _NEW_VERSION_NAMES)


def _record(qid, status):
    _LOG[qid] = status


def _cat():
    if "cat" not in _cache:
        _cache["cat"] = pd.read_csv(_SNAP / "comcat_msh_2004-2008.csv", parse_dates=["time", "updated"])
    return _cache["cat"]


def _wave():
    if "wave" not in _cache:
        from obspy import read
        _cache["wave"] = read(str(_SNAP / "msh_UW.HSR.EHZ_2004-11-29T15-21.mseed"))
    return _cache["wave"]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _close(v, target, rel=None, abs_=None):
    v = _num(v)
    if v is None:
        return False
    tol = abs_ if abs_ is not None else abs(target) * (rel or 0.02)
    return abs(v - target) <= tol


def _word(v):
    return str(v).strip().strip("() ").lower()


def _expected():
    if "exp" in _cache:
        return _cache["exp"]
    cat, st = _cat(), _wave()
    d = np.concatenate([tr.data for tr in st])
    gaps = st.get_gaps()
    prov = json.loads((_SNAP / "msh_2004-2008.provenance.json").read_text())
    sens = prov["station"]["EHZ_sensitivity_counts_per_m_per_s"]
    pair = cat[cat["time"].duplicated(keep=False)]["id"].tolist()
    e = {
        "rows": len(cat), "updated_max": int(cat["updated"].dt.year.max()),
        "nyquist": prov["station"]["EHZ_sample_rate_hz"] / 2, "v_clip": float(d.max() - np.median(d)) / sens,
        "n_magtypes": cat["magType"].nunique(), "most_missing": cat.isna().sum().idxmax(),
        "pair": pair, "defensible": [cat.loc[cat["depth"].idxmax(), "id"], cat.loc[cat["mag"].idxmin(), "id"]],
        "largest_day": str(cat.loc[cat["mag"].idxmax(), "time"].date()),
        "dt": 1 / st[0].stats.sampling_rate, "gap_min": sum(g[6] for g in gaps) / 60,
        "n_extreme": int(((d == d.max()) | (d == d.min())).sum()),
    }
    _cache["exp"] = e
    return e


HINTS = {
    "1.rows": "Count the rows of the course copy with len(); the header line is not a row.",
    "1.same": "Answer 'yes', 'no' or 'no download', from the comparison table (compare the fingerprints).",
    "1.updated": "Take the latest year in the column that records each row's last revision.",
    "1.nyquist": "The highest frequency a record can hold is half its sampling rate (StationXML).",
    "1.vclip": "Subtract the record's offset (its median) from the largest count, then divide by the sensitivity in counts per m/s.",
    "1.kind": "For each file: did a person or a program interpret many recordings to make each value?",
    "2.magtypes": "Use .nunique() on the magnitude-type column.",
    "2.missing": "Use isna().sum() on the table and take the largest count.",
    "2.error": "Compare each candidate in the screening output with your limits: what could the network's processing produce that the ground could not?",
    "2.largest": "Sort by magnitude and give the date of the first row as YYYY-MM-DD.",
    "3.dt": "The sampling interval is 1 divided by the sampling rate, in seconds.",
    "3.gap": "st.get_gaps() gives one list per gap; its item [6] is the gap length in seconds. Convert to minutes.",
    "3.extreme": "Count the samples equal to the record's maximum or to its minimum.",
    "3.rate": "Divide the number of detections by the hours that have data, not by the six hours requested.",
    "3.burst": "Compare the burst's start time at the three stations. Ground waves reach stations kilometers apart at different times.",
    "4.df": "Frequency resolution is the sampling rate divided by the number of samples per window.",
    "4.fpeak": "Read the peak of the spectrum of the first 2.56 s after each trigger, printed above the plot.",
    "4.low": "In the response plotted in Part 1, where does the curve drop below its plateau (dashed line)?",
    "6.pearsonA": "Use the Pearson value printed for the 5-minute bins.",
    "6.spearmanB": "Use the Spearman value printed for magnitude against time since the previous located event.",
    "6.trapA": "Read the lines printed under A: how much does the count vary, and what happens without the loud bins?",
    "6.trapB": "Compare how many earthquakes per day the catalog lists with how many the detector counts at HSR.",
    "7.task": "Use one of the five task names and write the target in words.",
    "7.frac": "Divide the eruption days by the days kept; give a fraction between 0 and 1, not a percent.",
    "7.acc": "A model that always says 'not erupting' is right on every day outside the eruption; give a fraction.",
    "7.test": "Count the eruption days in 2016 to 2025.",
    "7.big": "Count the rows with magnitude 2.5 or more and divide by all rows.",
    # Your-turn cells: where to look, never the answer
    "2.outside": "Keep the rows outside your limits: .between(low, high) is True inside them and ~ reverses it; one table for depth, one for magnitude.",
    "3.window": "Keep the rows with time at or after 15:00 and before 21:00 UTC on 29 November 2004; join the two conditions with &.",
    "3.rate": "Divide the number of detections by the hours that have data, not by the six hours requested.",
    "3.clock": "intervals: the spacing in seconds from .diff(), keeping only values below 600 s; cv_real: .std() / .mean() of intervals; per10: resample the detection times into 10-minute bins and count with .size().",
    "4.spectra": "Take the median along the time axis (axis=1); for med_with keep only the columns where has is True, for med_without the others.",
    "5.shift": "late is True from JUMP_MINUTE on; SHIFT_S is the median of lag_s on those rows minus the median on the others, in seconds.",
    "6.pearsonA": "Correlate A['detections'] with A['mean_abs_counts']; take element [0] of what pearsonr and spearmanr return.",
    "6.disp": "Divide the variance of A['detections'] by its mean.",
    "6.spearmanB": "Check the three conditions (from 2004-10-16, before 2006-01-01, type earthquake), that wait_h is in hours, and that the first row (no previous event) is dropped.",
    "7.frac": "Take the mean of the erupting column of daily: a fraction between 0 and 1.",
    "7.split": "train: rows with index before split; test: from split on; w_pos: non-eruption days over eruption days, in train only.",
    "7.hist": "counts is the first of the two arrays np.histogram returns; tr_m holds the magnitudes of rows before 2005-07-01 only.",
}

# Where a Your-turn check judges several results at once, one hint per result, so the verdict says which one to fix.
NAME_HINTS = {
    "out_d": "the rows whose depth is outside LIMITS['depth_km']: .between(low, high), then ~",
    "out_m": "the rows whose magnitude is outside LIMITS['mag']: .between(low, high), then ~",
    "intervals": "the spacing in s from .diff().dt.total_seconds(), keeping only values below 600",
    "cv_real": ".std() / .mean() of intervals",
    "per10": "count with .size() after resampling the detection times into 10-minute bins",
    "med_all": "the median of S along the time axis (axis=1)",
    "med_with": "the median over the columns where has is True only",
    "med_without": "the median over the other columns only (~has)",
    "rA": "element [0] of what stats.pearsonr returns for the two columns of A",
    "sA": "element [0] of what stats.spearmanr returns for the two columns of A",
    "B": "three conditions (from 2004-10-16, before 2006-01-01, type earthquake), wait_h in hours, the first row dropped",
    "rB": "element [0] of what stats.pearsonr returns for B['mag'] and B['wait_h']",
    "sB": "element [0] of what stats.spearmanr returns for B['mag'] and B['wait_h']",
    "train": "the rows of daily whose index is before split",
    "test": "the rows of daily whose index is at or after split",
    "w_pos": "non-eruption days over eruption days, counted in train only",
    "counts": "the first of the two arrays np.histogram returns",
    "tr_m": "the magnitudes of the rows before 2005-07-01 only",
}


def _empty(value):
    return value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, (list, dict)) and not value)


def check(qid: str, value, **ctx) -> None:
    """Immediate feedback: 'correct', or a hint that says where to look (never the answer)."""
    if _empty(value):
        print(f"[{qid}] not answered yet")
        _record(qid, "not answered")
        return
    e = _expected()
    ok = None
    if qid == "0.guess":
        print(f"[{qid}] recorded: {value} per hour. You test it against the detector in Part 3.")
        _record(qid, "recorded"); return
    if qid == "2.limits":
        print(f"[{qid}] recorded. The worked solution gives one defensible set."); _record(qid, "recorded"); return
    if qid == "6.sign":
        print(f"[{qid}] recorded: {value}. Compare it with the sign the data give below.")
        _record(qid, "recorded"); return
    if qid == "1.rows":
        ok = _num(value) == e["rows"]
    elif qid == "1.same":
        ok = _word(value) == str(ctx.get("truth", "")).lower()
    elif qid == "1.updated":
        ok = _num(value) == e["updated_max"]
    elif qid == "1.nyquist":
        ok = _close(value, e["nyquist"], abs_=0.01)
    elif qid == "1.vclip":
        ok = _close(value, e["v_clip"], rel=0.1)
    elif qid == "1.kind":
        ok = isinstance(value, dict) and _word(value.get("catalog", "")).startswith("derived") \
            and _word(value.get("seismogram", "")).startswith("raw")
    elif qid == "2.magtypes":
        ok = _num(value) == e["n_magtypes"]
    elif qid == "2.missing":
        ok = _word(value) == e["most_missing"].lower()
    elif qid == "2.error":
        v = str(value).strip()
        if v in e["defensible"]:
            print(f"[{qid}] defensible: say why in a comment. The worked solution flags a different pair; compare after.")
            _record(qid, "passed"); return
        ok = v in e["pair"]
    elif qid == "2.largest":
        ok = str(value).strip()[:10] == e["largest_day"]
    elif qid == "3.dt":
        ok = _close(value, e["dt"], abs_=1e-6)
    elif qid == "3.gap":
        ok = _close(value, e["gap_min"], abs_=0.5)
    elif qid == "3.extreme":
        ok = _num(value) == e["n_extreme"]
    elif qid == "3.rate":
        ok = _close(value, ctx["truth"], rel=0.03)
    elif qid == "3.burst":
        ok = _word(value).startswith("no") or _word(value) == "false"
    elif qid == "4.df":
        truth = ctx["truth"] if "truth" in ctx else float(ctx["freqs"][1] - ctx["freqs"][0])
        ok = _close(value, truth, rel=0.02)
    elif qid == "4.fpeak":
        ok = _close(value, ctx["truth"], abs_=0.5)
    elif qid == "4.low":
        ok = _close(value, ctx["truth"], abs_=0.5)
    elif qid in ("6.pearsonA", "6.spearmanB"):
        ok = _close(value, ctx["truth"], abs_=0.03)
    elif qid == "6.trapA":
        ok = _word(value)[:1] in ("c", "e")
    elif qid == "6.trapB":
        ok = _word(value)[:1] == "b"
    elif qid == "7.task":
        ok = _word(value) in ("classification", "regression", "detection", "forecasting", "clustering") \
            and not _empty(ctx.get("target"))
        if ok:
            print(f"[{qid}] recorded: now check it against the distributions below")
            _record(qid, "passed"); return
    elif qid == "7.frac":
        ok = _close(value, ctx["truth"], abs_=0.005)
    elif qid == "7.acc":
        truth = ctx["truth"] if "truth" in ctx else 1 - float(ctx["daily"]["erupting"].mean())
        ok = _close(value, truth, abs_=0.005)
    elif qid == "7.test":
        truth = ctx["truth"] if "truth" in ctx else int(ctx["daily"].loc[ctx["daily"].index >= ctx["split"], "erupting"].sum())
        ok = _num(value) == truth
    elif qid == "7.big":
        ok = _close(value, ctx["truth"], abs_=0.01)
    if ok is None:
        print(f"[{qid}] no check defined"); return
    print(f"[{qid}] correct" if ok else f"[{qid}] not yet. Hint: {HINTS[qid]}")
    _record(qid, "passed" if ok else "not yet")


def check_text(qid: str, text, needs: dict, min_words: int = 15) -> None:
    """Feedback on a written answer: which ideas are not there yet, by topic, never the wording.
    It looks for topics, not for correctness: covering the topics is not the same as being right."""
    if not text or not str(text).strip():
        print(f"[{qid}] not answered yet"); _record(qid, "not answered")
        return
    t = str(text).lower()
    if len(t.split()) < min_words:
        print(f"[{qid}] write a little more: at least {min_words} words with a physical reason.")
        _record(qid, "not yet")
        return
    missing = [topic for topic, pattern in needs.items() if not re.search(pattern, t)]
    print(f"[{qid}] covers the topics" if not missing else f"[{qid}] not yet. Think about: {', '.join(missing)}")
    _record(qid, "not yet" if missing else "passed")


def check_expectations(items) -> None:
    """Part 5a: which parts of the chain from ground to table a list of written expectations covers."""
    if not items:
        print("[5a] not answered yet"); _record("5a", "not answered"); return
    t = " ".join(map(str, items)).lower()
    topics = {
        "the recording range of the telemetry and digitizer": r"clip|saturat|range|full scale",
        "the telemetry and the recording computer": r"telemetry|radio|dropout|outage|gap|missing|zero",
        "the clock": r"clock|time|gps",
        "the station's gain or calibration (including calibration pulses)": r"gain|calibrat|sensitiv",
        "how the event list is built (people and programs)": r"duplicat|magnitude|label|catalog|analyst|detector",
    }
    covered = [k for k, p in topics.items() if re.search(p, t)]
    print(f"[5a] your list touches {len(covered)} of {len(topics)} parts of the chain from ground to table.")
    rest = [k for k in topics if k not in covered]
    if rest:
        print("     Think also about:", "; ".join(rest))
    _record("5a", "recorded")


def check_clean(three, changed: bool, error=None) -> None:
    """Part 5c: did the cleaning move the coefficient of variation of the drumbeat spacing toward the truth?
    error: what clean() raised, if it stopped; the cleaning is then not judged."""
    if error is not None:
        print("[5.clean] not yet: clean() stopped with the error printed above"); _record("5.clean", "not yet"); return
    if not changed:
        print("[5.clean] not answered yet"); _record("5.clean", "not answered"); return
    gap0 = abs(three.loc["contaminated", "cv"] - three.loc["truth", "cv"])
    gap1 = abs(three.loc["after your cleaning", "cv"] - three.loc["truth", "cv"])
    if gap1 < gap0:
        print("[5.clean] your cleaning moved the CV toward the truth"); _record("5.clean", "passed")
    else:
        print("[5.clean] not yet: the CV moved away from the truth. Which stretches have no data, and which rows are not earthquakes?")
        _record("5.clean", "not yet")


def check_pipeline(steps, record=True) -> None:
    """Part 7: an ordered list of (step, reason) pairs from raw file to a learnable table.
    The worked solution calls it with record=False, so that it does not count as the student's answer."""
    if not steps:
        print("[7.steps] not answered yet"); _record("7.steps", "not answered"); return
    names = [str(s[0] if isinstance(s, (list, tuple)) else s).lower() for s in steps]
    reasons = [str(s[1]).strip() if isinstance(s, (list, tuple)) and len(s) > 1 else "" for s in steps]
    msgs = []
    if len(steps) < 5:
        msgs.append("a chain from raw file to model input usually has at least five steps")
    if any(not r for r in reasons):
        msgs.append("give every step a reason, as (step, reason)")
    i_split = next((i for i, s in enumerate(names) if re.search(r"\bsplit", s)), None)
    # Balancing reweights or resamples the classes or target bins. Resampling a time series to daily counts
    # is aggregation, not balancing, so a bare "resample" does not count here.
    bal = (r"balanc|class weight|sample weight|\bweight|oversampl|undersampl|smote|"
           r"resampl\w*\s+(the\s+)?(training|train|minority|rare|positive|eruption|drumbeat|class)")
    i_bal = next((i for i, s in enumerate(names) if re.search(bal, s)), None)
    if i_split is None:
        msgs.append("where do you split the data into training and test sets?")
    if i_bal is None:
        msgs.append("where does balancing happen?")
    if i_split is not None and i_bal is not None and i_bal < i_split:
        msgs.append("look at the order of splitting and balancing: which one can leak information from the test set?")
    if not any(re.search(r"select|choose|keep|filter|box|period|type", s) for s in names):
        msgs.append("which rows or times do you select, and why?")
    print("[7.steps] covers the topics" if not msgs else "[7.steps] not yet. " + "; ".join(msgs))
    if record:
        _record("7.steps", "not yet" if msgs else "passed")


# ----------------------------------------------------------------------------- "Your turn" cells
# One defensible set of limits (worked solution of Part 2), used only when LIMITS is not filled in.
REF_LIMITS = {"depth_km": (-2.5, 30.0), "mag": (-2.0, 5.0)}


def _limits(ns):
    """The student's LIMITS if all four numbers are there, else the reference set; and whether they were there."""
    try:
        vals = list(ns["LIMITS"]["depth_km"]) + list(ns["LIMITS"]["mag"])
        if len(vals) == 4 and all(_num(v) is not None for v in vals):
            return ns["LIMITS"], True
    except (KeyError, TypeError):
        pass
    return REF_LIMITS, False


def _ref_outside(ns):
    cat, (lim, _) = _cat(), _limits(ns)
    return {"out_d": cat[~cat["depth"].between(*lim["depth_km"])], "out_m": cat[~cat["mag"].between(*lim["mag"])]}


def _ref_window(ns):
    cat = _cat()
    return {"in_window": cat[(cat["time"] >= "2004-11-29 15:00") & (cat["time"] < "2004-11-29 21:00")]}


def _ref_clock(ns):
    det, st = ns["det"], _wave()
    hours = sum(tr.stats.npts / tr.stats.sampling_rate for tr in st) / 3600
    gaps = det["time"].diff().dt.total_seconds()
    gaps = gaps[gaps < 600]
    return {"rate": len(det) / hours, "intervals": gaps, "cv_real": gaps.std() / gaps.mean(),
            "per10": det.set_index("time").resample("10min").size()}


def _ref_spectra(ns):
    S, has = ns["S"], ns["has"]
    return {"med_all": np.median(S, axis=1), "med_with": np.median(S[:, has], axis=1),
            "med_without": np.median(S[:, ~has], axis=1)}


def _shift_minute(ns):
    """Part 5b: does JUMP_MINUTE agree with the twin? A wrong minute can still give a consistent SHIFT_S."""
    twin, jump = ns.get("twin"), _num(ns.get("JUMP_MINUTE"))
    if twin is None:
        return True
    if "clock jump" in twin.planted:
        return jump is not None and abs(jump - twin.where["clock jump"]) < 2
    return _num(ns.get("SHIFT_S")) is not None and abs(_num(ns.get("SHIFT_S"))) < 0.5


def _ref_shift(ns):
    ev, jump = ns["ev"], _num(ns.get("JUMP_MINUTE"))
    if jump is None:
        return {"SHIFT_S": 0.0}
    late = ev["minute"] >= jump
    return {"SHIFT_S": float(ev.loc[late, "lag_s"].median() - ev.loc[~late, "lag_s"].median())}


def _ref_A(ns):
    from scipy import stats
    A = ns["A"]
    return {"rA": stats.pearsonr(A["detections"], A["mean_abs_counts"])[0],
            "sA": stats.spearmanr(A["detections"], A["mean_abs_counts"])[0],
            "disp": A["detections"].var() / A["detections"].mean()}


def _ref_B(ns):
    from scipy import stats
    cat = _cat()
    B = cat[(cat["time"] >= "2004-10-16") & (cat["time"] < "2006-01-01") & (cat["type"] == "earthquake")].copy()
    B["wait_h"] = B["time"].diff().dt.total_seconds() / 3600
    B = B.dropna(subset=["wait_h"])
    return {"B": B, "rB": stats.pearsonr(B["mag"], B["wait_h"])[0], "sB": stats.spearmanr(B["mag"], B["wait_h"])[0]}


def _ref_split(ns):
    daily, split = ns["daily"], ns["split"]
    train, test = daily[daily.index < split], daily[daily.index >= split]
    return {"frac_pos": float(daily["erupting"].mean()), "train": train, "test": test,
            "w_pos": (~train["erupting"]).sum() / train["erupting"].sum()}


def _ref_hist(ns):
    cat = _cat()
    return {"counts": np.histogram(cat["mag"], bins=ns["bins"])[0], "frac_big": float((cat["mag"] >= 2.5).mean()),
            "tr_m": cat.loc[cat["time"] < "2005-07-01", "mag"]}


# key: the results the cell produces, the check ids with the results each one judges, the reference, and
# (optionally) what must be filled in before the results can be judged.
TURNS = {
    "2.outside": {"products": ["out_d", "out_m"], "checks": {"2.outside": ["out_d", "out_m"]}, "ref": _ref_outside,
                  "ready": lambda ns: _limits(ns)[1], "need": "fill LIMITS first"},
    "3.window": {"products": ["in_window"], "checks": {"3.window": ["in_window"]}, "ref": _ref_window},
    "3.clock": {"products": ["rate", "intervals", "cv_real", "per10"],
                "checks": {"3.rate": ["rate"], "3.clock": ["intervals", "cv_real", "per10"]}, "ref": _ref_clock},
    "4.spectra": {"products": ["med_all", "med_with", "med_without"],
                  "checks": {"4.spectra": ["med_all", "med_with", "med_without"]}, "ref": _ref_spectra},
    "5.shift": {"products": ["SHIFT_S"], "checks": {"5.shift": ["SHIFT_S"]}, "ref": _ref_shift,
                "ready": lambda ns: _num(ns.get("JUMP_MINUTE")) is not None, "need": "set JUMP_MINUTE first",
                "also": (_shift_minute, "Look again at panel 4: from which minute does listed minus onset stay several "
                                        "seconds away from -0.3 s? If it never does, keep 90 and the shift should be near 0."),
                "unready_say": "[SHIFT_S] set JUMP_MINUTE first; continuing with no shift"},
    "6.A": {"products": ["rA", "sA", "disp"], "checks": {"6.pearsonA": ["rA", "sA"], "6.disp": ["disp"]}, "ref": _ref_A},
    "6.B": {"products": ["B", "rB", "sB"], "checks": {"6.spearmanB": ["B", "rB", "sB"]}, "ref": _ref_B},
    "7.split": {"products": ["frac_pos", "train", "test", "w_pos"],
                "checks": {"7.frac": ["frac_pos"], "7.split": ["train", "test", "w_pos"]}, "ref": _ref_split},
    "7.hist": {"products": ["counts", "frac_big", "tr_m"],
               "checks": {"7.hist": ["counts", "tr_m"], "7.big": ["frac_big"]}, "ref": _ref_hist},
}
# Tolerance per numeric result: ("rel", x) or ("abs", x)
TOL = {"rate": ("rel", 0.03), "cv_real": ("abs", 0.01), "SHIFT_S": ("abs", 0.05), "rA": ("abs", 0.01),
       "sA": ("abs", 0.01), "disp": ("rel", 0.02), "rB": ("abs", 0.01), "sB": ("abs", 0.01),
       "frac_pos": ("abs", 0.002), "w_pos": ("rel", 0.02), "frac_big": ("abs", 0.002)}


def _is_number(v):
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)) and np.isfinite(float(v))


def _same_kind(value, ref):
    """Can the cells that follow use this value in place of the reference? (type, columns, shape)"""
    if value is None:
        return False
    if isinstance(ref, pd.DataFrame):
        return isinstance(value, pd.DataFrame) and set(ref.columns) <= set(value.columns)
    if isinstance(ref, pd.Series):
        return isinstance(value, pd.Series) and len(value) > 0
    if isinstance(ref, np.ndarray):
        return isinstance(value, (np.ndarray, pd.Series, list)) and np.shape(value) == ref.shape
    return _is_number(value)


def _matches(value, ref, tol=None):
    """Is the student's result the reference result? Tables are compared by their rows, arrays by their values."""
    try:
        if isinstance(ref, pd.DataFrame):
            ok = isinstance(value, pd.DataFrame) and set(value.index) == set(ref.index)
            if ok and "wait_h" in ref.columns:
                ok = "wait_h" in value.columns and np.allclose(value.loc[ref.index, "wait_h"].astype(float), ref["wait_h"])
            return bool(ok)
        if isinstance(ref, pd.Series):
            return (isinstance(value, pd.Series) and len(value) == len(ref)
                    and np.allclose(value.to_numpy(dtype=float), ref.to_numpy(dtype=float), equal_nan=True))
        if isinstance(ref, np.ndarray):
            v = np.asarray(value, dtype=float)
            return v.shape == ref.shape and np.allclose(v, ref, rtol=1e-6, equal_nan=True)
        kind, x = tol or ("rel", 0.02)
        return _close(value, float(ref), **({"rel": x} if kind == "rel" else {"abs_": x}))
    except (TypeError, ValueError, KeyError):
        return False


def use_or_reference(value, reference_fn, name, say=True):
    """Return (the value to continue with, whether it is the reference). The student's value is kept when the
    cells that follow can use it, even if it is not correct; otherwise the reference result is used."""
    ref = reference_fn()
    if _same_kind(value, ref):
        return value, False
    if say:
        print(f"[{name}] Using the reference result so you can continue; compare with yours.")
    return ref, True


def your_turn(key: str, ns: dict) -> None:
    """Check a "Your turn" cell and, where a result is missing or unusable, put the reference in its place in ns
    (the notebook's globals()), saying so in one line. Feedback is 'correct' or a hint, never the answer."""
    spec = TURNS[key]
    ref = spec["ref"](ns)
    ready = spec.get("ready", lambda _: True)(ns)
    for qid, names in spec["checks"].items():
        ok = {n: _matches(ns.get(n), ref[n], TOL.get(n)) for n in names}
        if all(ns.get(n) is None for n in names):
            print(f"[{qid}] not answered yet"); _record(qid, "not answered")
        elif not ready:
            print(f"[{qid}] not answered yet: {spec['need']}"); _record(qid, "not answered")
        elif "also" in spec and not spec["also"][0](ns):
            print(f"[{qid}] not yet. Hint: {spec['also'][1]}"); _record(qid, "not yet")
        elif all(ok.values()):
            print(f"[{qid}] correct"); _record(qid, "passed")
        elif len(names) == 1:
            print(f"[{qid}] not yet. Hint: {HINTS[qid]}"); _record(qid, "not yet")
        else:      # one verdict per result, so it is clear which one to fix
            say = [f"{n} correct" if ok[n] else (f"{n} not answered yet" if ns.get(n) is None
                                                 else f"{n} not yet: {NAME_HINTS[n]}") for n in names]
            print(f"[{qid}] " + "; ".join(say)); _record(qid, "not yet")
    replaced = []
    for n in spec["products"]:
        ns[n], used = use_or_reference(ns.get(n), lambda: ref[n], n, say=False)
        if used:
            replaced.append(n)
    if replaced and not ready and "unready_say" in spec:
        print(spec["unready_say"])
    elif replaced:
        print(f"[{', '.join(replaced)}] Using the reference result so you can continue; compare with yours.")


# ----------------------------------------------------------------------------- the summary
def _answered(v):
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, dict):
        return bool(v) and all(_answered(x) for x in v.values())
    if isinstance(v, tuple):
        return bool(v) and all(_answered(x) for x in v)
    if isinstance(v, list):
        return bool(v)
    return True


def _short(text, n=300):
    t = " ".join(str(text or "").split())
    return t[:n] if t else "not answered"


def print_summary(ns: dict) -> None:
    """Print the block the instructor reads. Call it as print_summary(globals()) in the last cell."""
    netid = str(ns.get("NETID") or "").strip()
    new = _is_new_version(ns)
    graded, part_names = (GRADED, PARTS) if new else (GRADED_EARLIER, PARTS_EARLIER)
    parts = []
    for p, names in part_names.items():
        k = sum(_LOG.get("5.clean" if n == "<clean>" else n[1:-1]) in ("passed", "not yet") if n.startswith("<")
                else _answered(ns.get(n)) for n in names)
        parts.append(f"{p} {k}/{len(names)}")
    passed = sum(_LOG.get(q) == "passed" for q in graded)

    twin = ns.get("twin")
    if twin is not None:
        found = len(twin.match_findings(ns.get("findings") or []))
        planted = f"{found} of {len(twin.planted)} (seed {twin.seed})"
        truth = twin.clock(twin.truth_events)["cv"]
        given = twin.clock(twin.events)["cv"]
        three = ns.get("three")
        after = (f"{three.loc['after your cleaning', 'cv']:.3f}"
                 if _LOG.get("5.clean") in ("passed", "not yet") and three is not None else "not answered")
        derived = f"{truth:.3f} / {given:.3f} / {after} (CV of the drumbeat spacing, no unit)"
    else:
        planted, derived = "not answered (run Part 5)", "not answered (run Part 5)"

    print("==== SESSION 7 SUMMARY ====")
    print("track: volcano")
    print(f"netid: {netid or 'not answered'}")
    print("parts answered: " + " | ".join(parts))
    print(f"self-checks passed: {passed} of {len(graded)}")
    if not new:     # the grading script reads only the keys it knows and skips this line
        print("notebook version: before the 14 October update, counted against that version's questions")
    print(f"planted problems found: {planted}")
    print(f"derived quantity (truth / as given / after cleaning): {derived}")
    for i in (1, 2, 3):
        print(f"reflection {i}: {_short(ns.get(f'w{i}'))}")
    print("==== END SUMMARY ====")
