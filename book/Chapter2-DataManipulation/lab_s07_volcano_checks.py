"""Self-checks and the end-of-session summary for the Session 7 volcano lab (Mount St Helens, 2004 to 2008).

Each check prints `correct`, a hint that says where to look (never the answer), or `not answered yet`,
and remembers its last result so that `print_summary` can report it. The expected values are computed
here from the course copies in snapshots/, never typed in. Nothing here downloads anything.
Run from book/Chapter2-DataManipulation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_SNAP = Path("snapshots")
_cache = {}
_LOG = {}        # question id -> "passed", "not yet", "not answered" or "recorded" (last run of each check)

# Self-checks that give a verdict, by part. Their number is the m in "self-checks passed: n of m".
GRADED = ["1.rows", "1.same", "1.updated", "1.nyquist", "1.vclip", "1.kind",
          "2.magtypes", "2.missing", "2.error", "2.largest",
          "3.dt", "3.gap", "3.extreme", "3.rate", "3.burst",
          "4.df", "4.fpeak", "4.low", "4.line",
          "5.clean", "5d",
          "6.pearsonA", "6.spearmanB", "6.trapA", "6.trapB", "6.says",
          "7.steps", "7.task", "7.frac", "7.acc", "7.test", "7.big", "7.balance",
          "8", "w1", "w2", "w3"]

# Answer variables of each part, as named in the notebook. "<clean>" is the cleaning in Part 5c.
PARTS = {
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
    "2.magtypes": "Use value_counts() on the magnitude-type column.",
    "2.missing": "Use isna().sum() on the table and take the largest count.",
    "2.error": "Compare each candidate in the screening output with your limits: what could the network's processing produce that the ground could not?",
    "2.largest": "Sort by magnitude and give the date of the first row as YYYY-MM-DD.",
    "3.dt": "The sampling interval is 1 divided by the sampling rate, in seconds.",
    "3.gap": "st.get_gaps() gives the gap length in seconds in its last column; convert to minutes.",
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
        ok = _close(value, ctx["truth"], rel=0.02)
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
        ok = _close(value, ctx["truth"], abs_=0.005)
    elif qid == "7.test":
        ok = _num(value) == ctx["truth"]
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


def check_clean(three, changed: bool) -> None:
    """Part 5c: did the cleaning move the coefficient of variation of the drumbeat spacing toward the truth?"""
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
    parts = []
    for p, names in PARTS.items():
        k = sum(_LOG.get("5.clean") in ("passed", "not yet") if n == "<clean>" else _answered(ns.get(n)) for n in names)
        parts.append(f"{p} {k}/{len(names)}")
    passed = sum(_LOG.get(q) == "passed" for q in GRADED)

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
    print(f"self-checks passed: {passed} of {len(GRADED)}")
    print(f"planted problems found: {planted}")
    print(f"derived quantity (truth / as given / after cleaning): {derived}")
    for i in (1, 2, 3):
        print(f"reflection {i}: {_short(ns.get(f'w{i}'))}")
    print("==== END SUMMARY ====")
