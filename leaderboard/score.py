"""Score leaderboard submissions and write the standings page.

Two tracks:
- classification (Chapter 3.5): results/predictions_<uwnetid>.csv with header
  row_id,prediction — scored with macro-F1 against the canonical test split of
  the Zenodo 4-class seismic-source dataset (record 14025693).
- forecast (Chapter 4.10), DIAGNOSTIC: results/forecast_<uwnetid>.csv — scored
  with MASE against the held-out tail of the CO2 series. The holdout is public
  NOAA data, so this score is a diagnostic, not a grade.
- forecast_hidden (Chapter 4.10), GRADED: results/forecast_hidden_<uwnetid>.csv
  — scored with MASE against a synthetic series regenerated yearly from a
  private seed. The truth file lives in the gitignored leaderboard/private/;
  when it is absent (every public CI run), the section renders as pending and
  the scores come from the instructor's local run.

MASE is scaled by the naive one-step MAE of the TRAINING series (config
train_file / history_file), per the textbook definition — not by the holdout's
own naive error. A gap between public and hidden scores is how
public-leaderboard overfitting shows.

Usage: python leaderboard/score.py  (from the repo root; writes
book/leaderboard_standings.md)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text())

# Student-facing empty state. Configuration problems must never surface on the
# standings page — they fail the run (nonzero exit + stderr) instead.
EMPTY_MSG = "_No submissions yet — the first scored run posts here._"


def canonical_classification_truth():
    """Rebuild the canonical test split exactly as defined in Chapter 3.5."""
    import pooch

    frames = []
    # Insertion order in config.json defines the canonical concatenation order;
    # iterate a fixed list rather than trusting dict semantics implicitly.
    for fname in list(CONFIG["classification"]["files"]):
        entry = CONFIG["classification"]["files"][fname]
        url, known_hash = entry["url"], entry.get("known_hash")
        path = pooch.retrieve(url, known_hash=known_hash, fname=fname,
                              path=pooch.os_cache("mlgeo"))
        frames.append(pd.read_csv(path, index_col=0))
    df = pd.concat(frames, ignore_index=True).dropna(axis=1)
    X = df.drop(columns=["source", "serial_no"])
    y = df["source"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=2026, stratify=y
    )
    return pd.Series(y_test.values, index=X_test.index, name="truth")


def mase(y_true, y_pred, y_train):
    naive = np.mean(np.abs(np.diff(y_train)))
    if not np.isfinite(naive) or naive == 0:
        raise ValueError("naive baseline has zero variation; MASE undefined")
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / naive


def score_classification():
    truth = canonical_classification_truth()
    rows = []
    for f in sorted(RESULTS.glob("predictions_*.csv")):
        who = f.stem.replace("predictions_", "")
        try:
            sub = pd.read_csv(f).set_index("row_id")["prediction"]
            aligned = sub.reindex(truth.index)
            if aligned.isna().any():
                rows.append((who, None, f"missing {int(aligned.isna().sum())} rows"))
                continue
            score = f1_score(truth.values, aligned.values, average="macro")
            rows.append((who, score, "ok"))
        except Exception as e:  # malformed submission should not kill the run
            rows.append((who, None, f"error: {e}"))
    return rows


def _score_forecast_files(files, prefix, truth, y_train, cfg):
    """Score one forecast track: MASE against `truth`, scaled by the naive
    one-step MAE of `y_train` (the training series, per textbook MASE)."""
    rows = []
    for f in files:
        who = f.stem.replace(prefix, "")
        try:
            sub = pd.read_csv(f)
            merged = truth.merge(sub, on=cfg["key"], how="left",
                                 suffixes=("_true", "_pred"))
            col_t, col_p = cfg["value"] + "_true", cfg["value"] + "_pred"
            if merged[col_p].isna().any():
                rows.append((who, None, "missing horizon rows"))
                continue
            score = mase(merged[col_t], merged[col_p], y_train)
            rows.append((who, score, "ok"))
        except Exception as e:  # malformed submission should not kill the run
            rows.append((who, None, f"error: {e}"))
    return rows


def public_forecast_files():
    """Diagnostic-track submissions: forecast_*.csv minus the hidden track's
    forecast_hidden_*.csv, which the same glob would otherwise swallow."""
    return [f for f in sorted(RESULTS.glob("forecast_*.csv"))
            if not f.name.startswith("forecast_hidden_")]


def score_forecast():
    """Return (rows, config_ok). config_ok=False means the truth or training
    file is missing, which is a configuration failure — the caller reports it
    on stderr and exits nonzero, never on the standings page."""
    cfg = CONFIG["forecast"]
    truth_file = ROOT / cfg["truth_file"]
    train_file = ROOT / cfg["train_file"]
    if not truth_file.exists() or not train_file.exists():
        return [], False
    truth = pd.read_csv(truth_file)
    y_train = pd.read_csv(train_file)[cfg["value"]]
    return _score_forecast_files(public_forecast_files(), "forecast_",
                                 truth, y_train, cfg), True


def score_forecast_hidden():
    """Return (rows, status). status is one of:
    - "scored": private truth present (instructor's machine), rows are scores;
    - "pending": truth absent — the normal public-CI state, NOT an error —
      rows list received submissions with no score;
    - "config_error": the committed history file is missing."""
    cfg = CONFIG["forecast_hidden"]
    history_file = ROOT / cfg["history_file"]
    if not history_file.exists():
        return [], "config_error"
    files = sorted(RESULTS.glob("forecast_hidden_*.csv"))
    truth_file = ROOT / cfg["truth_file"]
    if not truth_file.exists():
        return [(f.stem.replace("forecast_hidden_", ""), None, "received")
                for f in files], "pending"
    truth = pd.read_csv(truth_file)
    y_train = pd.read_csv(history_file)[cfg["value"]]
    return _score_forecast_files(files, "forecast_hidden_",
                                 truth, y_train, cfg), "scored"


def main():
    lines = ["# Class leaderboard", "",
             "Scores computed by CI from submissions in `results/`. "
             "The instructor's hidden test set is scored separately — "
             "a public/hidden gap is the overfitting signal.", ""]

    cls = score_classification()
    lines += ["## Seismic source classification (Chapter 3.5) — macro-F1, higher is better", ""]
    if cls:
        lines += ["| rank | submission | macro-F1 | status |", "|---|---|---|---|"]
        ranked = sorted(cls, key=lambda r: (r[1] is None, -r[1] if r[1] is not None else 0.0))
        for i, (who, s, status) in enumerate(ranked, 1):
            lines.append(f"| {i} | {who} | {s:.4f} | {status} |" if s is not None
                         else f"| {i} | {who} | — | {status} |")
    else:
        lines.append(EMPTY_MSG)
    lines.append("")

    fc, forecast_config_ok = score_forecast()
    lines += ["## CO2 forecasting diagnostic (Chapter 4.10) — MASE, lower is better", "",
              "This track is a diagnostic, not a grade: the holdout months are "
              "public NOAA data. Grading weight sits on the hidden synthetic "
              "track below.", ""]
    if fc:
        lines += ["| rank | submission | MASE | status |", "|---|---|---|---|"]
        ranked = sorted(fc, key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0.0))
        for i, (who, s, status) in enumerate(ranked, 1):
            lines.append(f"| {i} | {who} | {s:.4f} | {status} |" if s is not None
                         else f"| {i} | {who} | — | {status} |")
    else:
        lines.append(EMPTY_MSG)
    lines.append("")

    hid, hidden_status = score_forecast_hidden()
    lines += ["## Hidden synthetic forecast (Chapter 4.10, graded) — MASE, lower is better", ""]
    if hidden_status == "scored" and not hid:
        lines.append(EMPTY_MSG)
    elif hidden_status == "scored":
        lines += ["| rank | submission | MASE | status |", "|---|---|---|---|"]
        ranked = sorted(hid, key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0.0))
        for i, (who, s, status) in enumerate(ranked, 1):
            lines.append(f"| {i} | {who} | {s:.4f} | {status} |" if s is not None
                         else f"| {i} | {who} | — | {status} |")
    elif hid:  # pending, with submissions received
        lines += ["The truth series is private (regenerated yearly from a "
                  "private seed) and is scored in the instructor's run. "
                  "Submissions received:", ""]
        lines += ["| submission | status |", "|---|---|"]
        for who, _, status in hid:
            lines.append(f"| {who} | {status} |")
    else:
        lines += ["Scored in the instructor's run against the private truth "
                  "series; results are announced in class.", "", EMPTY_MSG]
    lines.append("")

    out = ROOT / "book" / "leaderboard_standings.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")

    ok = True
    if not forecast_config_ok:
        print(
            f"CONFIG ERROR: forecast truth file "
            f"({CONFIG['forecast']['truth_file']}) or training-tail file "
            f"({CONFIG['forecast']['train_file']}) not found (relative to the "
            "repo root). Standings were written with a clean empty state; "
            "restore the file or fix the path in leaderboard/config.json.",
            file=sys.stderr,
        )
        ok = False
    if hidden_status == "config_error":
        print(
            f"CONFIG ERROR: hidden-track history file not found at "
            f"{CONFIG['forecast_hidden']['history_file']} (relative to the "
            "repo root). It is committed data, not private truth — restore it "
            "or fix the path in leaderboard/config.json.",
            file=sys.stderr,
        )
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
