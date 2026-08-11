"""Score leaderboard submissions and write the standings page.

Two tracks:
- classification (Chapter 3.5): results/predictions_<uwnetid>.csv with header
  row_id,prediction — scored with macro-F1 against the canonical test split of
  the Zenodo 4-class seismic-source dataset (record 14025693).
- forecast (Chapter 4.10): results/forecast_<uwnetid>.csv — scored with MASE
  against the held-out tail of the CO2 series (config in leaderboard/config.json).

The instructor also holds hidden test sets (private seeds / later data). A gap
between public and hidden scores is how public-leaderboard overfitting shows.

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


def score_forecast():
    cfg = CONFIG["forecast"]
    truth_file = ROOT / cfg["truth_file"]
    if not truth_file.exists():
        return []
    truth = pd.read_csv(truth_file)
    rows = []
    for f in sorted(RESULTS.glob("forecast_*.csv")):
        who = f.stem.replace("forecast_", "")
        try:
            sub = pd.read_csv(f)
            merged = truth.merge(sub, on=cfg["key"], how="left",
                                 suffixes=("_true", "_pred"))
            col_t, col_p = cfg["value"] + "_true", cfg["value"] + "_pred"
            if merged[col_p].isna().any():
                rows.append((who, None, "missing horizon rows"))
                continue
            score = mase(merged[col_t], merged[col_p], truth[cfg["value"]])
            rows.append((who, score, "ok"))
        except Exception as e:
            rows.append((who, None, f"error: {e}"))
    return rows


def main():
    lines = ["# Class leaderboard", "",
             "Scores computed by CI from submissions in `results/`. "
             "The instructor's hidden test set is scored separately — "
             "a public/hidden gap is the overfitting signal.", ""]

    cls = score_classification()
    lines += ["## Seismic source classification (Chapter 3.5) — macro-F1, higher is better", ""]
    if cls:
        lines += ["| rank | submission | macro-F1 | status |", "|---|---|---|---|"]
        ranked = sorted(cls, key=lambda r: (r[1] is None, -(r[1] or 0)))
        for i, (who, s, status) in enumerate(ranked, 1):
            lines.append(f"| {i} | {who} | {s:.4f} | {status} |" if s is not None
                         else f"| {i} | {who} | — | {status} |")
    else:
        lines.append("_No submissions yet._")
    lines.append("")

    fc = score_forecast()
    lines += ["## CO2 forecasting (Chapter 4.10) — MASE, lower is better", ""]
    if fc:
        lines += ["| rank | submission | MASE | status |", "|---|---|---|---|"]
        ranked = sorted(fc, key=lambda r: (r[1] is None, r[1] or 1e9))
        for i, (who, s, status) in enumerate(ranked, 1):
            lines.append(f"| {i} | {who} | {s:.4f} | {status} |" if s is not None
                         else f"| {i} | {who} | — | {status} |")
    else:
        lines.append("_No submissions yet (or truth file not configured)._")
    lines.append("")

    out = ROOT / "book" / "leaderboard_standings.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
