"""Slide figure: the three-way split drawn on a real GNSS time series.

Each panel shows the same daily east-displacement record from GNSS station
P395 (Parkfield, California), with points colored train/validation for one
representative fold of each splitting strategy — so the leakage is visible
in the data, not abstracted into index bars. Falls back to the synthetic
mlgeo_synth.gnss_series if the P395 cache is unavailable.

Regenerate: pixi run python book/slides/2026/figscripts/gnss_split_ladder.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from sklearn.model_selection import KFold, ShuffleSplit, TimeSeriesSplit  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})


BOOK = Path(__file__).resolve().parents[3]


def load_p395():
    """Real NGL tenv3 record from the repo data cache that 1.7 populates."""
    import pandas as pd

    path = next(p for p in (
        BOOK / "Chapter1-GettingStarted" / "data" / "P395.tenv3",
        BOOK / "Chapter2-DataManipulation" / "data" / "P395.tenv3",
    ) if p.exists())
    df = pd.read_csv(path, sep=r"\s+")
    t = df["yyyy.yyyy"].to_numpy()
    east_mm = df["__east(m)"].to_numpy() * 1000.0
    east_mm -= east_mm[0]
    keep = t >= t[-1] - 6.0          # last six years
    t, east_mm = t[keep], east_mm[keep]
    return t[::5], east_mm[::5], ("GNSS station P395 (Oregon Coast Range) — daily east position, "
                                  "NGL / Nevada Geodetic Laboratory")


def load_synthetic():
    import sys

    sys.path.insert(0, str(BOOK))
    from mlgeo_synth import gnss_series

    df = gnss_series(n_years=6, seed=11)
    t = 2020.75 + np.arange(len(df)) / 365.25
    return t[::5], df["disp_mm"].to_numpy()[::5], "Synthetic GNSS displacement (mlgeo_synth)"


try:
    t, y, tag = load_p395()
except Exception as exc:  # noqa: BLE001 - any failure falls back to synthetic
    print(f"P395 unavailable ({exc}); using synthetic fallback")
    t, y, tag = load_synthetic()

N = len(t)
X = np.zeros((N, 1))
panels = [
    ("Random 20% of days held out for validation", "R² ≈ 0.9  (a lie)",
     list(ShuffleSplit(n_splits=5, test_size=0.2, random_state=0).split(X))[0]),
    ("Shuffled five-fold — every day near a training day", "R² ≈ 0.9  (same lie)",
     list(KFold(n_splits=5, shuffle=True, random_state=0).split(X))[2]),
    ("Train on the past, validate on the future", "R² ≈ −0.3  (honest)",
     list(TimeSeriesSplit(n_splits=5).split(X))[4]),
]

fig, axes = plt.subplots(3, 1, figsize=(15, 8.6), sharex=True, sharey=True)
for ax, (name, verdict, (tr, va)) in zip(axes, panels):
    ax.scatter(t[tr], y[tr], s=14, color="#1f77b4", label="train")
    ax.scatter(t[va], y[va], s=14, color="#ff7f0e", label="validation")
    ax.set_title(name, loc="left")
    ax.text(0.99, 0.93, verdict, transform=ax.transAxes, ha="right", va="top",
            fontsize=19, fontweight="bold",
            color="#b3402a" if "lie" in verdict else "#116b66")
    ax.set_ylabel("east (mm)")
axes[0].legend(loc="lower left", ncol=2, fontsize=17, framealpha=0.95)
axes[-1].set_xlabel("year")
fig.suptitle(tag, x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.7)

out = (Path(__file__).resolve().parent.parent / "figs" / "3.8_robust_training"
       / "gnss_split_ladder_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
