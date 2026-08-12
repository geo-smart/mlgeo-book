"""Slide figure: aliasing on the tide gauge — filter first, then subsample.

Re-plots the 2.6 downsampling comparison at lecture scale: six months of
synthetic hourly sea level (mlgeo_synth, planted truth) reduced to a daily
series three ways — naive midnight subsampling (which folds the M2 tide into
a ~14.8-day artifact), a daily mean, and a proper anti-alias decimation —
graded against the true subtidal signal. Same data and seed as the executed
notebook; RMS numbers are recomputed live and printed into the legend.

Regenerate: pixi run python book/slides/2026/figscripts/tide_aliasing.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402

import mlgeo_synth  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

# Identical to the executed 2.6 notebook cells
tide, tide_truth = mlgeo_synth.tide_gauge_series(n_days=180, seed=42)
sl = tide.set_index("time")["sea_level_m"]
subtidal = (tide.set_index("time")["sea_level_m"]
            - tide.set_index("time")["tide_m"]).resample("D").mean()

naive = sl.iloc[::24]
daily_mean = sl.resample("D").mean()
dec = signal.decimate(signal.decimate(sl.to_numpy(), 4, ftype="fir", zero_phase=True),
                      6, ftype="fir", zero_phase=True)
dec = pd.Series(dec, index=sl.index[::24])


def rms(s_daily):
    err = (s_daily - subtidal).dropna()
    return np.sqrt((err ** 2).mean())


fig, ax = plt.subplots(figsize=(15, 6.5))
ax.plot(naive, color="#b3402a", lw=1.2,
        label=f"midnight sample — RMS {rms(naive):.2f} m (aliased M2 tide)")
ax.plot(daily_mean, color="tab:orange", lw=1.8,
        label=f"daily mean — RMS {rms(daily_mean):.2f} m")
ax.plot(dec, color="tab:blue", lw=1.8,
        label=f"anti-alias filter, then subsample — RMS {rms(dec):.2f} m")
ax.plot(subtidal, "k--", lw=1.8, label="true subtidal sea level")
ax.set_ylabel("sea level (m)")
ax.set_title("Filter below the new Nyquist first — or the tide invents a fortnight",
             loc="left")
ax.legend(ncols=2, fontsize=17, loc="lower left", framealpha=0.95)

fig.suptitle("Synthetic hourly tide gauge, 180 days — mlgeo_synth (planted truth)",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = (Path(__file__).resolve().parent.parent / "figs" / "2.6_resampling"
       / "tide_aliasing_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
for name, s_daily in [("midnight", naive), ("daily mean", daily_mean), ("decimate", dec)]:
    print(f"{name:12s} RMS {rms(s_daily):.3f} m")
