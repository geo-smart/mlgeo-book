"""Slide figure: a detection limit writes itself into a displacement record.

Re-plots the 2.4 censored-values demonstration at lecture scale: a synthetic
GNSS displacement series (mlgeo_synth, planted truth) reported by a sensor
with a 5 mm floor, plus the year-1 histogram whose heap at the limit is the
fingerprint of censoring. Same data and seeds as the executed notebook.

Regenerate: pixi run python book/slides/2026/figscripts/censored_series.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

import mlgeo_synth  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

# Identical to the executed 2.4 notebook cell
gnss = mlgeo_synth.gnss_series(n_years=4, velocity_mm_yr=12.0, seed=7)
censored, truth = mlgeo_synth.degrade_series(gnss, detection_limit_mm=5.0, seed=7)
t = np.arange(len(censored)) / 365.25

fig, ax = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[1.5, 1])
ax[0].plot(t, truth["clean"], lw=0.7, color="gray", label="true (uncensored)")
ax[0].plot(t, censored["disp_mm"], lw=0.7, color="#b3402a", label="reported")
ax[0].axhline(5.0, color="k", ls="--", lw=1.5, label="detection limit (5 mm)")
ax[0].set_xlabel("time (yr)")
ax[0].set_ylabel("displacement (mm)")
ax[0].set_title("The sensor reports its floor, not the ground", loc="left")
ax[0].legend(fontsize=17, loc="upper left")

ax[1].hist(censored["disp_mm"][:365], bins=40, color="#b3402a", alpha=0.6,
           label="reported, year 1")
ax[1].hist(truth["clean"][:365], bins=40, histtype="step", lw=2, color="gray",
           label="true, year 1")
ax[1].set_xlabel("displacement (mm)")
ax[1].set_ylabel("count")
ax[1].set_title("The fingerprint: a heap at the limit", loc="left")
ax[1].legend(fontsize=17)

fig.suptitle("Synthetic GNSS displacement, 12 mm/yr — mlgeo_synth (planted truth)",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = (Path(__file__).resolve().parent.parent / "figs" / "2.4_dataframes_prep"
       / "censored_series_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")

# sanity: reproduce the numbers quoted on the slides
print(f"fraction censored: {censored['censored'].mean():.3f}")
print(f"trend on reported:   {np.polyfit(t, censored['disp_mm'], 1)[0]:.2f} mm/yr")
print(f"trend on uncensored: {np.polyfit(t, truth['clean'], 1)[0]:.2f} mm/yr")
