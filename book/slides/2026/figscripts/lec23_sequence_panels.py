"""Slide figures: the sequence-forecasting task and the vanishing gradient (lecture 23).

Figure 1  gnss_forecast_task_slide.png — the notebook's 10-year synthetic GNSS
          series with its known components, and one context/horizon window.
          Deterministic re-plot of mlgeo_4.4_RNN.ipynb sections 2-3 (same
          generator call: mlgeo_synth.gnss_series(n_years=10, eq_day=1800,
          seed=42); no training involved).
Figure 2  vanishing_gradient_slide.png — ninety multiplications: why the
          gradient that reaches day 1 of a 90-day window vanishes or explodes.

Regenerate: pixi run python book/slides/2026/figscripts/lec23_sequence_panels.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "legend.fontsize": 17,
})

BOOK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BOOK))
import mlgeo_synth  # noqa: E402

FIGS = Path(__file__).resolve().parent.parent / "figs" / "mlgeo_4.4_RNN"
FIGS.mkdir(parents=True, exist_ok=True)

# ------------- Figure 1: the forecasting task on the synthetic GNSS series -------------
gnss = mlgeo_synth.gnss_series(n_years=10.0, eq_day=1800, seed=42)
WINDOW, HORIZON = 90, 30
split = int(0.8 * len(gnss))          # notebook's temporal split: 8 years / 2 years
w0 = split + 300 + 0                  # start of the notebook's example pair (i=300)

fig, axes = plt.subplots(2, 1, figsize=(15, 8.4), sharex=True)

ax = axes[0]
ax.plot(gnss["date"], gnss["disp_mm"], color="#333333", lw=0.7)
ax.axvspan(gnss["date"].iloc[split], gnss["date"].iloc[-1],
           color="#ff7f0e", alpha=0.12)
ax.text(gnss["date"].iloc[split + 40], gnss["disp_mm"].min() + 12,
        "validation:\nlast 2 years", fontsize=18, color="#b3402a",
        fontweight="bold", va="bottom")
ax.text(gnss["date"].iloc[30], gnss["disp_mm"].max() * 0.92,
        "training: the first 8 years", fontsize=18, color="#1f77b4",
        fontweight="bold", va="top")
ax.set_ylabel("east displacement (mm)")
ax.set_title("Train on the past, validate on the future", loc="left")

ax = axes[1]
ax.plot(gnss["date"], gnss["trend_mm"], color="#4477AA", lw=2.5,
        label="tectonic trend (~12 mm/yr)")
ax.plot(gnss["date"], gnss["seasonal_mm"], color="#CCBB44", lw=2,
        label="seasonal loading")
ax.plot(gnss["date"], gnss["eq_mm"], color="#EE6677", lw=2.5,
        label="earthquake step + postseismic")
ax.set_ylabel("component (mm)")
ax.set_xlabel("year")
ax.set_title("What the record is made of (known, because we built it)", loc="left")
ax.legend(loc="upper left", ncol=1, framealpha=0.95)

for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("Synthetic GNSS displacement, 10 years daily (mlgeo_synth)",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.9)
out1 = FIGS / "gnss_forecast_task_slide.png"
fig.savefig(out1, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out1}")

# ------------- Figure 2: ninety multiplications -------------
steps = np.arange(0, 91)
fig, ax = plt.subplots(figsize=(15, 6.8))
for factor, color, label in [(0.95, "#b3402a", "each step multiplies by 0.95 → vanishes"),
                             (1.00, "#6e675c", "each step multiplies by exactly 1"),
                             (1.05, "#7b3294", "each step multiplies by 1.05 → explodes")]:
    ax.semilogy(steps, factor ** steps.astype(float), lw=3, color=color, label=label)
ax.axhline(1, color="#6e675c", lw=0.8, alpha=0.4)
ax.set_xlabel("time steps the gradient travels back through")
ax.set_ylabel("surviving gradient (relative, log scale)")
ax.set_title("Ninety small multiplications: almost nothing reaches day 1", loc="left")
ax.legend(loc="center left", framealpha=0.95)
ax.grid(alpha=0.3, which="both")
ax.set_xlim(0, 90)
fig.tight_layout()
out2 = FIGS / "vanishing_gradient_slide.png"
fig.savefig(out2, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out2}")
