"""Slide figure: aggregating the irregular multi-well network, two ways.

Re-plots the 2.6 multi-well comparison at lecture scale: forty years of
synthetic groundwater-well visits (mlgeo_synth, planted truth) aggregated to
a quarterly regional series naively (mean of raw heads — the result tracks
which wells were visited) and gap-aware (per-well anomalies plus
inverse-variance weights). Same data and seed as the executed notebook; RMS
errors against the planted regional signal are recomputed live and printed
into the panel titles.

Regenerate: pixi run python book/slides/2026/figscripts/well_aggregation.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mlgeo_synth  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

# Identical to the executed 2.6 notebook cells
wells, wtruth = mlgeo_synth.well_table(seed=42)

naive_q = wells.set_index("date")["head_m"].resample("QS").mean()

w = wells.assign(
    anom_m=wells["head_m"] - wells.groupby("well_id")["head_m"].transform("mean"),
    weight=1.0 / wells["sigma_m"] ** 2,
    quarter=wells["date"].dt.to_period("Q").dt.start_time,
)
w["wx"] = w["weight"] * w["anom_m"]
g = w.groupby("quarter")
aware_q = g["wx"].sum() / g["weight"].sum()


def rms_vs_regional(series):
    t_yr = (series.index - wells["date"].min()).days / 365.25
    reg = wtruth["regional"](t_yr)
    return np.sqrt(np.mean(((series - series.mean()) - (reg - reg.mean())) ** 2))


t_yr = (aware_q.index - wells["date"].min()).days / 365.25
regional_true = pd.Series(wtruth["regional"](t_yr), index=aware_q.index)

fig, ax = plt.subplots(2, 1, figsize=(15, 8.6), sharex=True)
ax[0].plot(naive_q - naive_q.mean(), color="#b3402a", lw=1.2)
ax[0].set_title("Average the raw heads: you measure the network, not the aquifer"
                f" — RMS {rms_vs_regional(naive_q):.2f} m", loc="left")
ax[1].plot(aware_q - aware_q.mean(), color="tab:blue", lw=1.4,
           label="per-well anomalies, weighted by measurement quality")
ax[1].plot(regional_true - regional_true.mean(), "k--", lw=1.4,
           label="true regional signal")
ax[1].set_title("Remove each well's own level first, then average"
                f" — RMS {rms_vs_regional(aware_q):.2f} m", loc="left")
ax[1].legend(fontsize=17, loc="upper right", framealpha=0.95)
for a in ax:
    a.set_ylabel("head anomaly (m)")

fig.suptitle("Synthetic 25-well groundwater network, 1981–2019 — mlgeo_synth "
             "(planted truth)",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.7)

out = (Path(__file__).resolve().parent.parent / "figs" / "2.6_resampling"
       / "well_aggregation_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
print(f"naive RMS {rms_vs_regional(naive_q):.2f} m, aware RMS {rms_vs_regional(aware_q):.2f} m")
