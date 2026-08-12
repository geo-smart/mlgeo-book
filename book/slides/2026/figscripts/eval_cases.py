"""Slide figure: four of notebook 6.3's twelve eval cases, at lecture scale.

Re-plots the notebook's 2x2 case panel (C01, C04, C06, C08) with lecture-size
fonts. Generator parameters are copied verbatim from the CASES list in
book/Chapter6-AgenticAI/6.3_build_an_eval_set.ipynb, so the planted truth
(velocity, offset day, offset size) matches the numbers quoted on the slides.

Regenerate: pixi run python book/slides/2026/figscripts/eval_cases.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from mlgeo_synth import gnss_series  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

# Verbatim from 6.3's CASES list (case id, slide title, generator parameters).
PANELS = [
    ("C01 — clean record, nothing planted",
     dict(n_years=10, velocity_mm_yr=12, eq_day=None, seed=101)),
    ("C04 — strong seasonal cycle + 20 mm offset",
     dict(n_years=10, velocity_mm_yr=6, annual_mm=8, eq_day=2000,
          coseismic_mm=20, postseismic_mm=0, seed=104)),
    ("C06 — low signal-to-noise + 25 mm offset",
     dict(n_years=10, velocity_mm_yr=3, white_mm=3, flicker_mm=6,
          eq_day=1500, coseismic_mm=25, postseismic_mm=0, seed=106)),
    ("C08 — 4 mm offset: edge of detectability",
     dict(n_years=10, velocity_mm_yr=12, eq_day=1500,
          coseismic_mm=4, postseismic_mm=0, seed=108)),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 8.2))
for ax, (title, params) in zip(axes.ravel(), PANELS):
    df = gnss_series(**params)
    ax.plot(np.arange(len(df)), df["disp_mm"].to_numpy(),
            lw=0.5, color="#1f77b4")
    if params.get("eq_day") is not None:
        ax.axvline(params["eq_day"], color="#b3402a", ls="--", lw=2.5)
    ax.set_title(title, loc="left", fontsize=21)
    ax.set_xlabel("day")
    ax.set_ylabel("disp (mm)")
fig.suptitle("Synthetic GNSS displacement (mlgeo_synth) — red dash: the planted offset day",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2)

out = (Path(__file__).resolve().parent.parent / "figs" / "6.3_build_an_eval_set"
       / "eval_cases_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
