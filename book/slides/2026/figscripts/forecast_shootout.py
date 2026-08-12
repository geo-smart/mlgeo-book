"""Slide figure: the CO2 forecasting shootout ranking with seed spread
(lecture 26, book 4.10 §3.6 + §3.9).

Horizontal MASE bars on a log axis, one per contestant, ordered by skill.
Stochastic entries (LSTM, transformer) show their seed mean with a min-max
whisker; deterministic entries carry a "no seed consumed" note — including
LightGBM, whose five seeds returned the identical score (std 0.000).

Every number is copied from the executed outputs of
book/Chapter4-DeepLearning/mlgeo_4.10_timeseriesforecast.ipynb (the 3.6
comparison table and the 3.9 seed-spread table); re-verify there after any
notebook re-run. Data: Mauna Loa monthly CO2 (NOAA Global Monitoring
Laboratory), temporal split, 48-month test window 2019-09 .. 2023-08.

Regenerate: pixi run python book/slides/2026/figscripts/forecast_shootout.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

# Executed-notebook numbers. Deterministic entries: single-run MASE (3.6 table).
# Stochastic entries: seed mean with (min, max) over seeds (3.9 table).
entries = [  # (name, mase, (lo, hi) or None, note)
    ("naive persistence",  6.090, None, "deterministic"),
    ("seasonal naive",     5.436, None, "deterministic"),
    ("LightGBM",           0.939, None, "5 seeds, identical score (std 0.000)"),
    ("LSTM",               0.827, (0.808, 0.856), "3 seeds, min-max"),
    ("transformer",        0.356, (0.327, 0.403), "3 seeds, min-max"),
    ("SARIMA",             0.256, None, "deterministic"),
]

names = [e[0] for e in entries]
mase = np.array([e[1] for e in entries])
ypos = np.arange(len(entries))[::-1]

fig, ax = plt.subplots(figsize=(15, 6.4))
colors = ["#9c948a" if "naive" in n else ("#116b66" if n == "SARIMA" else "#1f77b4")
          for n in names]
ax.barh(ypos, mase, color=colors, height=0.62)
for y, (name, m, span, note) in zip(ypos, entries):
    label_x = (span[1] if span is not None else m) * 1.09
    if span is not None:
        ax.errorbar([m], [y], xerr=[[m - span[0]], [span[1] - m]], fmt="none",
                    ecolor="#3a3630", elinewidth=2.4, capsize=7, capthick=2.4)
    ax.text(label_x, y, f"{m:.3f}", va="center", fontsize=20, fontweight="bold")
    ax.text(0.013, y, f"{name}  ·  {note}", va="center", fontsize=17,
            color="white", fontweight="bold")

ax.axvline(1.0, color="#b3402a", lw=2.5, ls="--")
ax.text(1.06, 0.5, "MASE = 1: tie with the\nnaive one-step rule",
        fontsize=16, color="#b3402a", ha="left", va="center", fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(0.012, 14)
ax.set_xlabel("MASE on the 48-month test window (lower is better, log scale)")
ax.set_yticks([])
ax.set_title("Four years of CO$_2$ ahead: the ranking, with its seed spread", loc="left")
ax.spines[["top", "right", "left"]].set_visible(False)

fig.suptitle("Mauna Loa monthly CO$_2$ — NOAA Global Monitoring Laboratory; "
             "train 1958-2019, test 2019-2023 (executed 4.10 notebook)",
             x=0.01, ha="left", fontsize=17, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = (Path(__file__).resolve().parent.parent / "figs" / "mlgeo_4.10_timeseriesforecast"
       / "shootout_mase_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
