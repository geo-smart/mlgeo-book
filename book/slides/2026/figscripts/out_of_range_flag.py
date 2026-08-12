"""Slide figure: error and ensemble disagreement across the training bound
(lecture 25, book 4.5 §4.6).

Left: error rate per compositional-range bin. Right: mean ensemble
disagreement per bin. The training bound (|differentiation index| = 1) is
where the mapped range of the field campaign ends; both quantities rise
past it, and the annotation carries the silent-failure count.

Every number is copied from the executed outputs of
book/Chapter4-DeepLearning/mlgeo_4.5_ModelTraining.ipynb (Section 4.6
out-of-range table: n / error rate / mean disagreement per range bin, and
the 77-errors / 1-silent-failure printout); re-verify there after any
notebook re-run. Data: synthetic lithology table (mlgeo_synth.geochem_table).

Regenerate: pixi run python book/slides/2026/figscripts/out_of_range_flag.py
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

# Executed-notebook numbers (Section 4.6 ood_table).
bins = ["in range\n(mapped)", "just past\nthe bound", "far out\nof range"]
n = [2046, 824, 130]
error_rate = np.array([0.049, 0.080, 0.085])
disagreement = np.array([0.013, 0.017, 0.021])

fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 5.9), sharex=True,
                               gridspec_kw={"wspace": 0.32})

for ax, vals, color, ylabel, title in [
    (axl, error_rate, "#b3402a", "error rate on fresh samples",
     "Error nearly doubles\npast the mapped range"),
    (axr, disagreement, "#1f77b4", "mean ensemble disagreement",
     "Disagreement rises with it —\nthe flag mostly works"),
]:
    bars = ax.bar(bins, vals, color=color, width=0.6)
    for b, v, count in zip(bars, vals, n):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.03, f"{v:.3f}",
                ha="center", va="bottom", fontsize=20, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v * 0.07, f"n = {count}",
                ha="center", va="bottom", fontsize=15, color="white")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_ylim(0, vals.max() * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0.5, color="#3a3630", lw=2, ls="--",
               ymax=0.99 if ax is axl else 0.62)
    ax.text(0.44, vals.max() * 0.5, "training bound", fontsize=15,
            color="#3a3630", ha="right", rotation=90, va="center")

axr.text(0.03, 0.99,
         "77 out-of-range errors:\ndisagreement flagged 76,\n1 silent failure",
         transform=axr.transAxes, ha="left", va="top", fontsize=19,
         fontweight="bold", color="#116b66")

fig.suptitle("Ensemble trained only on typical compositions, tested on the full range — "
             "synthetic (mlgeo_synth.geochem_table), executed 4.5 notebook",
             x=0.01, y=0.995, ha="left", va="top",
             fontsize=17, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.5)
fig.subplots_adjust(top=0.78)

out = (Path(__file__).resolve().parent.parent / "figs" / "mlgeo_4.5_ModelTraining"
       / "out_of_range_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
