"""Slide figure: what label-error structure costs (lecture 24, book 4.5 §3.1).

Two panels at lecture scale. Left: clean-test accuracy for the same MLP
trained on clean labels, 30% uniform flips, and 30% adjacent-class
disagreement (the two-mappers experiment), with the min-max span over the
three seeds. Right: the clean-test confusion matrix after training on
adjacent-class disagreement.

Every number is copied from the executed outputs of
book/Chapter4-DeepLearning/mlgeo_4.5_ModelTraining.ipynb (Sections 3.1
"Label noise" and "Structured disagreement"); re-verify there after any
notebook re-run. Data: synthetic lithology table (mlgeo_synth.geochem_table).

Regenerate: pixi run python book/slides/2026/figscripts/pillar1_label_quality.py
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

# Executed-notebook numbers (clean-test accuracy, 3000 training samples,
# mean over 3 seeds with min-max seed span).
conditions = ["no label\nerrors", "30% uniform\nflips", "30% adjacent\ndisagreement"]
acc = np.array([0.945, 0.910, 0.820])
span_lo = np.array([0.943, 0.899, 0.799])
span_hi = np.array([0.946, 0.919, 0.850])

# Clean-test confusion after training on adjacent-class disagreement
# (rows = true class, columns = predicted class; fractions of true class).
classes = ["granite", "basalt", "andesite"]
confusion = np.array([
    [0.86, 0.01, 0.14],
    [0.01, 0.82, 0.16],
    [0.29, 0.10, 0.61],
])

fig, (axl, axr) = plt.subplots(
    1, 2, figsize=(15, 6.2),
    gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.35})

# Left: same error rate, different structure.
colors = ["#116b66", "#1f77b4", "#b3402a"]
bars = axl.bar(conditions, acc, color=colors, width=0.62)
axl.errorbar(conditions, acc, yerr=[acc - span_lo, span_hi - acc],
             fmt="none", ecolor="#3a3630", elinewidth=2.2, capsize=7, capthick=2.2)
for b, a in zip(bars, acc):
    axl.text(b.get_x() + b.get_width() / 2, a + 0.017, f"{a:.3f}",
             ha="center", va="bottom", fontsize=20, fontweight="bold")
axl.set_ylim(0.75, 1.0)
axl.set_ylabel("accuracy on clean test set")
axl.set_title("Same 30% error rate,\nthree times the damage", loc="left")
axl.tick_params(axis="x", labelsize=16)
axl.text(0.98, 0.98, "whiskers: min-max over 3 seeds",
         transform=axl.transAxes, ha="right", va="top",
         fontsize=16, color="#6e675c")
axl.spines[["top", "right"]].set_visible(False)

# Right: where the points went (sequential single-hue heatmap).
im = axr.imshow(confusion, cmap="Blues", vmin=0, vmax=1)
axr.set_xticks(range(3), classes)
axr.set_yticks(range(3), classes)
axr.set_xlabel("predicted")
axr.set_ylabel("true")
axr.set_title("Where the errors went:\neverything slides to andesite", loc="left")
for i in range(3):
    for j in range(3):
        v = confusion[i, j]
        axr.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=20,
                 fontweight="bold", color="white" if v > 0.5 else "#3a3630")
cb = fig.colorbar(im, ax=axr, fraction=0.046, pad=0.03)
cb.set_label("fraction of true class", fontsize=17)
cb.ax.tick_params(labelsize=15)

fig.suptitle("Synthetic lithology table (mlgeo_synth.geochem_table) — executed 4.5 notebook",
             x=0.01, y=0.995, ha="left", va="top",
             fontsize=18, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=2.5)
fig.subplots_adjust(top=0.80)

out = (Path(__file__).resolve().parent.parent / "figs" / "mlgeo_4.5_ModelTraining"
       / "label_quality_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
