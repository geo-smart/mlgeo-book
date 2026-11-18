"""Slide figure: the six sections of the deep-learning rubric (book section 4.8), with one item and its evidence each.

Schematic, no data. The items quoted are copied from book/Chapter4-DeepLearning/mlgeo_4.8_dl_rubric.md.
Writes figs/lec21b_train_evaluate/rubric_sections.png.
Regenerate: pixi run python book/slides/2026/figscripts/lec21b_rubric_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lec21b_train_evaluate"
INK, MUTED = "#26241f", "#6e675c"
COLORS = ["#0072B2", "#CC79A7", "#009E73", "#E69F00", "#D55E00", "#56B4E9"]
sections = [
    ("A", "Framing", "A7: test samples\ninside the\ntraining range?", "ranges"),
    ("B", "Network", "B: input structure\n→ kind of layer", "sentence"),
    ("C", "Design", "C1.2: last\nlayer fits Y", "code"),
    ("D", "Training", "D4: training and\nvalidation loss\nevery epoch", "figure"),
    ("E", "Evaluation", "E3: gap below\nthe seed spread\nis a tie", "numbers"),
    ("F", "Reporting", "F3: items the\nagent could\nnot check", "list"),
]
fig, ax = plt.subplots(figsize=(21, 7.4))
ax.set(xlim=(0, 21), ylim=(0, 7.4)); ax.axis("off")
w, gap = 3.05, 0.42
for k, (letter, name, item, evidence) in enumerate(sections):
    x0 = 0.2 + k * (w + gap)
    c = COLORS[k]
    ax.add_patch(FancyBboxPatch((x0, 4.6), w, 2.2, boxstyle="round,pad=0.08", fc=c, ec=c, lw=2))
    ax.text(x0 + w / 2, 6.15, letter, ha="center", va="center", fontsize=40, fontweight="bold", color="white")
    ax.text(x0 + w / 2, 5.15, name, ha="center", va="center", fontsize=21, fontweight="bold", color="white")
    ax.add_patch(FancyBboxPatch((x0, 1.55), w, 2.6, boxstyle="round,pad=0.08", fc="white", ec=c, lw=3))
    ax.text(x0 + w / 2, 3.2, item, ha="center", va="center", fontsize=19, color=INK)
    ax.text(x0 + w / 2, 1.95, evidence, ha="center", va="center", fontsize=19, color=c, fontweight="bold")
    if k < len(sections) - 1:
        ax.add_patch(FancyArrowPatch((x0 + w + 0.03, 5.7), (x0 + w + gap - 0.03, 5.7), arrowstyle="-|>", mutation_scale=22, color=MUTED, lw=2.5))
ax.text(10.5, 0.75, "Top: section. Middle: one item. Bottom: the evidence. The assistant reports each item: pass or fail, with its evidence.", ha="center", fontsize=22, color=INK)
fig.savefig(FIGS / "rubric_sections.png", dpi=150, bbox_inches="tight")
print("written")
