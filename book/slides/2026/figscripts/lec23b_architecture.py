"""Slide figures for lecture 23b: architecture follows the data, and the rubric for the AI assistant.

  gnss_two_structures.png  one geoscience question, two data structures: GNSS position as a table of six
                           numbers (notebook 4.2, real station P395, numbers from
                           figs/lec21b_train_evaluate/numbers.json) and as a 90-day sequence (notebook 4.4,
                           synthetic GNSS series, validation MAE as reported in lecture 23)
  decision_guide.png       questions that lead from the data to a kind of network (schematic)
  assistant_workflow.png   how the rubric of book section 4.8 drives and checks an AI assistant (schematic)

Regenerate: pixi run python book/slides/2026/figscripts/lec23b_architecture.py
(run lec21b_train_evaluate.py first)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs" / "lec23b_architecture"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 22, "axes.titlesize": 24, "axes.titleweight": "bold", "axes.labelsize": 22,
                     "xtick.labelsize": 19, "ytick.labelsize": 19, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREEN, RED, PURPLE, INK, MUTED, GRAY = "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#26241f", "#6e675c", "#b8b1a4"

# --- one question, two structures: GNSS -------------------------------------------------------
p395 = json.loads((HERE.parent / "figs" / "lec21b_train_evaluate" / "numbers.json").read_text())["test_mae_mm"]
table = [("persistence", p395["persistence"]["mean"], GRAY), ("linear regression", p395["linear regression"]["mean"], BLUE),
         ("fully connected,\npredicts the change", p395["network, predicts the change, MAE"]["mean"], RED)]
seq = [("persistence", 2.10, GRAY), ("recurrent (RNN)", 1.83, GREEN), ("LSTM", 1.86, GREEN),
       ("attention", 1.87, GREEN), ("Transformer", 1.84, GREEN)]
fig, (a, b) = plt.subplots(1, 2, figsize=(20, 7.4), gridspec_kw={"wspace": 0.55})
for ax, rows, title, xlab in [(a, table, "As a table: 6 numbers\nGNSS P395, real (4.2)", "Test error, 30-day position (mm)"),
                              (b, seq, "As a sequence: 90 days\nsynthetic GNSS (4.4)", "Validation error, 30-day horizon (mm)")]:
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ax.barh(range(len(rows)), vals, color=[r[2] for r in rows], height=0.6)
    for k, v in enumerate(vals):
        ax.text(v + 0.03 * max(vals), k, f"{v:.2f}", va="center", fontsize=21)
    ax.set_yticks(range(len(rows)), names)
    ax.invert_yaxis()
    ax.set(xlabel=xlab, xlim=(0, max(vals) * 1.3))
    ax.set_title(title, loc="left")
fig.text(0.5, -0.06, "Left: the network ties or loses to linear regression. Right: every sequence network beats persistence by about 13%.\n"
         "Different stations and windows: compare each model with its own baselines, not across panels.", ha="center", fontsize=17, color=MUTED)
fig.savefig(FIGS / "gnss_two_structures.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- decision guide -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(21, 8.8))
ax.set(xlim=(0, 21), ylim=(0, 8.8)); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=21, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.12", fc=fc, ec=ec, lw=2.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color)


def arrow(x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=24, color=MUTED, lw=2.5))
    if label:
        ax.text((x0 + x1) / 2 + 0.1, (y0 + y1) / 2 + 0.15, label, fontsize=18, color=MUTED)


box(2.4, 7.4, 4.2, 1.2, "One data sample:\nwhat is it?", "#f3efe6", INK, bold=True)
qs = [(6.1, "a table of features,\nno order between them", "Fully connected", BLUE, "tree models and linear first"),
      (4.55, "a grid: waveform, image, map;\npatterns can move", "Convolutional", GREEN, "classical filter or template first"),
      (3.0, "a sequence in time;\nthe past informs the future", "Recurrent, attention", RED, "persistence, seasonal first"),
      (1.45, "stations or cells\nlinked in a network", "Graph network", PURPLE, "spatial interpolation first")]
for y, q, net, c, base in qs:
    arrow(2.4, 6.75, 2.4, y + 0.1)
    box(6.4, y, 5.4, 1.15, q, "white", c, fs=19)
    arrow(9.2, y, 11.0, y)
    box(13.3, y, 4.6, 1.0, net, c, c, fs=20, bold=True, color="white")
    ax.text(15.9, y, base, ha="left", va="center", fontsize=18, color=MUTED, style="italic")
ax.text(4.9, 8.1, "Ask in this order; combine when the data have two structures\n(a map that changes in time: convolution and recurrence).",
        ha="left", va="center", fontsize=18, color=INK)
ax.text(15.9, 6.95, "baseline to beat", ha="left", va="center", fontsize=19, color=MUTED, fontweight="bold")
fig.savefig(FIGS / "decision_guide.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- the assistant workflow ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(21, 7.6))
ax.set(xlim=(0, 21), ylim=(0, 7.6)); ax.axis("off")
steps = [(2.2, "Rubric 4.8", "items, checks,\nevidence", BLUE),
         (6.4, "Assistant", "builds, trains,\nevaluates", PURPLE),
         (10.6, "Report", "one row per item:\npass / fail + evidence", GREEN),
         (14.8, "Scientist", "checks the evidence;\nreruns the scripts", RED),
         (19.0, "Decision", "use, revise,\nor keep the baseline", INK)]
for k, (x, title, sub, c) in enumerate(steps):
    box(x, 5.0, 3.6, 1.0, title, c, c, fs=23, bold=True, color="white")
    ax.text(x, 3.9, sub, ha="center", va="top", fontsize=19, color=INK)
    if k < len(steps) - 1:
        arrow(x + 1.85, 5.0, steps[k + 1][0] - 1.85, 5.0)
ax.add_patch(FancyArrowPatch((14.8, 5.6), (6.4, 5.6), connectionstyle="arc3,rad=0.35", arrowstyle="-|>", mutation_scale=24, color=RED, lw=2.5))
ax.text(10.6, 7.25, "fail: back to the assistant with the failed items", ha="center", fontsize=19, color=RED)
box(10.5, 1.2, 19.4, 1.25, "Test the assistant on cases with a known answer: the P395 range trap (A7), the synthetic-to-real detector (E5),\n"
    "the wrong seasonal baseline (C3.3), the network that ties logistic regression (E3).", "#f3efe6", MUTED, fs=17)
fig.savefig(FIGS / "assistant_workflow.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("written")
