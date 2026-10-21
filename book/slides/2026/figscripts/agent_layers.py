"""Slide diagram: the layers an agent run passes through, with one failure message per layer.

A language model proposes an action; a tool executes it inside a software
environment, inside a sandbox, on hardware; the observation goes back to the
model. The three error messages are the ones students classify aloud in
lecture 3 ("permission denied", "module not found", "CUDA out of memory").
No data: a schematic. The example task is the lecture's P395 calibration.

Regenerate: pixi run python book/slides/2026/figscripts/agent_layers.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, BAD = "#26241f", "#6e675c", "#b3402a"
MODEL_C = "#2a78d6"
OUT = Path(__file__).resolve().parent.parent / "figs" / "lit_agents" / "agent_layers.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(16, 6.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16); ax.set_ylim(0, 6.2); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=2.0, r=0.18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


# Left: the specification and the model
box(0.15, 4.35, 4.3, 1.7, "#ffffff", MUTED, lw=1.5)
ax.text(0.4, 5.75, "Specification (you)", fontsize=19, fontweight="bold", color=INK, va="top")
ax.text(0.4, 5.25, "\u201cReport the east velocity of\nP395 in mm/yr, IGS20 frame\u201d",
        fontsize=17, color=INK, va="top", style="italic")
box(0.15, 0.6, 4.3, 3.1, "#e8f1fc", MODEL_C, lw=2.5)
ax.text(0.4, 3.4, "Language model", fontsize=21, fontweight="bold", color=INK, va="top")
ax.text(0.4, 2.8, "chooses the next action\nreads each observation\ndecides when to stop\nwrites the answer",
        fontsize=17, color=INK, va="top", linespacing=1.45)
ax.add_patch(FancyArrowPatch((2.3, 4.35), (2.3, 3.72), arrowstyle="-|>", mutation_scale=22, color=MUTED, lw=2))

# Right: nested layers, outermost first; one failure message per layer on its header line
layers = [
    ("Hardware", "CPU, GPU, memory, disk", "CUDA out of memory", "#f3efe8", "#b9b2a6"),
    ("Sandbox", "allowed files, hosts, write locations", "Permission denied", "#ece6dc", "#a39b8e"),
    ("Environment", "Python + packages (pixi.lock)", "ModuleNotFoundError", "#e2dbcf", "#8b8376"),
]
for k, (name, what, err, fc, ec) in enumerate(layers):
    x, y = 6.2 + 0.3 * k, 0.1 + 0.25 * k
    w, h = 15.85 - 0.3 * k - x, 6.1 - 0.75 * k - y
    box(x, y, w, h, fc, ec)
    ty = y + h - 0.2
    ax.text(x + 0.25, ty, name, fontsize=19, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.25 + 0.17 * len(name) + 0.35, ty - 0.04, what, fontsize=16, color=MUTED, va="top")
    ax.text(x + w - 0.2, ty - 0.04, err, fontsize=15, color=BAD, va="top", ha="right", family="monospace")
tx, ty_, tw, th = 7.1, 0.85, 14.95 - 7.1, 2.75
box(tx, ty_, tw, th, "#ffffff", INK, lw=2.2)
ax.text(tx + 0.25, ty_ + th - 0.22, "Tools", fontsize=19, fontweight="bold", color=INK, va="top")
ax.text(tx + 0.25, ty_ + th - 0.85, "read a file   \u00b7   run Python   \u00b7   fetch a URL\nquery a catalog   \u00b7   write a file",
        fontsize=17, color=INK, va="top", linespacing=1.5)
ax.text(tx + 0.25, ty_ + 0.3, "each call returns output, warnings, or an error", fontsize=16, color=MUTED, va="bottom")

# The loop: action out, observation back
ax.add_patch(FancyArrowPatch((4.5, 2.75), (tx - 0.05, 2.75), arrowstyle="-|>", mutation_scale=26,
                             color=MODEL_C, lw=3, connectionstyle="arc3,rad=-0.15"))
ax.text(4.75, 3.2, "action", fontsize=17, color=INK, fontweight="bold")
ax.add_patch(FancyArrowPatch((tx - 0.05, 1.45), (4.5, 1.45), arrowstyle="-|>", mutation_scale=26,
                             color=MODEL_C, lw=3, connectionstyle="arc3,rad=-0.15"))
ax.text(4.52, 0.62, "observation", fontsize=16, color=INK, fontweight="bold")

fig.savefig(OUT, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {OUT}")
