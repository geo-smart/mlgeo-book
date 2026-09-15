"""Replaces the PhD Comics "final.doc" strip in 1.5 with an original cartoon.

Same joke, drawn by us: a folder where every save is a new filename, so
nobody can say which one is current or what changed between any two.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

TEAL, ORANGE, GREY = "#116b66", "#b3402a", "#6e675c"
files = [
    ("paper.docx",                          "Sep 12  09:14", GREY),
    ("paper_v2.docx",                       "Sep 14  17:40", GREY),
    ("paper_final.docx",                    "Sep 20  23:58", GREY),
    ("paper_final_REAL.docx",               "Sep 21  00:31", GREY),
    ("paper_final_REAL_advisor_edits.docx", "Sep 23  11:05", GREY),
    ("paper_final_v2_USE_THIS_ONE.docx",    "Sep 25  02:12", ORANGE),
    ("paper_final_v2_USE_THIS_ONE (1).docx","Sep 25  02:13", ORANGE),
]
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.4, 0.6), 9.2, 8.8, boxstyle="round,pad=0.1",
                            fc="white", ec=GREY, lw=1.2))
ax.text(0.8, 8.9, "thesis/", fontsize=12, weight="bold", color=TEAL, va="center")
ax.plot([0.7, 9.3], [8.45, 8.45], color=GREY, lw=0.8)
for k, (name, stamp, col) in enumerate(files):
    y = 7.8 - k * 1.02
    ax.add_patch(FancyBboxPatch((0.95, y-0.22), 0.36, 0.44, boxstyle="round,pad=0.02", fc="#cfe3ee", ec=GREY, lw=0.6))
    ax.text(1.55, y, name, fontsize=9.5, family="monospace", va="center", color=col)
    ax.text(9.25, y, stamp, fontsize=8, va="center", ha="right", color=GREY)
ax.text(5.0, 0.15, "Which one is current? What changed between any two? Nobody knows.",
        fontsize=8.5, ha="center", color=ORANGE, style="italic")
fig.savefig("book/Chapter1-GettingStarted/versioning-by-filename.png",
            bbox_inches="tight", facecolor="white")
print("wrote book/Chapter1-GettingStarted/versioning-by-filename.png")
