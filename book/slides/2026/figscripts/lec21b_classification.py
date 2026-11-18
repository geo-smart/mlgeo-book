"""Slide figure for lecture 21b: the classification of notebook 4.1, at lecture scale.

Runs the code cells of notebook 4.1 (book/Chapter4-DeepLearning/mlgeo_4.1_NN.ipynb) up to the
"remove dropout" Your turn, in order, so that every number is the notebook's: Pacific Northwest
seismic sources (Zenodo record 14025693), 61 features, 2,400 / 800 / 800 events.
Plots the loss curves of the three networks and the test confusion matrix of the one-hidden-layer
network. Numbers to figs/lec21b_train_evaluate/numbers_classification.json.

Regenerate: pixi run python book/slides/2026/figscripts/lec21b_classification.py
"""
import json
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
NB = HERE.parents[2] / "Chapter4-DeepLearning" / "mlgeo_4.1_NN.ipynb"
FIGS = HERE.parent / "figs" / "lec21b_train_evaluate"
BLUE, ORANGE, GREEN, RED, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c"

cells = json.loads(NB.read_text())["cells"]
ns = {"__name__": "__nb41__"}
cwd = os.getcwd()
os.chdir(tempfile.mkdtemp())                 # the notebook writes checkpoints to ./checkpoints
plt.show = lambda *a, **k: None
for c in cells:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.lstrip().startswith(("%", "!")):
        continue
    exec(compile(src, "nb41", "exec"), ns)
    plt.close("all")
    if "history_nodrop" in ns and "test_acc_nodrop" in ns:
        break
os.chdir(cwd)

from sklearn.metrics import confusion_matrix  # noqa: E402

cm = confusion_matrix(ns["y_test"], ns["y_pred"])
runs = [("1 hidden layer", ns["history"], ns["test_acc"]),
        ("3 layers, dropout", ns["history_deep"], ns["test_acc_deep"]),
        ("3 layers, no dropout", ns["history_nodrop"], ns["test_acc_nodrop"])]

plt.rcParams.update({"font.size": 21, "axes.titlesize": 21, "axes.titleweight": "bold", "axes.labelsize": 21,
                     "xtick.labelsize": 17, "ytick.labelsize": 17, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(22, 7.6))
grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.32)
for k, (title, h, acc) in enumerate(runs):
    ax = fig.add_subplot(grid[0, k])
    ep = np.arange(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_loss"], color=INK, lw=3.5, label="training")
    ax.plot(ep, h["val_loss"], color=ORANGE, lw=3.5, ls="--", label="validation")
    ax.set_ylim(0, 1.2)
    ax.set_title(f"{title}\ntest {acc:.1f}%", fontsize=21)
    ax.set_xlabel("Epoch")
    if k == 0:
        ax.set_ylabel("Cross-entropy loss")
        ax.legend(frameon=False, fontsize=18)
    else:
        ax.set_yticklabels([])
ax = fig.add_subplot(grid[0, 3])
ax.imshow(cm, cmap="Blues")
names = ["earthquake", "explosion", "noise", "surface"]
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=21, color="white" if cm[i, j] > 100 else INK)
ax.set_xticks(range(4), names, rotation=35, ha="right")
ax.set_yticks(range(4), names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Observed")
ax.set_title("1 hidden layer,\n800 test events", fontsize=21)
for s in ax.spines.values():
    s.set_visible(False)
fig.text(0.5, -0.1, "Pacific Northwest seismic sources, 61 waveform features (Zenodo record 14025693, real data). Notebook 4.1, seed 42.",
         ha="center", fontsize=17, color=MUTED)
fig.savefig(FIGS / "cls_evaluation.png", dpi=150, bbox_inches="tight")

recall = cm.diagonal() / cm.sum(axis=1)
numbers = {"confusion_matrix": cm.tolist(), "recall": dict(zip(names, recall.round(3).tolist())),
           "test_accuracy": {t: float(a) for t, _, a in runs},
           "val_loss_min": {t: [float(min(h["val_loss"])), int(np.argmin(h["val_loss"])) + 1] for t, h, _ in runs},
           "standard_error_points_800": float(100 * np.sqrt(0.871 * 0.129 / 800))}
(FIGS / "numbers_classification.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
