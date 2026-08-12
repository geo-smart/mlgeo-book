"""Slide figures for lecture 15: the PNW four-class problem at lecture scale.

Reproduces notebook 3.5's random-forest confusion matrix and one-vs-rest ROC
curves on the canonical split (random_state=2026, stratified), re-plotted at
20pt so the room can read them. The confusion counts are asserted against the
executed notebook's crosstab, so slide and book always agree.

Regenerate: pixi run python book/slides/2026/figscripts/pnw_multiclass.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler, label_binarize  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

TAG = ("Pacific Northwest seismic events — 1000 per class, "
       "Zenodo record 14025693 (real data)")
SEISMIC_FILES = {
    "1000_earthquakes_physical_features.csv": "md5:28129c8dd1b3e14f655d489577b841b5",
    "1000_explosion_physical_features.csv": "md5:af1342d32e163e961e043364136359b0",
    "1000_noise_physical_features.csv": "md5:16cdb992fed6cf6273d5624f5df905da",
    "1000_surface_physical_features.csv": "md5:9a2c2643030cf058704d68e130654e9d",
}

frames = []
for fname, checksum in SEISMIC_FILES.items():
    path = pooch.retrieve(
        url=f"https://zenodo.org/api/records/14025693/files/{fname}/content",
        known_hash=checksum, fname=fname, path=pooch.os_cache("mlgeo"))
    frames.append(pd.read_csv(path, index_col=0))
seismic = pd.concat(frames, ignore_index=True).dropna(axis=1)

X = seismic.drop(columns=["source", "serial_no"])
y = seismic["source"]
# The canonical leaderboard split from notebook 3.5 — do not change.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2026, stratify=y)

classes = ["earthquake", "explosion", "noise", "surface event"]
FIGS = Path(__file__).resolve().parent.parent / "figs" / "3.5_multiclass_classification"
FIGS.mkdir(parents=True, exist_ok=True)

# --- Figure 1: random-forest confusion matrix -------------------------------
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
rf_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, rf_pred, labels=classes)
expected = np.array([[217, 13, 9, 11], [32, 199, 4, 15],
                     [7, 8, 234, 1], [5, 7, 4, 234]])
assert (cm == expected).all(), f"confusion counts drifted from notebook:\n{cm}"

fig, ax = plt.subplots(figsize=(10.5, 8.6))
im = ax.imshow(cm, cmap="viridis")
ax.set_xticks(range(4), classes, rotation=20, ha="right")
ax.set_yticks(range(4), classes)
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=22,
                fontweight="bold",
                color="black" if cm[i, j] > cm.max() / 2 else "white")
ax.set_xlabel("Predicted source type")
ax.set_ylabel("True source type")
ax.set_title("Random forest — 1000 held-out events", loc="left")
fig.colorbar(im, ax=ax, shrink=0.85, label="events")
fig.suptitle(TAG, x=0.01, ha="left", fontsize=18, fontweight="normal",
             color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = FIGS / "confusion_rf_slide.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")

# --- Figure 2: one-vs-rest ROC curves ---------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
y_train_bin = label_binarize(y_train, classes=classes)
y_test_bin = label_binarize(y_test, classes=classes)
ovr = OneVsRestClassifier(SVC(kernel="linear"))
y_score = ovr.fit(X_train_s, y_train_bin).decision_function(X_test_s)

fig, ax = plt.subplots(figsize=(9.6, 8.6))
ax.plot([0, 1], [0, 1], "k--", lw=2, label="chance")
for i, name in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    ax.plot(fpr, tpr, lw=3, label=f"{name} vs rest  (AUC = {auc(fpr, tpr):.2f})")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.03)
ax.grid(alpha=0.3)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("One class against the other three, four times", loc="left")
ax.legend(loc="lower right", fontsize=17)
fig.suptitle(TAG, x=0.01, ha="left", fontsize=18, fontweight="normal",
             color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = FIGS / "ovr_roc_slide.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
