"""Slide figure for lecture 17: ensemble vote spread as a first uncertainty.

Reproduces notebook 3.9's agreement-vs-error bar chart — 200 bagged trees on
the PNW four-class test set — at 20pt. Error rates per agreement bin are
asserted against the executed notebook (~40% at <50% agreement, ~3% at >90%).

Regenerate: pixi run python book/slides/2026/figscripts/vote_agreement.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.ensemble import BaggingClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 17,
})

TAG = ("Pacific Northwest seismic events — 1000 held-out events, "
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2026, stratify=y)

oob_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(), n_estimators=200,
    oob_score=True, random_state=42).fit(X_train, y_train)

tree_votes = np.stack([tree.predict(X_test.to_numpy())
                       for tree in oob_clf.estimators_])
n_classes = len(oob_clf.classes_)
vote_frac = np.stack([(tree_votes == k).mean(axis=0)
                      for k in range(n_classes)], axis=1)
agreement = vote_frac.max(axis=1)
errors = oob_clf.predict(X_test) != y_test.to_numpy()

edges = [0.25, 0.5, 0.7, 0.9, 1.001]
bin_labels = ["25–50%\n(the forest splits)", "50–70%", "70–90%",
              "90–100%\n(the forest agrees)"]
bin_idx = np.digitize(agreement, edges) - 1
error_rate = [errors[bin_idx == b].mean() for b in range(4)]
counts = [int((bin_idx == b).sum()) for b in range(4)]
assert 0.35 < error_rate[0] < 0.45 and error_rate[3] < 0.05, error_rate
print("error rate per bin:", [f"{e:.3f}" for e in error_rate],
      "counts:", counts)

fig, ax = plt.subplots(figsize=(12.5, 7.6))
colors = ["#b3402a", "#c8763a", "#c9a83c", "#116b66"]
ax.bar(bin_labels, error_rate, color=colors)
for i, (e, n) in enumerate(zip(error_rate, counts)):
    ax.text(i, e + 0.008, f"{e:.0%} wrong\n(n = {n})", ha="center",
            fontsize=19, fontweight="bold")
ax.set_ylim(0, 0.50)
ax.set_xlabel("Vote agreement — fraction of 200 trees voting the winning class")
ax.set_ylabel("Misclassification rate, held-out events")
ax.set_title("The ensemble ranks its own predictions by trustworthiness",
             loc="left")
ax.grid(alpha=0.3, axis="y")
fig.suptitle(TAG, x=0.01, ha="left", fontsize=18, fontweight="normal",
             color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = (Path(__file__).resolve().parent.parent / "figs"
       / "3.9_ensemble_learning" / "vote_agreement_slide.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
