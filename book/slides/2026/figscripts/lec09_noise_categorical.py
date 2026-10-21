"""Slide figure (lecture 9): noise in labels, random and systematic.

Synthetic, fixed seed. 300 events per class for three source classes,
earthquake, explosion and noise, described by two features of the kind used to
tell them apart (both synthetic, drawn from Gaussian clusters that overlap):
  x  log10 of the P/S amplitude ratio (explosions make relatively more P)
  y  log10 of the ratio of 5-10 Hz to 1-5 Hz energy (noise windows are poorer
     in high frequencies)
The true labels are then corrupted three ways, following the taxonomy of
Frenay and Verleysen (2014):
  random 10 %, random 30 %  each label flips, with that probability, to one of
                            the two other classes chosen uniformly (noise
                            completely at random)
  class-dependent           40 % of explosions are labeled earthquakes and
                            15 % of noise windows are labeled earthquakes,
                            the rest kept (noise at random, depending on the
                            class: a quarry blast entered as an earthquake)
Top: the events colored by the label a user would receive. Bottom: confusion
matrix of observed label (columns) against true label (rows), in percent of
the true class. A logistic-regression classifier is trained on each label set
(first half of the events) and scored against the TRUE labels of the other
half; the score is printed.

Regenerate: pixi run python book/slides/2026/figscripts/lec09_noise_categorical.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED = "#26241f", "#6e675c"
CLASSES = ["earthquake", "explosion", "noise"]
CCOL = ["#2a78d6", "#eb6834", "#1baf7a"]          # dataviz reference slots 1-3
MARK = ["o", "s", "^"]
SEED, NPER = 2026, 300
CENTERS = np.array([[-0.25, 0.05], [0.35, 0.15], [0.05, -0.45]])
SPREAD = np.array([[0.20, 0.17], [0.20, 0.17], [0.20, 0.17]])
P_SYS = {(1, 0): 0.40, (2, 0): 0.15}               # (true, observed): probability

rng = np.random.default_rng(SEED)
X = np.vstack([c + s * rng.standard_normal((NPER, 2)) for c, s in zip(CENTERS, SPREAD)])
y_true = np.repeat(np.arange(3), NPER)
perm = rng.permutation(len(y_true))
X, y_true = X[perm], y_true[perm]


def random_flip(y, p):
    y = y.copy()
    flip = rng.random(y.size) < p
    y[flip] = (y[flip] + rng.integers(1, 3, flip.sum())) % 3   # one of the two other classes, uniformly
    return y


def class_flip(y):
    y = y.copy()
    u = rng.random(y.size)
    for (t, o), p in P_SYS.items():
        y[(y_true == t) & (u < p)] = o
    return y


SETS = [("true labels", y_true), ("random 10 %", random_flip(y_true, 0.10)),
        ("random 30 %", random_flip(y_true, 0.30)), ("class-dependent", class_flip(y_true))]

half = len(y_true) // 2
results = []
for name, y_obs in SETS:
    cm = confusion_matrix(y_true, y_obs, labels=[0, 1, 2])
    clf = LogisticRegression(max_iter=1000).fit(X[:half], y_obs[:half])
    pred = clf.predict(X[half:])
    results.append(dict(name=name, y=y_obs, cm=cm, agree=float(np.mean(y_obs == y_true)),
                        per_class=cm.diagonal() / cm.sum(axis=1),
                        acc_true=float(np.mean(pred == y_true[half:])),
                        acc_obs=float(np.mean(pred == y_obs[half:])),
                        recall=confusion_matrix(y_true[half:], pred, labels=[0, 1, 2], normalize="true").diagonal()))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16, 9.0))
gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.42, wspace=0.12)
for j, r in enumerate(results):
    ax = fig.add_subplot(gs[0, j])
    for k in range(3):
        m = r["y"] == k
        ax.scatter(X[m, 0], X[m, 1], s=22, color=CCOL[k], marker=MARK[k], lw=0, alpha=0.85,
                   label=CLASSES[k])
    ax.set_xlim(-0.95, 1.05)
    ax.set_ylim(-1.15, 0.75)
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_title(r["name"], loc="left", fontsize=17)
    ax.text(0.03, 0.03, f"{100 * r['agree']:.0f} % agree with truth", transform=ax.transAxes, fontsize=14,
            color=INK)
    if j == 0:
        ax.set_ylabel("log₁₀ (5–10 Hz / 1–5 Hz)")
        fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=16, loc="upper right",
                   bbox_to_anchor=(0.9, 0.985), ncol=3, handletextpad=0.1, markerscale=2.2)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("log₁₀ P/S amplitude")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    axc = fig.add_subplot(gs[1, j])
    pct = 100 * r["cm"] / r["cm"].sum(axis=1, keepdims=True)
    axc.imshow(pct, cmap="Greys", vmin=0, vmax=140)
    for a in range(3):
        for b in range(3):
            axc.text(b, a, f"{pct[a, b]:.0f}", ha="center", va="center", fontsize=17,
                     color="white" if pct[a, b] > 70 else INK, fontweight="bold" if a == b else "normal")
    axc.set_xticks(range(3))
    axc.set_xticklabels(["eq", "exp", "noise"], fontsize=15)
    axc.set_yticks(range(3))
    axc.set_yticklabels(["eq", "exp", "noise"] if j == 0 else [], fontsize=15)
    axc.set_xlabel("observed label", fontsize=15)
    if j == 0:
        axc.set_ylabel("true label", fontsize=15)
    axc.text(0.5, -0.29, f"classifier accuracy {100 * r['acc_true']:.0f} %", transform=axc.transAxes,
             ha="center", va="top", fontsize=15, color=INK, fontweight="bold")

out = Path(__file__).resolve().parent.parent / "figs" / "lec09_noise" / "noise_categorical.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"seed {SEED}; {NPER} events per class; centers {CENTERS.tolist()}; spreads {SPREAD.tolist()}; "
      f"class-dependent flips {P_SYS} (true, observed)")
print("label set | agreement with truth | per-class agreement eq/exp/noise | classifier trained on these labels: "
      "accuracy vs true labels (test half) | vs these labels | recall eq/exp/noise vs truth")
for r in results:
    print(f"{r['name']:<16} | {100 * r['agree']:.1f} % | " + "/".join(f"{100 * v:.1f}" for v in r["per_class"])
          + f" | {100 * r['acc_true']:.1f} % | {100 * r['acc_obs']:.1f} % | "
          + "/".join(f"{100 * v:.1f}" for v in r["recall"]))
for r in results:
    print(f"confusion counts, {r['name']} (rows true eq/exp/noise, columns observed): {r['cm'].tolist()}")
