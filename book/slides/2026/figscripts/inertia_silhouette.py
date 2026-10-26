"""Slide figures: inertia and silhouette, drawn on Old Faithful (notebook 3.3, sections 2 and 5).

Data: the 271 consecutive eruption-duration pairs of the Old Faithful geyser,
Yellowstone (faithful.csv, the same cached file the notebook downloads),
standardized; scikit-learn KMeans, k = 3, k-means++, n_init=20,
random_state=42, as km_faith in the notebook.
  inertia_terms_slide.png       every pair joined to its cluster center; the
                                271 squared distances whose sum is the inertia J.
  silhouette_anatomy_slide.png  (i) pair 173 (SIL_SAMPLE in the notebook) with
                                the lines whose mean lengths are a and b;
                                (ii) every pair colored by its silhouette s(i);
                                (iii) silhouettes sorted within clusters, mean marked.

Regenerate: pixi run python book/slides/2026/figscripts/inertia_silhouette.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_samples  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
MUTED = "#6e675c"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
SEED, N_RESTARTS, SIL_SAMPLE = 42, 20, 173          # as in notebook 3.3
OUT = Path(__file__).resolve().parent.parent / "figs" / "3.3_clustering"
OUT.mkdir(parents=True, exist_ok=True)

path = pooch.retrieve("https://raw.githubusercontent.com/UW-MLGEO/MLGeo-dataset/main/data/faithful.csv",
                      known_hash=None, fname="faithful.csv", path=pooch.os_cache("mlgeo"), progressbar=False)
faithful = pd.read_csv(path)
X = StandardScaler().fit_transform(faithful[["current", "next"]])
km = KMeans(n_clusters=3, init="k-means++", n_init=N_RESTARTS, random_state=SEED).fit(X)
labels, centers = km.labels_, km.cluster_centers_
XLAB, YLAB = "eruption duration (standardized)", "next duration (standardized)"

# ================================================================ inertia
d2 = ((X - centers[labels]) ** 2).sum(axis=1)       # one term of J per pair
order = np.argsort(d2)[::-1]
green = np.where(labels == labels[np.argmax(X[:, 0] + X[:, 1])])[0]          # the long-then-long cluster
shown = [int(order[0]), int(green[np.argmax(d2[green])]), int(np.argmin(np.abs(d2 - np.median(d2))))]
# the largest term, the largest of the long-then-long cluster, and a typical (median) one
print(f"k = 3: J = {d2.sum():.1f} (KMeans inertia_ {km.inertia_:.1f}) over {len(d2)} pairs; "
      f"mean term {d2.mean():.2f} (root {np.sqrt(d2.mean()):.2f}); largest terms {np.round(d2[order[:4]], 2).tolist()}")
print("labeled pairs (current, next min; term):",
      [(float(faithful.loc[i, "current"]), float(faithful.loc[i, "next"]), round(float(d2[i]), 2)) for i in shown])

fig, (a0, a1) = plt.subplots(1, 2, figsize=(21, 8.0), gridspec_kw={"width_ratios": [1, 1.25]})
for i in range(len(X)):
    a0.plot(*zip(X[i], centers[labels[i]]), color="#c9c3b8", lw=0.8, zorder=1)
for j in range(3):
    a0.scatter(X[labels == j, 0], X[labels == j, 1], s=30, color=COLORS[j], zorder=2)
for i, offset in zip(shown, [(-12, -30), (-12, -30), (-75, 12)]):
    a0.plot(*zip(X[i], centers[labels[i]]), color="black", lw=2.5, zorder=3)
    a0.annotate(f"{d2[i]:.2f}", X[i], xytext=offset, textcoords="offset points", fontsize=19, fontweight="bold")
a0.scatter(centers[:, 0], centers[:, 1], marker="X", s=420, color="black", zorder=4)
a0.set_xlabel(XLAB); a0.set_ylabel(YLAB)
a0.set_title("Each line: one squared distance", loc="left")
a0.grid(True, alpha=0.4)
a1.bar(np.arange(len(d2)), d2[order], width=1.0, color=[COLORS[labels[i]] for i in order])
for i in shown:
    r = int(np.where(order == i)[0][0])
    a1.annotate(f"{d2[i]:.2f}", (r, d2[i]), xytext=(18, 6), textcoords="offset points", fontsize=19,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color="black", lw=1))
a1.set_xlabel("eruption pairs, sorted by their term")
a1.set_ylabel("squared distance to center")
a1.set_xlim(-3, len(d2) + 2)
a1.set_title(f"J = sum of {len(d2)} terms = {d2.sum():.1f}", loc="left")
a1.text(0.97, 0.75, f"mean term {d2.mean():.2f}\nunit: squared\nstandardized units", transform=a1.transAxes,
        ha="right", va="top", fontsize=19, color=MUTED)
a1.grid(True, axis="y", alpha=0.4)
fig.suptitle("Old Faithful geyser, Yellowstone: 271 consecutive eruption pairs, k-means with k = 3 "
             "(colors: clusters; crosses: centers)", x=0.01, ha="left", fontsize=20, fontweight="normal", color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=2.5)
out = OUT / "inertia_terms_slide.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")

# ================================================================ silhouette
s = silhouette_samples(X, labels)
i0 = SIL_SAMPLE
d = np.linalg.norm(X - X[i0], axis=1)
own = labels == labels[i0]
near = min((j for j in range(3) if j != labels[i0]), key=lambda j: d[labels == j].mean())
a_i, b_i = d[own].sum() / (own.sum() - 1), d[labels == near].mean()
print(f"pair {i0} ({faithful.loc[i0, 'current']} then {faithful.loc[i0, 'next']} min), cluster {labels[i0]}, "
      f"nearest other cluster {near}: a = {a_i:.3f}, b = {b_i:.3f}, s = {(b_i - a_i) / max(a_i, b_i):.3f}; "
      f"silhouette_samples {s[i0]:.3f}")
print(f"mean silhouette {s.mean():.3f}; per cluster {[round(float(s[labels == j].mean()), 3) for j in range(3)]}; "
      f"lowest five {np.round(np.sort(s)[:5], 3).tolist()}")

fig, ax = plt.subplots(1, 3, figsize=(25, 8.4), gridspec_kw={"width_ratios": [1, 1.12, 0.8]})
for i in np.where(own)[0]:
    ax[0].plot(*zip(X[i0], X[i]), color=COLORS[labels[i0]], lw=0.7, alpha=0.7)
for i in np.where(labels == near)[0]:
    ax[0].plot(*zip(X[i0], X[i]), color=COLORS[near], lw=0.7, alpha=0.7, ls="--")
for j in range(3):
    ax[0].scatter(X[labels == j, 0], X[labels == j, 1], s=22, color=COLORS[j], zorder=2)
ax[0].scatter(*X[i0], marker="*", s=900, color="black", zorder=4)
ax[0].text(0.03, 0.26, f"a = {a_i:.2f} (solid)\nb = {b_i:.2f} (dashed)\ns = (b − a) / max(a, b)\n   = {s[i0]:.2f}",
           transform=ax[0].transAxes, fontsize=17, va="bottom", bbox=dict(fc="white", ec=MUTED, alpha=0.95))
ax[0].set_title(f"(i) One pair: a and b", loc="left")
sc = ax[1].scatter(X[:, 0], X[:, 1], c=s, cmap="viridis", vmin=-0.1, vmax=1, s=40)
ax[1].scatter(*X[i0], marker="*", s=700, color="red", edgecolor="white", zorder=4)
cb = fig.colorbar(sc, ax=ax[1], fraction=0.05, pad=0.02)
cb.set_label("silhouette s(i)")
ax[1].set_title("(ii) Low on the boundaries", loc="left")
for a in ax[:2]:
    a.set_xlabel(XLAB); a.set_ylabel(YLAB)
    a.grid(True, alpha=0.4)
y0 = 0
for j in range(3):
    sj = np.sort(s[labels == j])
    ax[2].fill_betweenx(np.arange(y0, y0 + len(sj)), 0, sj, color=COLORS[j], alpha=0.85)
    ax[2].text(-0.04, y0 + len(sj) / 2, str(j), ha="right", va="center", fontsize=19)
    y0 += len(sj) + 8
ax[2].axvline(s.mean(), color="black", ls="--", lw=2)
ax[2].text(s.mean() + 0.02, y0 + 2, f"mean {s.mean():.3f}", ha="left", va="bottom", fontsize=19)
ax[2].set_xlim(-0.15, 1); ax[2].set_ylim(-4, y0 + 30)
ax[2].set_yticks([])
ax[2].set_xlabel("silhouette s(i)"); ax[2].set_ylabel("pairs, by cluster")
ax[2].set_title("(iii) Sorted by cluster", loc="left")
ax[2].grid(True, axis="x", alpha=0.4)
fig.suptitle("Old Faithful geyser, Yellowstone: 271 eruption pairs, k-means with k = 3; pair 173 "
             f"({faithful.loc[i0, 'current']:.1f} then {faithful.loc[i0, 'next']:.1f} min) starred",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=2.0)
out = OUT / "silhouette_anatomy_slide.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
