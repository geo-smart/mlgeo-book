"""Slide figures: why and how to reduce the seismic features (notebook 3.3, section 6).

Data: 4000 Pacific Northwest seismic events (1000 per analyst label), 61 waveform
features, Zenodo record 14025693 (Kharita 2024; Kharita et al. 2026, Seismica),
read from the same checksummed cache as the notebook; column Gamma (missing
values) dropped. Settings as in the notebook: StandardScaler, PCA() with 15
components for 80% of the variance, t-SNE on the 15 PCs of 2000 events drawn
with default_rng(42), init="pca", random_state=42, perplexities 5 and 50.
  seis_feature_corr_slide.png  correlation matrix of the 58 non-constant
                               features, histogram of |r|, one redundant pair.
  pca_2d_slide.png             FMaxFFT vs FCentroid standardized, principal axes
                               as arrows of one standard deviation; PC scores.
  seis_pca_slide.png           cumulative explained variance; PC1 vs PC2
                               colored by analyst label; largest loadings.
  seis_tsne_slide.png          PC1-PC2 and t-SNE maps of the same 2000 events,
                               with the share of 10 nearest neighbors kept.

Regenerate: pixi run python book/slides/2026/figscripts/seismic_dimred.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
MUTED = "#6e675c"
SEED, VAR_TARGET, N_TSNE, PERPLEXITIES = 42, 0.80, 2000, (5, 50)       # as in notebook 3.3
SOURCES = ["earthquake", "explosion", "surface event", "noise"]
SRC_COLORS = dict(zip(SOURCES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))   # notebook colors C0-C3
TAG = "4000 Pacific Northwest seismic events, 61 waveform features (Kharita et al. 2026, Seismica; Zenodo 14025693)"
OUT = Path(__file__).resolve().parent.parent / "figs" / "3.3_clustering"
OUT.mkdir(parents=True, exist_ok=True)

ZENODO_FILES = "https://zenodo.org/api/records/14025693/files"
SEISMIC_FILES = {"earthquake": ("1000_earthquakes_physical_features.csv", "md5:28129c8dd1b3e14f655d489577b841b5"),
                 "explosion": ("1000_explosion_physical_features.csv", "md5:af1342d32e163e961e043364136359b0"),
                 "noise": ("1000_noise_physical_features.csv", "md5:16cdb992fed6cf6273d5624f5df905da"),
                 "surface event": ("1000_surface_physical_features.csv", "md5:9a2c2643030cf058704d68e130654e9d")}
frames = [pd.read_csv(pooch.retrieve(f"{ZENODO_FILES}/{fname}/content", known_hash=md5, fname=fname,
                                     path=pooch.os_cache("mlgeo"), progressbar=False), index_col=0)
          for fname, md5 in SEISMIC_FILES.values()]
seismic = pd.concat(frames, ignore_index=True).dropna(axis=1)
X_seis = seismic.drop(columns=["source", "serial_no"])
y_seis = seismic["source"].to_numpy()

Z_seis = StandardScaler().fit_transform(X_seis)
pca_seis = PCA().fit(Z_seis)
cum_var = np.cumsum(pca_seis.explained_variance_ratio_)
n_pcs = int(np.searchsorted(cum_var, VAR_TARGET) + 1)
X_pca = pca_seis.transform(Z_seis)[:, :n_pcs]
print(f"{len(X_seis)} events, {X_seis.shape[1]} features; {n_pcs} PCs explain {cum_var[n_pcs - 1]:.1%}; "
      f"PC1 {pca_seis.explained_variance_ratio_[0]:.1%}, PC2 {pca_seis.explained_variance_ratio_[1]:.1%}, "
      f"together {cum_var[1]:.1%}")


def save(fig, name):
    out = OUT / name
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")


# ================================================================ (1) why reduce
varying = X_seis.columns[X_seis.std() > 0]
corr = X_seis[varying].corr()
iu = np.triu_indices(len(corr), k=1)
r = corr.to_numpy()[iu]
print(f"{len(varying)} features vary, {len(r)} pairs; |r| > 0.9: {(np.abs(r) > 0.9).sum()} pairs; "
      f"|r| > 0.5: {(np.abs(r) > 0.5).mean():.0%}; median |r| {np.median(np.abs(r)):.2f}")
print("r(Average_Envelope_Amplitude, Envelope_Area) =",
      f"{corr.loc['Average_Envelope_Amplitude', 'Envelope_Area']:.4f}")
groups = [("signal and\nenvelope\nstatistics", "RappMaxMean", "RMSDecPhaseLine"), ("spectrum", "MeanFFT", "Gamma2"),
          ("spectrogram", "KurtoMaxDFT", "DistQ3Q1DFT"), ("envelope", "Peak_Envelope_Amplitude", "Envelope_Rise_Time"),
          ("frequency\nbands", "E_0.1_1", "Kurto_10_20")]
cols = list(corr.columns)

fig, ax = plt.subplots(1, 3, figsize=(26, 8.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
im = ax[0].imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_xticks([]); ax[0].set_yticks([])
for name, first, last in groups:
    i0, i1 = cols.index(first), cols.index(last)
    ax[0].add_patch(plt.Rectangle((i0 - 0.5, i0 - 0.5), i1 - i0 + 1, i1 - i0 + 1, fill=False, ec="black", lw=1.8))
    ax[0].text(-1.5, (i0 + i1) / 2, name.replace("\n", " ") if i1 - i0 > 8 else name, ha="right", va="center",
               fontsize=15)
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02, label="correlation r")
ax[0].set_title(f"{len(varying)} features, table order", loc="left")
ax[1].hist(np.abs(r), bins=np.arange(0, 1.0001, 0.05), color="#8c8576", edgecolor="white")
ax[1].axvline(0.9, color="#b3402a", ls="--", lw=2)
ax[1].text(0.88, 0.95, f"{(np.abs(r) > 0.9).sum()} pairs\nabove 0.9", transform=ax[1].get_xaxis_transform(),
           ha="right", va="top", fontsize=18, color="#b3402a")
ax[1].set_xlabel("|r| between two features"); ax[1].set_ylabel("number of pairs")
ax[1].set_title(f"{len(r)} pairs of features", loc="left")
ax[1].grid(True, alpha=0.4)
ax[2].scatter(X_seis["Average_Envelope_Amplitude"], X_seis["Envelope_Area"], s=8, color="#3b3b3b", alpha=0.5)
ax[2].set_xlabel("Average_Envelope_Amplitude"); ax[2].set_ylabel("Envelope_Area")
ax[2].set_title(f"r = {corr.loc['Average_Envelope_Amplitude', 'Envelope_Area']:.4f}", loc="left")
ax[2].grid(True, alpha=0.4)
fig.suptitle(TAG, x=0.01, ha="left", fontsize=20, fontweight="normal", color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.0)
save(fig, "seis_feature_corr_slide.png")

# ================================================================ (2) PCA in 2-D
FREQ_PAIR = ["FMaxFFT", "FCentroid"]
Z_freq = StandardScaler().fit_transform(X_seis[FREQ_PAIR])
pca_freq = PCA().fit(Z_freq)
sd = np.sqrt(pca_freq.explained_variance_)
print(f"FMaxFFT-FCentroid: r = {np.corrcoef(Z_freq.T)[0, 1]:.3f}; s.d. along PC1, PC2 {sd.round(3)}; "
      f"explained variance ratio {pca_freq.explained_variance_ratio_.round(3)}; "
      f"directions {pca_freq.components_.round(3).tolist()}")
fig, ax = plt.subplots(1, 2, figsize=(21, 8.2))
ax[0].scatter(Z_freq[:, 0], Z_freq[:, 1], s=8, color="#a39d92", alpha=0.5)
for k, (vec, s_k) in enumerate(zip(pca_freq.components_, sd)):
    color = ["#1f5fa8", "#b3402a"][k]
    tip = pca_freq.mean_ + 2 * vec * s_k
    ax[0].annotate("", tip, pca_freq.mean_, arrowprops=dict(arrowstyle="-|>", color=color, lw=4, mutation_scale=30))
    ax[0].text(*(tip + 0.25 * vec), f"PC{k + 1}: {pca_freq.explained_variance_ratio_[k]:.0%}\n(s.d. {s_k:.2f})",
               color=color, fontsize=20, fontweight="bold", ha="left" if vec[0] > 0 else "right", va="center")
ax[0].set_xlabel("FMaxFFT, dominant frequency (standardized)")
ax[0].set_ylabel("FCentroid (standardized)")
ax[0].set_title(f"Two frequency features, r = {np.corrcoef(Z_freq.T)[0, 1]:.2f}", loc="left")
scores = pca_freq.transform(Z_freq)
ax[1].scatter(scores[:, 0], scores[:, 1], s=8, color="#a39d92", alpha=0.5)
ax[1].set_xlabel("score on PC1"); ax[1].set_ylabel("score on PC2")
ax[1].set_title("Same events, rotated onto the PCs", loc="left")
lim = (-3.2, 6.2)
for a in ax:
    a.set_xlim(lim); a.set_ylim(-3.2, 4.2)
    a.set_aspect("equal")
    a.grid(True, alpha=0.4)
fig.suptitle(TAG.replace(", 61 waveform features", "") + "; arrows: 2 standard deviations along each component",
             x=0.01, ha="left", fontsize=19, fontweight="normal", color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5)
save(fig, "pca_2d_slide.png")

# ================================================================ (3) PCA on the 61 features
loadings = pd.DataFrame(pca_seis.components_[:2].T, index=X_seis.columns, columns=["PC1", "PC2"])
top = {pc: loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index).head(6) for pc in loadings}
for pc in top:
    print(f"{pc} largest loadings: {top[pc].round(3).to_dict()}")
med = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"]).groupby(y_seis).median().reindex(SOURCES).round(2)
print("median scores by label:\n" + med.to_string())

fig = plt.figure(figsize=(27, 8.8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.05, 0.95], hspace=0.55, wspace=0.42)
a0, a1 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
a2, a3 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
ks = np.arange(1, len(cum_var) + 1)
a0.plot(ks, cum_var, "o-", color="black", ms=5, lw=2)
a0.axhline(VAR_TARGET, color="#b3402a", ls="--", lw=2)
a0.axvline(n_pcs, color="#1f5fa8", ls=":", lw=2.5)
a0.text(n_pcs + 1.5, 0.35, f"{n_pcs} components\n{cum_var[n_pcs - 1]:.1%} of the variance", color="#1f5fa8",
        fontsize=19)
a0.set_xlabel("number of components"); a0.set_ylabel("cumulative explained variance")
a0.set_ylim(0, 1.02)
a0.set_title("Variance kept", loc="left")
a0.grid(True, alpha=0.4)
for s in SOURCES:
    m = y_seis == s
    a1.scatter(X_pca[m, 0], X_pca[m, 1], s=7, alpha=0.45, color=SRC_COLORS[s], label=s)
a1.set_xlabel(f"PC1 ({pca_seis.explained_variance_ratio_[0]:.1%}): sustained vs peaked")
a1.set_ylabel(f"PC2 ({pca_seis.explained_variance_ratio_[1]:.1%}): high vs low frequency")
a1.set_title("Events on PC1 and PC2", loc="left")
a1.legend(fontsize=16, markerscale=4, loc="upper left", handletextpad=0.2, borderpad=0.3)
a1.grid(True, alpha=0.4)
for a, pc in [(a2, "PC1"), (a3, "PC2")]:
    t = top[pc][::-1]
    a.barh(range(len(t)), t.values, color=["#1f5fa8" if v > 0 else "#b3402a" for v in t.values])
    a.set_yticks(range(len(t)), t.index, fontsize=14)
    a.axvline(0, color="black", lw=1)
    a.set_xlim(-0.3, 0.3)
    a.tick_params(axis="x", labelsize=14)
    a.set_title(f"Largest loadings, {pc}", loc="left", fontsize=20)
    a.grid(True, axis="x", alpha=0.4)
fig.suptitle(TAG + "; labels color the points only", x=0.01, ha="left", fontsize=19, fontweight="normal",
             color=MUTED)
fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.1)
save(fig, "seis_pca_slide.png")

# ================================================================ (4) t-SNE
idx_tsne = np.sort(np.random.default_rng(SEED).choice(len(X_pca), N_TSNE, replace=False))
maps_tsne = {p: TSNE(n_components=2, perplexity=p, init="pca", random_state=SEED).fit_transform(X_pca[idx_tsne])
             for p in PERPLEXITIES}


def neighbor_overlap(X_full, X_2d, k=10):
    nn_full = NearestNeighbors(n_neighbors=k + 1).fit(X_full).kneighbors(X_full, return_distance=False)[:, 1:]
    nn_2d = NearestNeighbors(n_neighbors=k + 1).fit(X_2d).kneighbors(X_2d, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nn_full, nn_2d)]))


views = {"PC1 and PC2": X_pca[idx_tsne, :2]}
views.update({f"t-SNE, perplexity {p}": m for p, m in maps_tsne.items()})
overlap = {name: neighbor_overlap(X_pca[idx_tsne], m) for name, m in views.items()}
print("share of 10 nearest neighbors (15 PCs) kept in 2-D:", {k: round(v, 3) for k, v in overlap.items()})
y_sub = y_seis[idx_tsne]
fig, ax = plt.subplots(1, 3, figsize=(26, 8.8))
for a, (name, m) in zip(ax, views.items()):
    for s in SOURCES:
        sel = y_sub == s
        a.scatter(m[sel, 0], m[sel, 1], s=9, alpha=0.6, color=SRC_COLORS[s], label=s)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(name, loc="left")
    a.text(0.02, 0.02, f"{overlap[name]:.0%} of 10 nearest\nneighbors kept", transform=a.transAxes, fontsize=19,
           va="bottom", bbox=dict(fc="white", ec=MUTED, alpha=0.9))
    a.set_aspect("equal", adjustable="datalim")
ax[0].legend(fontsize=16, markerscale=4, loc="upper left", handletextpad=0.2)
fig.suptitle(f"{N_TSNE} of the 4000 Pacific Northwest seismic events, 15 principal components; colors: analyst labels "
             "(viewing only); t-SNE axes have no units", x=0.01, ha="left", fontsize=19, fontweight="normal",
             color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.5)
save(fig, "seis_tsne_slide.png")
