"""Slide figures: the distance decides the clusters (notebook 3.3, section 4).

Two synthetic examples with known groups, clustered by agglomerative clustering
(average linkage) on a precomputed distance matrix, scored with the V-measure:
  epicenters_dateline_slide.png   180 synthetic epicenters in three groups along
                                  the Aleutian arc, one on the 180° meridian;
                                  Euclidean distance on (lon, lat) vs great-circle
                                  (haversine) distance. Land: Natural Earth 1:50m.
  waveform_correlation_slide.png  150 synthetic traces, impulsive vs emergent
                                  sources, peak amplitudes log-uniform 1 to 100;
                                  Euclidean vs correlation distance.
The generators, seeds and settings are copied verbatim from notebook 3.3; the
script checks that the notebook still holds the same code and stops if not.

Regenerate: pixi run python book/slides/2026/figscripts/distance_metrics.py
"""
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.metrics import pairwise_distances, v_measure_score  # noqa: E402
from sklearn.metrics.pairwise import haversine_distances  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
MUTED = "#6e675c"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parents[2] / "Chapter3-MachineLearning" / "3.3_clustering.ipynb"
OUT = HERE.parent / "figs" / "3.3_clustering"
OUT.mkdir(parents=True, exist_ok=True)

# ---- settings and generators, verbatim from notebook 3.3 ----------------------
SEED = 42
R_EARTH_KM = 6371.0             # mean radius of the Earth (km), for great-circle distances
ARC_GROUPS = {"Near Islands": (52.8, 173.0),             # (latitude °N, longitude °E) of each group center
              "Rat and Delarof Islands": (51.5, 180.0),   # on the 180° meridian
              "Andreanof Islands": (51.9, -175.5)}
KM_PER_DEG = np.pi * R_EARTH_KM / 180                     # km per degree of latitude


def make_epicenters(n_per_group=60, sd_km=(40.0, 20.0), seed=SEED):
    """Epicenters around each center; sd_km = (along the arc, across it) in km; longitudes in -180 to 180."""
    rng = np.random.default_rng(seed)
    frames = []
    for g, (name, (lat0, lon0)) in enumerate(ARC_GROUPS.items()):
        east_km = rng.normal(0, sd_km[0], n_per_group)   # the arc runs roughly east-west here
        north_km = rng.normal(0, sd_km[1], n_per_group)
        lat = lat0 + north_km / KM_PER_DEG
        lon = lon0 + east_km / (KM_PER_DEG * np.cos(np.radians(lat0)))
        lon = (lon + 180) % 360 - 180                     # wrap to -180 to 180, as in a catalog
        frames.append(pd.DataFrame({"lon": lon, "lat": lat, "group": g, "segment": name}))
    return pd.concat(frames, ignore_index=True)


FS_WAVE, T_WAVE, ONSET = 50.0, 20.0, 2.0   # samples per second, window length (s), signal onset (s)
SOURCE_SHAPES = {"impulsive": dict(rise=0.05, decay=0.6, freq=8.0),   # short, sharp source
                 "emergent": dict(rise=4.0, decay=5.0, freq=2.5)}     # long source with a gradual onset


def make_waveforms(n_per_type=75, noise_sd=0.2, seed=SEED):
    """Traces aligned on their onset; returns times (s), traces (n, samples), source type (0, 1), peak amplitude."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T_WAVE, 1 / FS_WAVE)
    tau = np.clip(t - ONSET, 0, None)                              # time since the onset, 0 before it
    traces, kind, amp = [], [], []
    for g, p in enumerate(SOURCE_SHAPES.values()):
        for _ in range(n_per_type):
            rise, decay = p["rise"] * rng.uniform(0.8, 1.2), p["decay"] * rng.uniform(0.8, 1.2)
            env = (1 - np.exp(-tau / rise)) * np.exp(-tau / decay)   # envelope, zero before the onset
            a = 10 ** rng.uniform(0, 2)                                # peak amplitude, 1 to 100
            traces.append(a * env / env.max() * np.sin(2 * np.pi * p["freq"] * tau)
                          + rng.normal(0, noise_sd, len(t)))
            kind.append(g)
            amp.append(a)
    return t, np.array(traces), np.array(kind), np.array(amp)


# ---- the notebook must hold the same code --------------------------------------
nb_code = "\n".join("".join(c["source"]) for c in json.load(open(NOTEBOOK))["cells"] if c["cell_type"] == "code")
for obj in (make_epicenters, make_waveforms):
    assert inspect.getsource(obj) in nb_code, f"{obj.__name__} differs from notebook 3.3"
for line in ['"Rat and Delarof Islands": (51.5, 180.0)', '"Andreanof Islands": (51.9, -175.5)',
             'ARC_GROUPS = {"Near Islands": (52.8, 173.0)', "FS_WAVE, T_WAVE, ONSET = 50.0, 20.0, 2.0",
             'SOURCE_SHAPES = {"impulsive": dict(rise=0.05, decay=0.6, freq=8.0)',
             '"emergent": dict(rise=4.0, decay=5.0, freq=2.5)}', "SEED = 42", "R_EARTH_KM = 6371.0"]:
    assert line in nb_code, f"setting differs from notebook 3.3: {line}"
print(f"generators and settings identical to {NOTEBOOK.name}")


def avg_link(k):
    return AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")


# ================================================================ epicenters
epi = make_epicenters()
D_deg = pairwise_distances(epi[["lon", "lat"]])
labels_deg = avg_link(3).fit_predict(D_deg)
vm_deg = v_measure_score(epi["group"], labels_deg)
D_km = haversine_distances(np.radians(epi[["lat", "lon"]])) * R_EARTH_KM
labels_hav = avg_link(3).fit_predict(D_km)
vm_hav = v_measure_score(epi["group"], labels_hav)
i_e, i_w = epi["lon"].idxmax(), epi["lon"].idxmin()
print(epi.groupby("segment", sort=False)[["lon", "lat"]].agg(["min", "max"]).round(2))
print(f"Euclidean on (lon, lat): V-measure {vm_deg:.3f}")
print(pd.crosstab(labels_deg, epi["segment"], rownames=["cluster"], colnames=["arc segment"]))
print(f"great-circle: V-measure {vm_hav:.3f}")
print(f"closest pair across 180°: {D_km[i_e, i_w]:.0f} km great-circle, {D_deg[i_e, i_w]:.1f}° Euclidean")

land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip",
                                    known_hash=None, fname="ne_50m_land.zip",
                                    path=pooch.os_cache("mlgeo-figscripts"), progressbar=False))
lon360 = epi["lon"].to_numpy() % 360
fig, axes = plt.subplots(1, 2, figsize=(20, 5.9))
for ax, (title, labels) in zip(axes, [(f"Euclidean on (lon, lat): V = {vm_deg:.3f}", labels_deg),
                                      (f"Great-circle distance: V = {vm_hav:.3f}", labels_hav)]):
    ax.set_facecolor("#eef3f8")
    for shift in (0, 360):
        land.translate(xoff=shift).plot(ax=ax, color="#f4efe6", edgecolor=MUTED, lw=0.8)
    for j in np.unique(labels):
        ax.scatter(lon360[labels == j], epi["lat"][labels == j], s=45, color=COLORS[j], edgecolor="white",
                   lw=0.4, zorder=3, label=f"cluster {j} ({np.sum(labels == j)})")
    ax.axvline(180, color="black", ls=":", lw=1.5)
    ax.text(180.2, 50.55, "180°", fontsize=17, color="black", va="bottom")
    ax.set_xticks([170, 175, 180, 185], ["170°E", "175°E", "180°", "175°W"])
    ax.set_xlim(168, 188); ax.set_ylim(50.4, 54.2)
    ax.set_aspect(1 / np.cos(np.radians(52)))
    ax.set_xlabel(""); ax.set_ylabel("latitude (°N)")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=16, loc="lower left", framealpha=0.9)
fig.subplots_adjust(left=0.05, right=0.99, wspace=0.12)
fig.canvas.draw()                                    # axes positions are final only after drawing (fixed aspect)
top = axes[0].get_position().y1
fig.text(0.01, top + 0.16, "Synthetic epicenters, Aleutian arc: three groups of 60, one on the 180° meridian "
         "(average linkage, 3 clusters; land: Natural Earth)", ha="left", fontsize=19, color=MUTED)
out = OUT / "epicenters_dateline_slide.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")

# ================================================================ waveforms
t, waves, kind, amp = make_waveforms()
labels_euc = avg_link(2).fit_predict(pairwise_distances(waves))
vm_euc = v_measure_score(kind, labels_euc)
D_corr = pairwise_distances(waves, metric="correlation")
labels_corr = avg_link(2).fit_predict(D_corr)
vm_corr = v_measure_score(kind, labels_corr)
same = kind[:, None] == kind[None, :]
print(f"{len(waves)} traces; peak amplitude {amp.min():.2f} to {amp.max():.1f}")
for name, lab, vm in [("Euclidean", labels_euc, vm_euc), ("correlation", labels_corr, vm_corr)]:
    print(f"{name}: V-measure {vm:.3f}; traces per cluster {np.bincount(lab)}; median peak amplitude per cluster "
          f"{[round(float(np.median(amp[lab == j])), 1) for j in range(2)]}; "
          f"source types in each cluster {[np.bincount(kind[lab == j], minlength=2).tolist() for j in range(2)]}")
print(f"mean correlation distance: same type {D_corr[same & ~np.eye(len(waves), dtype=bool)].mean():.3f}, "
      f"different types {D_corr[~same].mean():.3f}")

names = list(SOURCE_SHAPES)
TRACE_COLORS = ["#6250d6", "#8c564b"]
fig = plt.figure(figsize=(24, 9.2))
gs = fig.add_gridspec(4, 3, width_ratios=[1.5, 1, 1], hspace=0.65, wspace=0.28)
examples = [(g, i) for g in range(2) for i in (np.where(kind == g)[0][np.argmin(amp[kind == g])],
                                               np.where(kind == g)[0][np.argmax(amp[kind == g])])]
for r, (g, i) in enumerate(examples):
    ax = fig.add_subplot(gs[r, 0])
    ax.plot(t, waves[i], color=TRACE_COLORS[g], lw=0.9)
    ax.set_title(f"{names[g]}, peak amplitude {amp[i]:.1f}", fontsize=18, loc="left", fontweight="normal")
    ax.set_xlim(0, 20)
    ax.tick_params(labelsize=14, labelbottom=(r == 3))
    ax.grid(True, alpha=0.4)
ax.set_xlabel("time (s)")
jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(kind))   # vertical spread, for viewing only
for c, (name, lab, vm) in enumerate([("Euclidean", labels_euc, vm_euc), ("Correlation", labels_corr, vm_corr)]):
    ax = fig.add_subplot(gs[:, c + 1])
    for j in range(2):
        m = lab == j
        ax.scatter(amp[m], kind[m] + jitter[m], s=55, color=COLORS[j], edgecolor="white", lw=0.4,
                   label=f"cluster {j} ({m.sum()})")
    ax.set_xscale("log")
    ax.set_yticks([0, 1], names)
    ax.set_ylim(-0.45, 1.75)
    ax.set_xlabel("peak amplitude")
    ax.set_title(f"{name}: V = {vm:.3f}", loc="left")
    ax.legend(fontsize=16, loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.3)
    ax.grid(True, alpha=0.4)
fig.suptitle("Synthetic seismograms: 75 impulsive and 75 emergent traces, peak amplitudes 1 to 100, noise s.d. 0.2 "
             "(average linkage, 2 clusters)", x=0.01, ha="left", fontsize=19, fontweight="normal", color=MUTED)
fig.subplots_adjust(left=0.04, right=0.99, bottom=0.09, top=0.88)
out = OUT / "waveform_correlation_slide.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out}")
