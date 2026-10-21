"""Slide figures: rerunning the aftershock-location test of DeVries et al. (2018).

DeVries, Viegas, Wattenberg & Meade (2018, Nature 560, 632-634) trained a
13,451-parameter neural network to say whether a 5 km cell near a large
earthquake hosts aftershocks, from the six components of the static stress
change. Mignan & Broccardo (2019, Nature 574, E1-E3) showed that one scalar
stress metric in a two-parameter logistic regression scores the same AUC.

This script reruns the comparison from the authors' own open release instead
of copying the papers' figures:
  code     https://github.com/phoebemrdevries/Learning-aftershock-location-patterns
  data     Google Drive folder linked from that README (one CSV per slip model,
           the train/test file lists, the trained weights, fault outlines)
           https://drive.google.com/drive/folders/1lAHfdjFd-Uv3wJcA0Tk2ViDIZeFt0mCA
  land     Natural Earth 10m land
The network's forward pass is rewritten in numpy from TheBestWeights.h5 (the
original code is Python 2.7 with Keras on Theano). Coulomb stress follows the
authors' Calculate_ROC_and_AUC_OtherQuantities.py: best of four sign conventions
per slip model, 2010 Darfield excluded.

The 57 test CSVs total about 390 MB; they are cached in
pooch.os_cache("mlgeo-figscripts")/devries2018/ after the first run.
Writes figs/lit_aftershock_2018/aftershock_rerun.png and prints every number quoted on the slide.
Regenerate: pixi run python book/slides/2026/figscripts/aftershock_auc_rerun.py
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import pyproj  # noqa: E402
import requests  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
CACHE = Path(pooch.os_cache("mlgeo-figscripts"))
INK, MUTED, ACCENT, GOOD, GOLD, BLUE = "#26241f", "#6e675c", "#c0392b", "#1f6f6b", "#b7791f", "#1f5f8b"
OUT = Path(__file__).resolve().parent.parent / "figs" / "lit_aftershock_2018"
OUT.mkdir(parents=True, exist_ok=True)
FIG_H = 7.4   # inches; sets the aspect ratio to fit the slide column

# Google Drive file ids of the authors' release (top-level folder listing)
DRIVE = {
    "Testing_FileNames.h5": "1GTF9n9s1gmvEX-9NixGrpSkCKuDa3tnL",
    "Training_FileNames.h5": "1vaYcELdnOFc4QVm0Y1z1Cl_ToepLiNTa",
    "TheBestWeights.h5": "13SBwHm7xx7d3oS3BR1L75DIgncYMuEq-",
    "1999CHICHI01MAxx.h5": "1ajquVfzW3YAoYu2AFo6No4ZYSnFlNokv",
}
ALLCSV_FOLDER = "1c5Rb_6EsuP2XedDjg37bFDyf8AadtGDa"
STRESS = ["stresses_full_xx", "stresses_full_yy", "stresses_full_xy",
          "stresses_full_xz", "stresses_full_yz", "stresses_full_zz"]


def drive_url(file_id):
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


def fetch_drive(name, file_id):
    return fetch(drive_url(file_id), f"devries2018/{name}")


def csv_ids():
    """Map CSV file name -> Drive id from the public AllCSV folder listing."""
    html = requests.get(f"https://drive.google.com/embeddedfolderview?id={ALLCSV_FOLDER}", timeout=60).text
    pairs = re.findall(r'href="https://drive.google.com/file/d/([^/]+)/view[^"]*".*?flip-entry-title">([^<]+)<',
                       html, re.S)
    return {name: fid for fid, name in pairs}


def h5_names(path, key):
    with h5py.File(path, "r") as h:
        return [x.decode() for x in h[key][:]]


def nn_predict(stress, weights_path):
    """Forward pass of the 12-50-50-50-50-50-50-1 tanh network (modelfunctions.create_model)."""
    s = np.abs(stress) / 1.0e6
    a = np.hstack([s, -s])
    with h5py.File(weights_path, "r") as h:
        layers = sorted(h["model_weights"], key=lambda k: int(k.split("_")[1]) if k.startswith("dense") else -1)
        layers = [k for k in layers if k.startswith("dense")]
        n_par = 0
        for i, k in enumerate(layers):
            g = h["model_weights"][k][k]
            a = a @ g["kernel"][:] + g["bias"][:]
            n_par += g["kernel"].size + g["bias"].size
            a = np.tanh(a) if i < len(layers) - 1 else 1.0 / (1.0 + np.exp(-a))
    return a.ravel(), n_par


# ---- train/test split by slip model ----------------------------------------------------
test_files = h5_names(fetch_drive("Testing_FileNames.h5", DRIVE["Testing_FileNames.h5"]), "file_names_testing")
train_files = h5_names(fetch_drive("Training_FileNames.h5", DRIVE["Training_FileNames.h5"]), "file_names_training")
seq = lambda f: f[:10]   # year + region code, e.g. 1997COLFIO; two digits and an author code follow
shared = sorted({f[:-9] for f in test_files if seq(f) in {seq(g) for g in train_files}})
print(f"slip models: {len(train_files)} train, {len(test_files)} test")
print(f"test slip models whose year+region tag also appears in training: {len(shared)} {shared}")

# ---- rerun on the 57 test slip models ------------------------------------------------------
weights = fetch_drive("TheBestWeights.h5", DRIVE["TheBestWeights.h5"])
ids = csv_ids()
lab, nn, msh, vm, sab, lab_c, cfs = [], [], [], [], [], [], []
chichi = None
for f in test_files:
    df = pd.read_csv(fetch_drive(f, ids[f]))
    y = (df["aftershocksyn"].to_numpy() > 0).astype(int)
    if len(np.unique(y)) < 2:
        continue
    p, n_par = nn_predict(df[STRESS].to_numpy(), weights)
    lab.append(y); nn.append(p)
    msh.append(df["stresses_full_max_shear"].to_numpy()); vm.append(df["von_mises"].to_numpy())
    sab.append(np.abs(df[STRESS].to_numpy()).sum(axis=1))
    if f != "2010DARFIE01ATZO_grid.csv":
        best = max(range(1, 5), key=lambda k: roc_auc_score(y, df[f"stresses_full_cfs_{k}"]))
        lab_c.append(y); cfs.append(df[f"stresses_full_cfs_{best}"].to_numpy())
    if f == "1999CHICHI01MAxx_grid.csv":
        chichi = df.assign(nn=p, cfs=df[f"stresses_full_cfs_{best}"])
y, yc = np.concatenate(lab), np.concatenate(lab_c)
print(f"test cells: {y.size:,}; with aftershocks: {y.sum():,} ({100 * y.mean():.1f}%); network parameters: {n_par:,}")
curves = {
    "Neural network, 13,451 parameters": (y, np.concatenate(nn), INK, "-"),
    "Maximum shear stress": (y, np.concatenate(msh), GOOD, "--"),
    "von Mises criterion": (y, np.concatenate(vm), BLUE, ":"),
    "Sum of |stress components|": (y, np.concatenate(sab), GOLD, "-."),
    "Coulomb failure stress": (yc, np.concatenate(cfs), ACCENT, "-"),
}
auc = {k: roc_auc_score(t, s) for k, (t, s, _, _) in curves.items()}
for k, v in auc.items():
    print(f"AUC {v:.3f}  {k}")

# ---- figure: Chi-Chi map (left) and ROC curves on the whole test set (right) -------------------
to_ll = pyproj.Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)   # UTM 51N (Taiwan)
z0 = -7500.0                                                                       # depth slice the authors plot
sl = chichi[chichi["z"] == z0].copy()
sl["lon"], sl["lat"] = to_ll.transform(sl["x"].to_numpy(), sl["y"].to_numpy())
grid = sl.pivot(index="y", columns="x", values="nn")
X, Y = np.meshgrid(grid.columns.to_numpy(), grid.index.to_numpy())
LON, LAT = to_ll.transform(X, Y)
with h5py.File(fetch_drive("1999CHICHI01MAxx.h5", DRIVE["1999CHICHI01MAxx.h5"]), "r") as h:
    fx = np.column_stack([h[f"x{i}Utm"][:, 0] for i in (1, 2, 4, 3)])
    fy = np.column_stack([h[f"y{i}Utm"][:, 0] for i in (1, 2, 4, 3)])
land = gpd.read_file(fetch("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip", "ne_10m_land.zip"))

fig = plt.figure(figsize=(12.4, FIG_H))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)
ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
pm = ax.pcolormesh(LON, LAT, grid.to_numpy(), cmap="Greys", vmin=0, vmax=1, shading="nearest", alpha=0.85)
land.boundary.plot(ax=ax, color=BLUE, lw=1.6)
for xs, ys in zip(fx, fy):
    lo, la = to_ll.transform(xs, ys)
    ax.fill(lo, la, facecolor="none", edgecolor=GOLD, lw=0.8)
hit = sl[sl["aftershocksyn"] > 0]
ax.scatter(hit["lon"], hit["lat"], s=30, color=ACCENT, edgecolor="white", lw=0.5, zorder=5,
           label=f"cell with aftershocks ({len(hit)})")
ax.plot([], [], color=GOLD, lw=2.5, label="fault slip model")
ax.set_xlim(LON.min(), LON.max()); ax.set_ylim(LAT.min(), LAT.max())
ax.set_aspect(1 / np.cos(np.deg2rad(LAT.mean())))
ax.set_xticks([120.0, 121.0, 122.0])
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.set_title("1999 Chi-Chi, 5–10 km depth", loc="left", fontsize=20)
ax.legend(loc="upper left", fontsize=14.5, framealpha=0.92, borderaxespad=0.3, handletextpad=0.4)
cax = ax.inset_axes([0.04, 0.745, 0.30, 0.03])
cb = fig.colorbar(pm, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
cb.ax.tick_params(labelsize=14)
cax.set_title("network probability", fontsize=15, fontweight="normal", pad=4)

for k, (t, s, c, ls) in curves.items():
    fpr, tpr, _ = roc_curve(t, s)
    ax2.plot(fpr, tpr, color=c, ls=ls, lw=3.4 if k.startswith("Neural") else 2.8,
             label=f"{auc[k]:.3f}  {k.split(',')[0]}")
ax2.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (2, 3)))
ax2.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate")
ax2.set_aspect("equal")
ax2.set_title(f"{len(lab)} test slip models, {y.size / 1e6:.2f} M cells", loc="left", fontsize=20)
ax2.legend(title="AUC (stress change metrics)", loc="lower right", bbox_to_anchor=(1.05, -0.01), fontsize=14, title_fontsize=14.5, frameon=False,
           handlelength=2.0, borderaxespad=0.1, alignment="left")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
fig.savefig(OUT / "aftershock_rerun.png", dpi=110, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ---- records for the slide table ---------------------------------------------------------------
lon, lat = to_ll.transform(chichi["x"].to_numpy(), chichi["y"].to_numpy())
rec = chichi.assign(lon=lon, lat=lat, depth_km=-chichi["z"] / 1000,
                    max_shear_kPa=chichi["stresses_full_max_shear"] / 1e3,
                    cfs_kPa=chichi["cfs"] / 1e3)
near = rec[(rec["z"] == z0)].sort_values("nn", ascending=False)
pick = pd.concat([near[near["aftershocksyn"] > 0].head(2), near[near["aftershocksyn"] == 0].head(1),
                  near[near["aftershocksyn"] > 0].tail(1)])
pd.set_option("display.width", 200)
print(pick[["lon", "lat", "depth_km", "max_shear_kPa", "cfs_kPa", "nn", "aftershocksyn"]].round(3).to_string())
print("Chi-Chi cells:", len(chichi), "with aftershocks:", int(chichi["aftershocksyn"].gt(0).sum()))
print("Counter of test sequences:", Counter(seq(f) for f in test_files).most_common(3))
