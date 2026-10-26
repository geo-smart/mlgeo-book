"""Slide figure: a Gaussian mixture on Argo temperature profiles finds the gyres (Maze et al., 2017).

Simplified rebuild of Maze, Mercier, Fablet, Tandeo, Lopez Radcenco, Lenca, Feucher
& Le Goff (2017), Coherent heat patterns revealed by unsupervised classification
of Argo temperature profiles in the North Atlantic Ocean, Progress in Oceanography
151, 275-292, doi:10.1016/j.pocean.2016.12.008 (accepted manuscript:
https://archimer.ifremer.fr/doc/00363/47431/47456.pdf).

Data: Argo core profiles from the Ifremer ERDDAP ArgoFloats table,
  https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats
first 10 days of every month of 2023 (an Argo float cycles every 10 days, so each
float appears about once per window), 0-70N, 90W-0E, adjusted pressure and
temperature, ascending profiles only, decimated on the server to the sample
closest to every 10 dbar (orderByClosest), cached in pooch.os_cache("mlgeo-figscripts").

Recipe, following the paper (section 3.5) where it can:
  QC flags 1, 2, 5, 8 on position, pressure and temperature (paper: same flags);
  pressure to depth (Saunders 1981); linear interpolation onto 5-1400 m every
  5 m (paper: 280 levels, 5 m grid, surface to 1400 m); profiles must start
  shallower than 10 m, reach 1400 m and have no gap over 100 m;
  training subset with at most CAP profiles per 2x2 degree cell, so that the
  densely sampled subpolar gyre does not dominate (paper: 10 per 2x2 cell);
  standardize each level with the training mean and standard deviation;
  PCA keeping the components that explain 99.88% of the variance (the paper's d = 11);
  Gaussian mixture with full covariance (scikit-learn EM; paper: Netlab EM);
  BIC (Eq. 4 of the paper) on 900-profile random subsets, 900 being the
  paper's estimate of independent profiles in the basin (150 km decorrelation
  scale), K = 1..15 (paper: 50 subsets, K = 1..30);
  the mapped classification uses the paper's K = 8; the BIC choice is printed too.

Regenerate: pixi run python book/slides/2026/figscripts/argo_profile_clusters.py
"""
import io
import urllib.parse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import requests  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
CACHE = Path(pooch.os_cache("mlgeo-figscripts"))
INK, MUTED = "#26241f", "#6e675c"
CLASS_COLORS = ["#2a78d6", "#1baf7a", "#6250d6", "#eda100", "#e87ba4", "#008300", "#eb6834", "#e34948"]
ERDDAP = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"
VARS = ("platform_number,cycle_number,time,latitude,longitude,position_qc,direction,"
        "pres_adjusted,pres_adjusted_qc,temp_adjusted,temp_adjusted_qc")
YEAR = 2023
LAT0, LAT1, LON0, LON1 = 0.0, 70.0, -90.0, 0.0
Z = np.arange(5.0, 1400.0 + 1, 5.0)          # 280 levels, as in the paper
GOOD_QC = {1, 2, 5, 8}
CAP = 4                                      # max training profiles per 2x2 degree cell
VAR_TARGET = 0.9988                          # paper: d = 11 keeps 99.88%
K_PAPER = 8
K_RANGE = range(1, 16)
N_BIC, N_REAL = 900, 20
SEED = 0
OUT = Path(__file__).resolve().parent.parent / "figs" / "lit_argo_clusters"
OUT.mkdir(parents=True, exist_ok=True)


def fetch_month(month):
    """First 10 days of one month, decimated to ~10 dbar on the server."""
    path = CACHE / f"argo_natl_{YEAR}_{month:02d}_d01-10_10dbar.csv.gz"
    if not path.exists():
        cons = [f"time>={YEAR}-{month:02d}-01T00:00:00Z", f"time<{YEAR}-{month:02d}-11T00:00:00Z",
                f"latitude>={LAT0}", f"latitude<={LAT1}", f"longitude>={LON0}", f"longitude<={LON1}",
                "pres_adjusted<=1500", 'orderByClosest("platform_number,cycle_number,pres_adjusted,10")']
        url = ERDDAP + "?" + urllib.parse.quote(VARS, safe=",") + "".join(
            "&" + urllib.parse.quote(c, safe="") for c in cons)
        r = requests.get(url, timeout=900)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(io.StringIO(r.text), skiprows=[1]).to_csv(path, index=False)
    return pd.read_csv(path)


def depth_from_pressure(p, lat):
    """Saunders (1981) pressure (dbar) to depth (m)."""
    c1 = (5.92 + 5.25 * np.sin(np.deg2rad(lat)) ** 2) * 1e-3
    return (1 - c1) * p - 2.21e-6 * p ** 2


# --- data -------------------------------------------------------------------
raw = pd.concat([fetch_month(m) for m in range(1, 13)], ignore_index=True)
n_rows_raw = len(raw)
n_prof_raw = raw.groupby(["platform_number", "cycle_number"]).ngroups
qc = lambda s: pd.to_numeric(s, errors="coerce").isin(GOOD_QC)  # noqa: E731
d = raw[(raw["direction"] == "A") & qc(raw["position_qc"]) & qc(raw["pres_adjusted_qc"])
        & qc(raw["temp_adjusted_qc"])].dropna(subset=["pres_adjusted", "temp_adjusted"])
d = d.drop_duplicates(["platform_number", "cycle_number", "pres_adjusted"]).sort_values(
    ["platform_number", "cycle_number", "pres_adjusted"])

rows, meta, reject = [], [], {"top>10m": 0, "bottom<1400m": 0, "gap>100m": 0, "few": 0}
for (plat, cyc), g in d.groupby(["platform_number", "cycle_number"], sort=False):
    if len(g) < 20:
        reject["few"] += 1
        continue
    z = depth_from_pressure(g["pres_adjusted"].to_numpy(), g["latitude"].iloc[0])
    t = g["temp_adjusted"].to_numpy()
    if z[0] > 10:
        reject["top>10m"] += 1
        continue
    if z[-1] < Z[-1]:
        reject["bottom<1400m"] += 1
        continue
    zz = z[z <= Z[-1] + 100]
    if np.max(np.diff(np.r_[min(zz[0], Z[0]), zz])) > 100:
        reject["gap>100m"] += 1
        continue
    rows.append(np.interp(Z, z, t))          # constant above the first sample (<= 10 m)
    meta.append((plat, cyc, g["time"].iloc[0], g["latitude"].iloc[0], g["longitude"].iloc[0]))
T = np.array(rows)
meta = pd.DataFrame(meta, columns=["platform", "cycle", "time", "lat", "lon"])
meta["time"] = pd.to_datetime(meta["time"])

# --- training subset: at most CAP profiles per 2x2 degree cell -------------
rng = np.random.default_rng(SEED)
cell = (np.floor(meta["lat"] / 2).astype(int).astype(str) + "_" + np.floor(meta["lon"] / 2).astype(int).astype(str))
order = rng.permutation(len(meta))
rank = pd.Series(np.arange(len(meta)), index=order).sort_index()
shuffled_cell = cell.iloc[order].reset_index(drop=True)
keep_sorted = shuffled_cell.groupby(shuffled_cell).cumcount() < CAP
train = np.sort(order[keep_sorted.to_numpy()])
n_cells = cell.nunique()
n_cells_full = (cell.value_counts() >= CAP).sum()

mu, sd = T[train].mean(0), T[train].std(0)
Xs = (T - mu) / sd
pca_full = PCA().fit(Xs[train])
cum = np.cumsum(pca_full.explained_variance_ratio_)
d_paper = int(np.searchsorted(cum, VAR_TARGET) + 1)
d_99 = int(np.searchsorted(cum, 0.99) + 1)
pca = PCA(n_components=d_paper).fit(Xs[train])
Y = pca.transform(Xs)
rmsd = np.sqrt(np.mean((pca.inverse_transform(Y) * sd + mu - T) ** 2, axis=0))

# --- BIC on 900-profile subsets (paper section 5) --------------------------
bic = np.full((N_REAL, len(K_RANGE)), np.nan)
for i in range(N_REAL):
    sub = rng.choice(len(Y), N_BIC, replace=False)
    for j, k in enumerate(K_RANGE):
        gm = GaussianMixture(k, covariance_type="full", n_init=2, random_state=SEED + i, reg_covar=1e-6).fit(Y[sub])
        bic[i, j] = gm.bic(Y[sub])
bic_mean, bic_sd = bic.mean(0), bic.std(0)
k_bic = list(K_RANGE)[int(np.argmin(bic_mean))]
k_bic_each = np.array(list(K_RANGE))[np.argmin(bic, axis=1)]
bic_train = [GaussianMixture(k, covariance_type="full", n_init=2, random_state=SEED).fit(Y[train]).bic(Y[train])
             for k in K_RANGE]
k_bic_train = list(K_RANGE)[int(np.argmin(bic_train))]


def classify(k):
    gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=SEED).fit(Y[train])
    post = gm.predict_proba(Y)
    lab = post.argmax(1)
    # order classes from coldest to warmest 0-1400 m mean temperature, as class 1 = coldest in the paper
    tmean = np.array([T[lab == c].mean() if np.any(lab == c) else np.inf for c in range(k)])
    remap = np.empty(k, int); remap[np.argsort(tmean)] = np.arange(k)
    lab = remap[lab]
    prior = np.empty(k); prior[remap] = gm.weights_
    pm = post.max(1)
    R = (pm - 1 / k) * k / (k - 1)          # paper Eq. 7, labelling robustness
    return lab, prior, R


lab, prior, R = classify(K_PAPER)
meta["cls"] = lab + 1
if k_bic != K_PAPER:
    lab_bic, prior_bic, _ = classify(k_bic)

# --- figure -------------------------------------------------------------------
land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip",
                                    known_hash=None, fname="ne_50m_land.zip", path=CACHE, progressbar=False))
fig = plt.figure(figsize=(24, 9.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1, 1], wspace=0.25,
                      left=0.04, right=0.99, bottom=0.12, top=0.92)
a0, a1, a2 = (fig.add_subplot(gs[i]) for i in range(3))

a0.set_facecolor("#eef3f8")
land.plot(ax=a0, color="#f4efe6", edgecolor=MUTED, lw=0.6)
plot_order = rng.permutation(len(meta))
a0.scatter(meta["lon"].to_numpy()[plot_order], meta["lat"].to_numpy()[plot_order],
           c=[CLASS_COLORS[c] for c in lab[plot_order]], s=11, lw=0)
a0.set_xlim(LON0, LON1 + 2); a0.set_ylim(LAT0, LAT1)
a0.set_aspect(1 / np.cos(np.deg2rad(35)))
a0.set_xticks([-80, -60, -40, -20, 0]); a0.set_xticklabels(["80°W", "60°W", "40°W", "20°W", "0°"])
a0.set_yticks([0, 20, 40, 60]); a0.set_yticklabels(["0°", "20°N", "40°N", "60°N"])
a0.set_xlabel(""); a0.set_ylabel("")
for c in range(K_PAPER):
    sel = lab == c
    a0.text(np.median(meta["lon"][sel]), np.median(meta["lat"][sel]), str(c + 1), fontsize=17, fontweight="bold",
            ha="center", va="center", color=INK,
            bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=CLASS_COLORS[c], lw=3))
a0.set_title(f"(a) {len(meta):,} profiles, colored by class", loc="left")

for c in range(K_PAPER):
    m = T[lab == c].mean(0)
    a1.plot(m, Z, color=CLASS_COLORS[c], lw=3, label=f"{c + 1}  ({np.mean(lab == c):.0%})")
a1.set_ylim(Z[-1], 0); a1.set_xlim(0, 29)
a1.set_xlabel("temperature (°C)"); a1.set_ylabel("depth (m)")
a1.grid(color="#e6e1d8", lw=0.8)
a1.legend(title="class (share)", fontsize=15, title_fontsize=15, loc="lower right", frameon=False,
          handlelength=1.2, labelspacing=0.3)
a1.set_title(f"(b) Class means, K = {K_PAPER}", loc="left")

ks = np.array(list(K_RANGE))
a2.fill_between(ks, (bic_mean - bic_sd) / 1e3, (bic_mean + bic_sd) / 1e3, color="#d9d3c7", lw=0)
a2.plot(ks, bic_mean / 1e3, "o-", color=INK, lw=2, ms=7)
a2.plot([k_bic], [bic_mean.min() / 1e3], "o", ms=16, mfc="none", mec="#c0392b", mew=3)
a2.axvline(K_PAPER, color=MUTED, ls="--", lw=1.5)
a2.text(K_PAPER + 0.25, 0.97, f"paper: K = {K_PAPER}", transform=a2.get_xaxis_transform(), fontsize=16,
        color=MUTED, va="top")
a2.set_xlabel("number of classes K"); a2.set_ylabel("BIC (thousands)")
a2.set_xticks([1, 4, 8, 12, 15])
a2.grid(color="#e6e1d8", lw=0.8)
a2.set_title(f"(c) BIC lowest at K = {k_bic}", loc="left")

fig.text(a0.get_position().x0, -0.06,
         f"Argo profiles via Ifremer ERDDAP (ArgoFloats), days 1–10 of each month of {YEAR}, 0–70°N, 90°W–0°, "
         f"QC 1/2/5/8, 5–1400 m every 5 m\nstandardized, PCA {d_paper} components "
         f"({100 * cum[d_paper - 1]:.2f}% of variance), Gaussian mixture, full covariance · "
         f"BIC: mean ± 1 s.d. over {N_REAL} random sets of {N_BIC} profiles · after Maze et al. (2017)",
         fontsize=13, color=MUTED)
out = OUT / "argo_clusters.png"
fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")

# --- numbers ------------------------------------------------------------------
print(f"raw rows {n_rows_raw:,}, raw profiles {n_prof_raw:,}; QC-passed ascending profiles "
      f"{d.groupby(['platform_number', 'cycle_number']).ngroups:,}; rejected {reject}")
print(f"profiles kept {len(meta):,} from {meta['platform'].nunique()} floats, "
      f"{meta['time'].min():%Y-%m-%d} to {meta['time'].max():%Y-%m-%d}; per month "
      f"{meta['time'].dt.month.value_counts().sort_index().to_dict()}")
print(f"depth levels {len(Z)} ({Z[0]:.0f}-{Z[-1]:.0f} m every {Z[1] - Z[0]:.0f} m)")
print(f"training subset {len(train):,} profiles (cap {CAP} per 2x2 cell; {n_cells} occupied cells, "
      f"{n_cells_full} at the cap)")
print(f"PCA: d = {d_99} for 99%, d = {d_paper} for {100 * VAR_TARGET:.2f}% (kept {d_paper}, "
      f"{100 * cum[d_paper - 1]:.3f}%); first 5 ratios {np.round(pca_full.explained_variance_ratio_[:5], 4).tolist()}")
print(f"PCA reconstruction RMSD: max {rmsd.max():.3f} °C (at {Z[rmsd.argmax()]:.0f} m), "
      f"max below 600 m {rmsd[Z >= 600].max():.3f} °C")
print(f"BIC (mean over {N_REAL} x {N_BIC}): minimum K = {k_bic}; per-realization minima "
      f"{np.bincount(k_bic_each, minlength=16)[1:].tolist()} for K=1..15")
print("BIC mean (thousands):", dict(zip(ks.tolist(), np.round(bic_mean / 1e3, 2).tolist())))
print(f"BIC on full training subset ({len(train)} profiles): minimum K = {k_bic_train} "
      f"(edge of range = {k_bic_train == ks[-1]})")
print(f"K = {K_PAPER} classes (ordered coldest to warmest 0-1400 m mean):")
for c in range(K_PAPER):
    sel = lab == c
    m = T[sel].mean(0)
    print(f"  class {c + 1}: n = {sel.sum():5d} ({sel.mean():5.1%}), prior {prior[c]:.3f}, "
          f"median lat {np.median(meta['lat'][sel]):5.1f}, lon {np.median(meta['lon'][sel]):6.1f}; "
          f"T at 5/300/1000/1400 m = {m[0]:.1f}/{m[Z == 300][0]:.1f}/{m[Z == 1000][0]:.1f}/{m[-1]:.1f} °C; "
          f"0-1400 m mean {m.mean():.2f} °C; R > 0.99 for {np.mean(R[sel] > 0.99):.0%}")
print(f"all profiles: R > 0.99 (virtually certain) {np.mean(R > 0.99):.1%}, R < 0.33 {np.mean(R < 0.33):.1%}")
if k_bic != K_PAPER:
    print(f"K = {k_bic} (BIC) class counts: {np.bincount(lab_bic, minlength=k_bic).tolist()}")
