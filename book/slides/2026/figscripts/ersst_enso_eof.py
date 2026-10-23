"""Slide figure: the first EOF of tropical Pacific sea surface temperature is El Niño.

NOAA ERSST v5 monthly SST (Huang et al. 2017, J. Climate, doi:10.1175/JCLI-D-16-0836.1),
read through OPeNDAP so only the region is transferred:
  https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc
Region 20S-20N, 120E-280E; 1950-01 to 2025-12; anomalies from the 1991-2020
monthly climatology; sqrt(cos(latitude)) weights; EOFs by SVD (same recipe as
notebook 2.12). PC1 is compared with the NOAA PSL Nino 3.4 index:
  https://psl.noaa.gov/data/correlation/nina34.anom.data

Regenerate: pixi run python book/slides/2026/figscripts/ersst_enso_eof.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import requests  # noqa: E402
import xarray as xr  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
})
CACHE = Path(pooch.os_cache("mlgeo-figscripts"))
INK, MUTED = "#26241f", "#6e675c"
URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc"

sub = CACHE / "ersst_v5_tropical_pacific_1950_2025.nc"
if not sub.exists():
    ds = xr.open_dataset(URL)
    ds["sst"].sel(lat=slice(20, -20), lon=slice(120, 280), time=slice("1950-01-01", "2025-12-31")).load().to_netcdf(sub)
sst = xr.open_dataarray(sub)
clim = sst.sel(time=slice("1991-01-01", "2020-12-31")).groupby("time.month").mean("time")
anom = (sst.groupby("time.month") - clim).drop_vars("month")
w = np.sqrt(np.cos(np.deg2rad(anom["lat"])))
X = (anom * w).stack(space=("lat", "lon")).transpose("time", "space")
ocean = np.isfinite(X).all("time")
Xo = X.sel(space=ocean).values
U, S, Vt = np.linalg.svd(Xo, full_matrices=False)
frac = S**2 / np.sum(S**2)
pc1 = U[:, 0] * S[0]
eof1 = np.full(X.sizes["space"], np.nan); eof1[ocean.values] = Vt[0]
eof1 = xr.DataArray(eof1, coords={"space": X["space"]}, dims="space").unstack("space") / w
# sign: positive in the eastern Pacific (Nino 3.4 box)
box = eof1.sel(lat=slice(5, -5), lon=slice(190, 240)).mean().item()
if box < 0:
    eof1, pc1 = -eof1, -pc1
pc1 = pd.Series(pc1 / pc1.std(), index=pd.to_datetime(anom["time"].values))

# NOAA PSL Nino 3.4 index (year + 12 monthly values; missing as -99.99)
txt = requests.get("https://psl.noaa.gov/data/correlation/nina34.anom.data", timeout=60).text
lines = txt.splitlines()
rows = [l.split() for l in lines[1:] if len(l.split()) == 13 and l.split()[0].isdigit()]
idx = pd.Series({pd.Timestamp(int(r[0]), m + 1, 1): float(r[m + 1]) for r in rows for m in range(12)}).sort_index()
idx = idx[idx > -90]
both = pd.concat([pc1.rename("pc1"), idx.rename("nino34")], axis=1, sort=True).dropna()
r = both.corr().iloc[0, 1]

land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip",
                                    known_hash=None, fname="ne_110m_land.zip", path=CACHE, progressbar=False))
fig = plt.figure(figsize=(22, 8.4))
gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], height_ratios=[1, 1], wspace=0.30, hspace=0.42)
a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
lim = np.nanmax(np.abs(eof1.values))
m = a0.pcolormesh(eof1["lon"], eof1["lat"], eof1, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
land.translate(xoff=360).plot(ax=a0, color="#f4efe6", edgecolor=MUTED, lw=0.6)
land.plot(ax=a0, color="#f4efe6", edgecolor=MUTED, lw=0.6)
a0.set_xlim(120, 280); a0.set_ylim(-20, 20); a0.set_aspect("equal")
a0.set_xticks([150, 180, 210, 240, 270]); a0.set_xticklabels(["150°E", "180°", "150°W", "120°W", "90°W"])
a0.set_yticks([-20, 0, 20]); a0.set_yticklabels(["20°S", "0°", "20°N"]); a0.set_xlabel(""); a0.set_ylabel("")
a0.set_title(f"EOF1, {100 * frac[0]:.1f}% of the variance", loc="left")
cb = fig.colorbar(m, ax=a0, orientation="vertical", fraction=0.025, pad=0.01); cb.set_label("loading")
a1.plot(both.index, both["nino34"], color="#c0392b", lw=1.2, label="Niño 3.4 index (°C)")
a1.plot(both.index, both["pc1"], color=INK, lw=1.2, alpha=0.8, label="PC1 (standardized)")
a1.axhline(0, color=MUTED, lw=0.8)
a1.set_title(f"PC1 and Niño 3.4, r = {r:.2f}", loc="left")
a1.legend(fontsize=15, loc="lower left", ncol=2, frameon=False)
a2.bar(np.arange(1, 11), 100 * frac[:10], color=MUTED)
a2.bar([1], [100 * frac[0]], color=INK)
a2.set_xlabel("mode"); a2.set_ylabel("variance explained (%)"); a2.set_xticks([1, 2, 3, 5, 10])
a2.set_title("Variance by mode", loc="left")
fig.text(0.125, -0.04, f"NOAA ERSST v5, 20°S–20°N, 120°E–80°W, 1950–2025 ({len(anom.time)} months, {int(ocean.sum())} ocean grid points); "
         "anomalies from 1991–2020 monthly means; Niño 3.4 index: NOAA PSL (ERSST v6, anomalies from 1981–2010).", fontsize=13, color=MUTED)
out = Path(__file__).resolve().parent.parent / "figs" / "lec13_dimensionality" / "ersst_enso_eof.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")
print(f"months {len(anom.time)}, ocean points {int(ocean.sum())}, N/D = {len(anom.time) / int(ocean.sum()):.2f}")
print("variance fraction modes 1-5:", np.round(frac[:5], 3).tolist())
print(f"corr(PC1, Nino3.4) = {r:.3f} over {len(both)} months; index header: {lines[0].strip()[:80]} | {lines[-3:]}")
