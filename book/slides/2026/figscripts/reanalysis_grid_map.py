"""Slide figure: the 2.5 reanalysis grid over North America, and which cell is Seattle.

Two-year mean of NCEP/NCAR Reanalysis 1 air temperature at sigma 0.995
(Kalnay et al., 1996), 2013-2014, on its 2.5 degree grid, with Seattle's grid
cell outlined. Same file and checksum as notebook 2.5:
  grid        pydata/xarray-data air_temperature.nc at commit 2baf0c2
  coastlines  Natural Earth 1:50m

Regenerate: pixi run python book/slides/2026/figscripts/reanalysis_grid_map.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pooch  # noqa: E402
import xarray as xr  # noqa: E402

plt.rcParams.update({"font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold"})
CACHE = pooch.os_cache("mlgeo-figscripts")
INK, MUTED = "#26241f", "#6e675c"

grid = pooch.retrieve(
    "https://github.com/pydata/xarray-data/raw/2baf0c22d9671a2058415f17e71b3fe06239a3dd/air_temperature.nc",
    known_hash="sha256:c606b89c35970a2983b914b76df4adbb409003ef34aa7cfd7f582e41f307482b",
    fname="air_temperature_raw.nc", path=CACHE, progressbar=False)
coast = gpd.read_file(pooch.retrieve(
    "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip",
    known_hash="sha256:640f805509b822f57f4840a2e18d9ff2412cf1cf6976124701c2789436166fde",
    fname="ne_50m_coastline.zip", path=CACHE, progressbar=False))

air = xr.open_dataset(grid)["air"]
mean_c = (air.mean("time") - 273.15).assign_coords(lon=air.lon - 360)
cell = mean_c.sel(lat=47.61, lon=-122.33, method="nearest", tolerance=1.25)

fig, ax = plt.subplots(figsize=(13, 7.2))
mean_c.plot(ax=ax, cmap="RdYlBu_r", cbar_kwargs={"label": "mean air temperature (°C)", "shrink": 0.8})
coast.plot(ax=ax, color=INK, linewidth=0.7)
lat0, lon0 = float(cell.lat), float(cell.lon)
ax.add_patch(plt.Rectangle((lon0 - 1.25, lat0 - 1.25), 2.5, 2.5, fill=False, edgecolor="k", lw=3))
ax.plot(-122.33, 47.61, marker="*", color="k", markersize=16)
ax.annotate("Seattle's cell\n47.5° N, 237.5° E", (lon0 + 1.4, lat0), xytext=(14, 0), textcoords="offset points",
            fontsize=17, va="center", color=INK, bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
ax.plot(-30, lat0, marker="X", color="k", markersize=14)
ax.annotate("nearest cell to 237.7\non the −180…180° copy", (-30, lat0), xytext=(-46, 32), textcoords="data",
            fontsize=16, ha="center", color=INK, bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.set_xlim(-160, -30); ax.set_ylim(15, 75)
ax.set_xlabel("longitude (°)"); ax.set_ylabel("latitude (°N)")
ax.set_title("One value per 2.5° cell, 2013–2014 mean", loc="left")
fig.text(0.01, 0.005, "Air temperature at sigma 0.995 (Kalnay et al., 1996) · file: pydata/xarray-data @2baf0c2, "
         "sha256 c606b89c… · coastlines: Natural Earth 1:50m", fontsize=11, color=MUTED)

out = Path(__file__).resolve().parent.parent / "figs" / "2.5_Arrays" / "reanalysis_grid_map_slide.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")
