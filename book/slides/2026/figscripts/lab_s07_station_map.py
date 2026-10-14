"""Book figure for the Session 7 lab: where the five lab datasets were recorded.

Left: the 2021-07-29 Mww 8.2 Chignik, Alaska, earthquake and the seismic
station UW.RATT that recorded it, with the great-circle path. Right: western
Washington with the Nooksack River gauge (station 0), the SeaTac weather station
(A), the ComCat query box and its epicentres (B), RATT (D), and the NCEP/NCAR
Reanalysis 1 grid cell nearest SeaTac (E).

Every coordinate is read at run time from the snapshot files in
book/Chapter2-DataManipulation/snapshots/ (provider values recorded in the
provenance JSON files, the epicentre list of the ComCat CSV, and the grid of
the NetCDF file). Land polygons: Natural Earth 1:10m and 1:50m.

Regenerate: pixi run python book/slides/2026/figscripts/lab_s07_station_map.py
Output:     book/Chapter2-DataManipulation/lab_s07_station_map.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import xarray as xr  # noqa: E402
from pyproj import Geod  # noqa: E402

BOOK = Path(__file__).resolve().parents[3]
CH2 = BOOK / "Chapter2-DataManipulation"
SNAP = CH2 / "snapshots"
OUT = CH2 / "lab_s07_station_map.png"
CACHE = pooch.os_cache("mlgeo-figscripts")
INK, MUTED, LAND, SEA = "#26241f", "#6e675c", "#f4efe6", "#eef3f8"
COL = {"0": "#2a78d6", "A": "#eb6834", "B": "#1baf7a", "D": "#eda100", "E": "#e87ba4"}   # dataviz slots 1-5


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


def prov(name):
    return json.loads((SNAP / f"{name}.provenance.json").read_text())


nook = prov("usgs_12213100_2021-11")
ghcn = prov("ghcnd_USW00024233_1948-2024")
ratt = prov("ratt_chignik_2021-07-29")
comcat = prov("comcat_puget_2000-2025")
sta, ev = ratt["station"], ratt["event"]
box = comcat["query"]
quakes = pd.read_csv(SNAP / "comcat_puget_2000-2025.csv", usecols=["latitude", "longitude"])
with xr.open_dataset(SNAP / "ncep_air_2013-2014.nc") as ds:
    cell = ds["air"].sel(lat=ghcn["latitude"], lon=360 + ghcn["longitude"], method="nearest")
    cell_lat, cell_lon = float(cell.lat), float(cell.lon) - 360
    half = abs(float(ds.lat[0] - ds.lat[1])) / 2

geod = Geod(ellps="WGS84")
_, _, dist_m = geod.inv(ev["longitude"], ev["latitude"], sta["longitude"], sta["latitude"])
path = np.array(geod.npts(ev["longitude"], ev["latitude"], sta["longitude"], sta["latitude"], 100))

land10 = gpd.read_file(fetch("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip", "ne_10m_land.zip"))
land50 = gpd.read_file(fetch("https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip", "ne_50m_land.zip"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
for ax, land, ext, lat0, step in ((ax1, land50, (-166, -112, 40, 64), 52, (10, 5)),
                                  (ax2, land10, (-125.2, -120.2, 45.2, 49.3), 47.3, (1, 1))):
    ax.set_facecolor(SEA)
    land.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=ax, color=LAND, edgecolor=MUTED, lw=0.6)
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
    ax.set_aspect(1 / np.cos(np.deg2rad(lat0)))
    ax.xaxis.set_major_formatter(lambda x, _: f"{-x:.0f}°W")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}°N")
    ax.set_xticks(np.arange(np.ceil(ext[0] / step[0]) * step[0], ext[1], step[0]))
    ax.set_yticks(np.arange(np.ceil(ext[2] / step[1]) * step[1], ext[3], step[1]))
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(labelsize=10, colors=MUTED)

label = dict(fontsize=10.5, color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))

# Left: the teleseismic path
ax1.plot(path[:, 0], path[:, 1], color=INK, lw=1.4, ls="--")
ax1.plot(ev["longitude"], ev["latitude"], marker="*", ms=20, color=COL["D"], mec=INK, mew=1)
ax1.annotate(f"Chignik, Alaska\n{ev['magnitude_type'].capitalize()} {ev['magnitude']}, {ev['origin_time_utc'][:10]}",
             (ev["longitude"], ev["latitude"]), xytext=(-10, -38), textcoords="offset points", ha="left", **label)
ax1.plot(sta["longitude"], sta["latitude"], marker="^", ms=11, color=COL["D"], mec=INK, mew=1)
ax1.annotate("D  UW.RATT", (sta["longitude"], sta["latitude"]),
             xytext=(8, 8), textcoords="offset points", **label)
mid = path[len(path) // 2]
ax1.annotate(f"{dist_m / 1e3:,.0f} km", (mid[0], mid[1]), xytext=(0, 10), textcoords="offset points",
             ha="center", **label)
ax1.add_patch(plt.Rectangle((-125.2, 45.2), 5.0, 4.1, fill=False, ec=INK, lw=1.2))
ax1.annotate("right panel", (-125.2, 45.2), xytext=(-4, 0), ha="right", textcoords="offset points", fontsize=9.5,
             color=MUTED)
ax1.set_title("Station D: the earthquake and the seismometer", fontsize=12, color=INK, loc="left")

# Right: western Washington
ax2.scatter(quakes["longitude"], quakes["latitude"], s=0.6, color=COL["B"], alpha=0.25, lw=0, rasterized=True)
ax2.add_patch(plt.Rectangle((box["minlongitude"], box["minlatitude"]), box["maxlongitude"] - box["minlongitude"],
                            box["maxlatitude"] - box["minlatitude"], fill=False, ec=COL["B"], lw=2, ls="--"))
ax2.annotate("B  ComCat query box\nand its epicentres", (box["minlongitude"], box["minlatitude"]),
             xytext=(6, 6), textcoords="offset points", **label)
ax2.add_patch(plt.Rectangle((cell_lon - half, cell_lat - half), 2 * half, 2 * half, fill=False, ec=COL["E"], lw=2.5))
ax2.annotate(f"E  NCEP grid cell\n{cell_lat:.1f}°N, {360 + cell_lon:.1f}°E", (cell_lon - half, cell_lat + half),
             xytext=(4, -32), textcoords="offset points", **label)
points = (("0", nook["latitude"], nook["longitude"], "v", "0  Nooksack River\nat Ferndale", (-8, 4), "right"),
          ("A", ghcn["latitude"], ghcn["longitude"], "o", "A  SeaTac airport", (-9, -4), "right"),
          ("D", sta["latitude"], sta["longitude"], "^", "D  RATT", (8, -4), "left"))
for key, lat, lon, m, text, off, ha in points:
    ax2.plot(lon, lat, marker=m, ms=11, color=COL[key], mec=INK, mew=1, ls="none")
    ax2.annotate(text, (lon, lat), xytext=off, textcoords="offset points", ha=ha, va="center", **label)
ax2.set_title("Stations 0, A, B, D and E", fontsize=12, color=INK, loc="left")

fig.tight_layout()
fig.savefig(OUT, dpi=150, facecolor="white")
print(f"wrote {OUT.relative_to(BOOK.parent)}; Chignik to RATT {dist_m / 1e3:,.0f} km; "
      f"NCEP cell {cell_lat} N, {360 + cell_lon} E")
