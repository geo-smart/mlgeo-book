"""Slide figure: where the 2.3 earthquake catalog's events are, colored by depth.

Map of the M >= 6 catalog used in section 2.3 (Global_Quakes_IRIS.csv, the
EarthScope/IRIS event service export in the UW-MLGEO dataset repository), with
plate boundaries. Depths in the file are in meters; the figure converts to km,
the same first transform the notebook makes. Every input is fetched at run time:
  events             UW-MLGEO/MLGeo-dataset, data/Global_Quakes_IRIS.csv
  plate boundaries   PB2002 (Bird, 2003)
  land               Natural Earth 110m

Regenerate: pixi run python book/slides/2026/figscripts/global_quakes_map.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

plt.rcParams.update({"font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold"})

CACHE = pooch.os_cache("mlgeo-figscripts")
INK, MUTED = "#26241f", "#6e675c"


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


quake = pd.read_csv(fetch(
    "https://raw.githubusercontent.com/UW-MLGEO/MLGeo-dataset/refs/heads/main/data/Global_Quakes_IRIS.csv",
    "Global_Quakes_IRIS.csv"))
quake["depth_km"] = quake["depth"] / 1000.0
quake = quake.sort_values("depth_km")              # deep events drawn last, on top
land = gpd.read_file(fetch("https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip",
                           "ne_110m_land.zip"))
plates = gpd.read_file(fetch(
    "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json",
    "PB2002_boundaries.json"))

# Sequential, one hue, light to dark; the lightest 30% is cut so shallow
# events stay visible against the land fill.
ramp = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.35, 1.0, 256)))

fig, ax = plt.subplots(figsize=(16, 8.4))
ax.set_facecolor("#eef3f8")
# Pacific-centered (-20 to 340 E) so the circum-Pacific subduction zones and
# the Tonga-Fiji deep cluster are not split at the antimeridian.
for shift in (0, 360):
    land.translate(xoff=shift).plot(ax=ax, color="#f4efe6", edgecolor=MUTED, lw=0.5)
    plates.translate(xoff=shift).plot(ax=ax, color=MUTED, lw=1.0)
lon = quake["longitude"].where(quake["longitude"] >= -20, quake["longitude"] + 360)
sc = ax.scatter(lon, quake["latitude"], c=quake["depth_km"], cmap=ramp,
                vmin=0, vmax=700, s=6 * (quake["magnitude"] - 5.5) ** 3 + 8,
                edgecolors="white", linewidths=0.6, zorder=3)
ax.set_xlim(-20, 340); ax.set_ylim(-70, 80)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01, shrink=0.8)
cb.set_label("depth (km)", color=INK)
cb.ax.invert_yaxis()                               # deeper is lower
t0, t1 = quake["time"].min()[:10], quake["time"].max()[:10]
ax.set_title(f"{len(quake):,} earthquakes M ≥ 6, {t0} to {t1}", loc="left")
fig.text(0.125, 0.13, "Events: EarthScope/IRIS catalog export (UW-MLGEO/MLGeo-dataset, "
         "Global_Quakes_IRIS.csv) · plate boundaries: PB2002 (Bird, 2003) · "
         "symbol size scales with magnitude", fontsize=12, color=MUTED)

out = Path(__file__).resolve().parent.parent / "figs" / "2.3_pandas_rendered" / "global_quakes_map_slide.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")
deep = quake["depth_km"] > 300
print(f"n={len(quake)}  depth max {quake['depth_km'].max():.1f} km  "
      f"deeper than 300 km: {deep.sum()}  M max {quake['magnitude'].max()}  "
      f"largest: {quake.loc[quake['magnitude'].idxmax(), ['time', 'description']].tolist()}")
