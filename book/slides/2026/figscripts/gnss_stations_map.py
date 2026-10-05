"""Slide figure: where the course GNSS stations P395 and P563 are, and how they move.

Map of the western United States plate boundary with both stations, their
horizontal MIDAS velocities relative to stable North America, and their
vertical rates. Every value is fetched from its provider at run time:
  station coordinates   NGL llh.out
  velocities            NGL MIDAS, IGS20 solution, NA-fixed (Blewitt et al., 2016)
  plate boundaries      PB2002 (Bird, 2003)
  land and state lines  Natural Earth 10m

Regenerate: pixi run python book/slides/2026/figscripts/gnss_stations_map.py
"""
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pooch  # noqa: E402

plt.rcParams.update({"font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold"})

STATIONS = ("P395", "P563")
NGL = "https://geodesy.unr.edu"
CACHE = pooch.os_cache("mlgeo-figscripts")
INK, MUTED = "#26241f", "#6e675c"
STATION_C, ARROW_C = "#2a78d6", "#eb6834"     # dataviz reference slots 1 and 2


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


def coords():
    out = {}
    for line in open(fetch(f"{NGL}/NGLStationPages/llh.out", "ngl_llh.out")):
        p = line.split()
        if p and p[0] in STATIONS:
            out[p[0]] = (float(p[1]), float(p[2]))
    return out


def midas_na():
    """MIDAS columns 9-11: east, north, up velocity (m/yr)."""
    out = {}
    for line in open(fetch(f"{NGL}/gps_timeseries/IGS20/midas/midas.NA.txt", "midas.NA.txt")):
        p = line.split()
        if p and p[0] in STATIONS:
            out[p[0]] = [float(x) * 1000 for x in p[8:11]]
    return out


xy, vel = coords(), midas_na()
land = gpd.read_file(fetch("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
                           "ne_10m_land.zip"))
states = gpd.read_file(fetch(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces_lines.zip",
    "ne_10m_admin_1_lines.zip"))
plates = gpd.read_file(fetch(
    "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json",
    "PB2002_boundaries.json"))

extent = (-128.0, -113.0, 32.0, 48.5)
fig, ax = plt.subplots(figsize=(9, 9.5))
ax.set_facecolor("#eef3f8")
for gdf, kw in ((land, dict(color="#f4efe6", edgecolor=MUTED, lw=0.8)),
                (states, dict(color=MUTED, lw=0.6, linestyle="--")),
                (plates, dict(color=INK, lw=2.2))):
    gdf.cx[extent[0]:extent[1], extent[2]:extent[3]].plot(ax=ax, **kw)
ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
ax.set_aspect(1 / np.cos(np.deg2rad(40)))
ax.tick_params(labelsize=17)
ax.set_xticks([-126, -122, -118, -114])
ax.xaxis.set_major_formatter(lambda x, _: f"{-x:.0f}°W")
ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}°N")

scale = 0.12                                   # degrees per mm/yr
offsets = {"P395": (0.35, -0.35), "P563": (0.6, 0.75)}
for sta in STATIONS:
    lat, lon = xy[sta]
    ve, vn, vu = vel[sta]
    ax.annotate("", xy=(lon + ve * scale, lat + vn * scale * np.cos(np.deg2rad(lat))),
                xytext=(lon, lat),
                arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.9", color=ARROW_C, lw=3.5))
    ax.plot(lon, lat, marker="^", ms=17, color=STATION_C, mec="white", mew=2, zorder=5)
    dx, dy = offsets[sta]
    ax.text(lon + dx, lat + dy,
            sta,
            fontsize=19, color=INK, va="top", fontweight="bold")

ax.text(-127.6, 36.0, "Pacific\nplate", fontsize=18, color=MUTED, style="italic")
ax.text(-118.2, 40.2, "North America\nplate", fontsize=18, color=MUTED, style="italic")
ax.text(-126.7, 42.6, "Cascadia\nsubduction zone", fontsize=16, color=INK, rotation=82, ha="center")
ax.text(-121.4, 34.0, "San Andreas\nfault system", fontsize=16, color=INK, rotation=-38, ha="center")
ax.annotate("", xy=(-116.6 + 10 * scale, 32.8), xytext=(-116.6, 32.8),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=3))
ax.text(-116.6, 33.05, "10 mm/yr", fontsize=16, color=INK)
ax.set_title("Course GNSS stations", loc="left")
fig.text(0.01, 0.005, f"Coordinates: NGL llh.out · velocities: NGL MIDAS IGS20,\nNorth America fixed, "
         f"retrieved {dt.date.today().isoformat()} · plate boundaries: PB2002 (Bird, 2003)",
         fontsize=12, color=MUTED)

out = Path(__file__).resolve().parent.parent / "figs" / "1.7_get_geodetic_gnss" / "gnss_stations_map_slide.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")
print({s: [round(v, 2) for v in vel[s]] for s in STATIONS}, xy)
