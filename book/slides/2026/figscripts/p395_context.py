"""Slide figure: where GNSS station P395 is, and what its motion records.

Left: map of the Cascadia margin with P395, the plate boundaries, and the
station's MIDAS velocity relative to stable North America. Right: the daily
east and north positions in the same North-America-fixed frame, with the
MIDAS rates drawn through the median position.

Every value is fetched from its provider at run time, nothing is typed in:
  station coordinates   NGL llh.out
  velocities            NGL MIDAS, IGS20 solution, NA-fixed (Blewitt et al., 2016)
  daily positions       NGL tenv3, IGS20 solution, NA-fixed (Blewitt et al., 2018)
  plate boundaries      PB2002 (Bird, 2003)
  land and state lines  Natural Earth 10m

Regenerate: pixi run python book/slides/2026/figscripts/p395_context.py
"""
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})

STATION = "P395"
NGL = "https://geodesy.unr.edu"
CACHE = pooch.os_cache("mlgeo-figscripts")
INK, MUTED, GRID = "#26241f", "#6e675c", "#d9d4ca"
SERIES, TREND = "#2a78d6", "#eb6834"      # dataviz reference slots 1 and 2


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


def station_coords():
    path = fetch(f"{NGL}/NGLStationPages/llh.out", "ngl_llh.out")
    for line in open(path):
        parts = line.split()
        if parts and parts[0] == STATION:
            return float(parts[1]), float(parts[2])
    raise KeyError(STATION)


def midas_na():
    """MIDAS columns 3-4 epochs, 9-11 ENU velocity (m/yr), 12-14 uncertainty."""
    path = fetch(f"{NGL}/gps_timeseries/IGS20/midas/midas.NA.txt", "midas.NA.txt")
    for line in open(path):
        p = line.split()
        if p and p[0] == STATION:
            v = [float(x) * 1000 for x in p[8:11]]
            s = [float(x) * 1000 for x in p[11:14]]
            return dict(t0=float(p[2]), t1=float(p[3]), ve=v[0], vn=v[1], vu=v[2],
                        se=s[0], sn=s[1], su=s[2])
    raise KeyError(STATION)


def positions_na():
    path = fetch(f"{NGL}/gps_timeseries/IGS20/tenv3/NA/{STATION}.NA.tenv3", f"{STATION}.NA.tenv3")
    df = pd.read_csv(path, sep=r"\s+")
    t = df["yyyy.yyyy"].to_numpy()
    e = (df["_e0(m)"] + df["__east(m)"]).to_numpy() * 1000
    n = (df["____n0(m)"] + df["_north(m)"]).to_numpy() * 1000
    return t, e - np.median(e), n - np.median(n)


def basemap():
    land = gpd.read_file(fetch(
        "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip", "ne_10m_land.zip"))
    states = gpd.read_file(fetch(
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces_lines.zip",
        "ne_10m_admin_1_lines.zip"))
    plates = gpd.read_file(fetch(
        "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json",
        "PB2002_boundaries.json"))
    return land, states, plates


lat, lon = station_coords()
vel = midas_na()
t, east, north = positions_na()
land, states, plates = basemap()
retrieved = dt.date.today().isoformat()

fig = plt.figure(figsize=(16, 7.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], wspace=0.18, hspace=0.32)
ax = fig.add_subplot(gs[:, 0])
extent = (-128.5, -119.0, 40.8, 49.3)
box = dict(xmin=extent[0], xmax=extent[1], ymin=extent[2], ymax=extent[3])
ax.set_facecolor("#eef3f8")
land.cx[box["xmin"]:box["xmax"], box["ymin"]:box["ymax"]].plot(ax=ax, color="#f4efe6", edgecolor=MUTED, lw=0.8)
states.cx[box["xmin"]:box["xmax"], box["ymin"]:box["ymax"]].plot(ax=ax, color=MUTED, lw=0.6, linestyle="--")
plates.cx[box["xmin"]:box["xmax"], box["ymin"]:box["ymax"]].plot(ax=ax, color=INK, lw=2.2)
ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
ax.set_aspect(1 / np.cos(np.deg2rad(45)))
ax.tick_params(labelsize=14)
ax.set_xticks([-128, -125, -122, -119]); ax.set_yticks([42, 45, 48])
ax.xaxis.set_major_formatter(lambda x, _: f"{-x:.0f}°W"); ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}°N")

scale = 0.15                                   # degrees of longitude per mm/yr
ax.annotate("", xy=(lon + vel["ve"] * scale, lat + vel["vn"] * scale * np.cos(np.deg2rad(lat))),
            xytext=(lon, lat), arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.9",
                                               color=TREND, lw=3.5))
ax.plot(lon, lat, marker="^", ms=17, color=SERIES, mec="white", mew=2, zorder=5)
ax.text(lon + 0.25, lat - 0.45, STATION, fontsize=19, fontweight="bold", color=INK)
ax.text(-127.9, 46.4, "Juan de Fuca\nplate", fontsize=15, color=MUTED, style="italic")
ax.text(-122.6, 43.6, "North America\nplate", fontsize=15, color=MUTED, style="italic")
ax.text(-126.15, 42.1, "Cascadia\nsubduction zone", fontsize=14, color=INK, rotation=83, ha="center")
ax.annotate("", xy=(-121.6 + 10 * scale, 41.3), xytext=(-121.6, 41.3),
            arrowprops=dict(arrowstyle="-|>", color=TREND, lw=3))
ax.text(-121.6, 41.5, "10 mm/yr", fontsize=14, color=INK)
ax.set_title("Location and velocity", loc="left")

for row, (y, v, s, comp) in enumerate([(east, vel["ve"], vel["se"], "East"),
                                       (north, vel["vn"], vel["sn"], "North")]):
    a = fig.add_subplot(gs[row, 1])
    a.plot(t, y, ".", ms=2.5, color=SERIES, alpha=0.6, rasterized=True)
    tm = np.median(t)
    a.plot([t.min(), t.max()], [v * (t.min() - tm), v * (t.max() - tm)], color=TREND, lw=2.5)
    a.set_ylabel(f"{comp.lower()} (mm)")
    a.grid(color=GRID, lw=0.8); a.spines[["top", "right"]].set_visible(False)
    a.text(0.02, 0.92, f"{comp} {v:+.1f} ± {s:.1f} mm/yr (MIDAS)", transform=a.transAxes,
           fontsize=18, fontweight="bold", color=INK, va="top")
    if row == 0:
        a.set_title("Daily position relative to stable North America", loc="left")
a.set_xlabel("year")

fig.suptitle(f"GNSS station {STATION}, Oregon Coast Range ({lat:.2f}°N, {-lon:.2f}°W) · "
             f"NGL IGS20 daily solutions, North America fixed · MIDAS velocities retrieved {retrieved}",
             x=0.01, ha="left", fontsize=15, color=MUTED)
out = Path(__file__).resolve().parent.parent / "figs" / "1.7_get_geodetic_gnss" / "p395_context_slide.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")
print({k: round(v, 3) for k, v in vel.items()}, f"lat={lat} lon={lon}", f"n={len(t)} t={t.min()}-{t.max()}")
