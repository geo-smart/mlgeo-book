"""Slide figure (lecture 8): equipment offsets and outages in a real GNSS record, Mount St. Helens.

GNSS station TGUA sits on Mount St. Helens, Washington, at 1,618 m ellipsoidal
height (coordinates from NGL llh.out). It started recording on 2004-10-21,
during the 2004-2008 dome-building eruption. The causes of its outages are not
recorded in the files used here.
  (a-c) daily east, north and up positions, IGS20 frame (NGL tenv3), with
        the equipment changes NGL lists in its step file (gray lines) and
        every outage of 7 days or longer (shaded)
  (d)   two years around the antenna-mount change of 2012-08-15, north
        component, with a line + step fit (+-90 days) that measures the offset
  (b)   also carries two velocity estimates: a straight line fit through all
        positions, ignoring the steps, and the NGL MIDAS velocity, which is
        insensitive to steps (Blewitt et al., 2016, JGR Solid Earth,
        doi:10.1002/2015JB012552)
Every value is fetched from NGL at run time: positions (tenv3, IGS20), step
dates (steps.txt), coordinates (llh.out) and velocities (MIDAS, IGS20 frame).
The offset sizes and the straight-line velocity are computed here and labeled so.

Regenerate: pixi run python book/slides/2026/figscripts/lec08_real_gnss_steps.py
"""
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import requests  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED, GAP = "#26241f", "#6e675c", "#e2d3b8"
POS_C, NAIVE_C, MIDAS_C = "#2a78d6", "#eb6834", "#1baf7a"   # dataviz reference slots 1, 2, 3
NGL = "https://geodesy.unr.edu"
CACHE = pooch.os_cache("mlgeo-figscripts")
STA = "TGUA"
FOCUS = pd.Timestamp("2012-08-15")            # NGL step file: Antenna_Mount_Changed
MIN_OUTAGE = 7                                # days


def get(url):
    return requests.get(url, timeout=600).text


# ---------------------------------------------------------------- provider values
d = pd.read_csv(io.StringIO(get(f"{NGL}/gps_timeseries/IGS20/tenv3/IGS20/{STA}.tenv3")), sep=r"\s+")
d["t"] = pd.to_datetime(d["YYMMMDD"], format="%y%b%d")
d = d.set_index("t")
pos = pd.DataFrame({"E": d["_e0(m)"] + d["__east(m)"], "N": d["____n0(m)"] + d["_north(m)"],
                    "U": d["u0(m)"] + d["____up(m)"]})
pos = (pos - pos.iloc[0]) * 1e3                                       # mm from the first day

steps = []
for ln in get(f"{NGL}/NGLStationPages/steps.txt").splitlines():
    p = ln.split()
    if len(p) >= 4 and p[0] == STA and p[2] == "1":
        steps.append((pd.to_datetime(p[1], format="%y%b%d"), p[3]))
lat = lon = h = None
for ln in get(f"{NGL}/NGLStationPages/llh.out").splitlines():
    p = ln.split()
    if p and p[0] == STA:
        lat, lon, h = float(p[1]), float(p[2]), float(p[3])
        lon = lon - 360 if lon > 180 else lon
midas = None
for ln in get(f"{NGL}/gps_timeseries/IGS20/midas/midas.IGS.txt").splitlines():
    p = ln.split()
    if p and p[0] == STA:
        midas = dict(E=float(p[8]) * 1e3, N=float(p[9]) * 1e3, U=float(p[10]) * 1e3,
                     sE=float(p[11]) * 1e3, sN=float(p[12]) * 1e3, sU=float(p[13]) * 1e3)

# ---------------------------------------------------------------- outages and offsets
days = pd.date_range(pos.index[0], pos.index[-1])
missing = days.difference(pos.index)
runs = []
if len(missing):
    r0 = prev = missing[0]
    for m in missing[1:]:
        if (m - prev).days > 1:
            runs.append((r0, prev))
            r0 = m
        prev = m
    runs.append((r0, prev))
outages = [(a, b) for a, b in runs if (b - a).days + 1 >= MIN_OUTAGE]


def step_fit(c, t, w=90):
    sub = pos[c][t - pd.Timedelta(days=w):t + pd.Timedelta(days=w)]
    x = (sub.index - t).days.values.astype(float)
    A = np.c_[np.ones_like(x), x, (x >= 0).astype(float)]
    coef, *_ = np.linalg.lstsq(A, sub.values, rcond=None)
    return coef, sub


offsets = {c: step_fit(c, FOCUS)[0][2] for c in "ENU"}
yrs = (pos.index - pos.index[0]).days.values / 365.25
naive = {c: np.polyfit(yrs, pos[c].values, 1) for c in "ENU"}           # slope mm/yr, intercept

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12.5, 7.4))
gs = fig.add_gridspec(3, 2, width_ratios=[3.3, 1], hspace=0.45, wspace=0.16)
axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
for ax in axs[1:]:
    ax.sharex(axs[0])
axm = fig.add_subplot(gs[0:2, 1])
axz = fig.add_subplot(gs[2, 1])
names = {"E": "east", "N": "north", "U": "up"}
for ax, c, tag in zip(axs, "ENU", "abc"):
    for a, b in outages:
        ax.axvspan(a, b + pd.Timedelta(days=1), color=GAP, lw=0)
    for t, why in steps:
        ax.axvline(t, color=MUTED, lw=0.9, ls=":")
    ax.plot(pos.index, pos[c], ".", ms=1.6, color=POS_C)
    ax.axvline(FOCUS, color=INK, lw=1.4)
    ax.set_ylabel(f"{names[c]} (mm)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_title(f"(a–c) {STA}: {len(pos):,} days, {len(outages)} outages ≥ {MIN_OUTAGE} days (shaded), "
                 f"{len(steps)} step-list entries (dotted)",
                 loc="left", fontsize=17)
x_line = pd.DatetimeIndex([pos.index[0], pos.index[-1]])
y_yrs = (x_line - pos.index[0]).days.values / 365.25
axs[1].plot(x_line, np.polyval(naive["N"], y_yrs), color=NAIVE_C, lw=2.4,
            label=f"line through all data: {naive['N'][0]:+.1f} mm/yr")
for k, seg in enumerate((pos["N"][:FOCUS - pd.Timedelta(days=1)], pos["N"][FOCUS:])):
    sy = (seg.index - pos.index[0]).days.values / 365.25
    xs = pd.DatetimeIndex([seg.index[0], seg.index[-1]])
    ys = (xs - pos.index[0]).days.values / 365.25
    axs[1].plot(xs, np.median(seg.values - midas["N"] * sy) + midas["N"] * ys, color=MIDAS_C, lw=2.4,
                label=f"NGL MIDAS: {midas['N']:+.1f} mm/yr" if k == 0 else None)
axs[1].legend(frameon=False, loc="lower right", fontsize=14)
axs[0].annotate(f"antenna mount changed\n{FOCUS:%Y-%m-%d}", (FOCUS, pos["E"].max()), xytext=(8, -4),
                textcoords="offset points", fontsize=14, va="top", color=INK)
for ax in axs[:2]:
    ax.tick_params(labelbottom=False)
axs[2].xaxis.set_major_locator(mdates.YearLocator(4))
axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# (d) zoom on the mount change, north
coef, sub = step_fit("N", FOCUS)
z0, z1 = FOCUS - pd.Timedelta(days=365), FOCUS + pd.Timedelta(days=365)
zz = pos["N"][z0:z1]
axz.plot(zz.index, zz.values, ".", ms=3, color=POS_C)
xf = (sub.index - FOCUS).days.values.astype(float)
axz.plot(sub.index, coef[0] + coef[1] * xf + coef[2] * (xf >= 0), color=INK, lw=2)
axz.axvline(FOCUS, color=INK, lw=1.0)
axz.set_title(f"(d) north, 2011–2013:\n{offsets['N']:+.0f} mm in one day", loc="left", fontsize=16)
axz.xaxis.set_major_locator(mdates.YearLocator())
axz.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axz.set_ylabel("mm")
for sp in ("top", "right"):
    axz.spines[sp].set_visible(False)

# location
land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
                                    known_hash=None, fname="ne_10m_land.zip", path=CACHE, progressbar=False))
states = gpd.read_file(pooch.retrieve(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces_lines.zip",
    known_hash=None, fname="ne_10m_admin_1_lines.zip", path=CACHE, progressbar=False))
places = gpd.read_file(pooch.retrieve(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip",
    known_hash=None, fname="ne_10m_populated_places_simple.zip", path=CACHE, progressbar=False))
ext = (-125.0, -119.5, 44.6, 48.6)
axm.set_facecolor("#eef3f8")
land.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=axm, color="#f4efe6", edgecolor=MUTED, lw=0.6)
states.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=axm, color=MUTED, lw=0.6, linestyle="--")
for city in ("Seattle", "Portland"):
    row = places[(places["name"] == city) & (places["adm1name"].isin(["Washington", "Oregon"]))].iloc[0]
    axm.plot(row.geometry.x, row.geometry.y, "s", ms=7, color=MUTED)
    axm.text(row.geometry.x + 0.12, row.geometry.y, city, fontsize=13, color=MUTED, va="center")
axm.plot(lon, lat, marker="^", ms=18, color=POS_C, mec="white", mew=2)
axm.text(lon + 0.2, lat + 0.08, f"{STA}\nMount St. Helens", fontsize=14, fontweight="bold", va="bottom")
axm.set_xlim(ext[:2]); axm.set_ylim(ext[2:])
axm.set_aspect(1 / np.cos(np.deg2rad(46.5)))
axm.set_xticks([]); axm.set_yticks([])
axm.set_xlabel(""); axm.set_ylabel("")
axm.set_title(f"{lat:.3f}°N, {-lon:.3f}°W\n{h:,.0f} m ellipsoidal height", fontsize=14, fontweight="normal", color=MUTED)

out = Path(__file__).resolve().parent.parent / "figs" / "lec08_real" / "gnss_steps_tgua.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"{STA} llh.out {lat:.6f} {lon:.6f} {h:.1f} m; tenv3 {pos.index[0].date()} to {pos.index[-1].date()}, "
      f"{len(pos)} daily positions over {len(days)} days ({len(missing)} missing)")
print(f"outages >= {MIN_OUTAGE} d: {[(str(a.date()), (b - a).days + 1) for a, b in outages]}")
print(f"NGL steps.txt equipment entries ({len(steps)} entries on {len({t for t, _ in steps})} dates): "
      f"{[(str(t.date()), w) for t, w in steps]}")
print(f"offset at {FOCUS.date()} (line+step fit, +-90 d): " + ", ".join(f"{c} {offsets[c]:+.1f} mm" for c in "ENU"))
print("velocity mm/yr, straight line through all data vs NGL MIDAS (IGS20 frame, +- sigma): "
      + "; ".join(f"{c} {naive[c][0]:+.2f} vs {midas[c]:+.2f} +- {midas['s' + c]:.2f}" for c in "ENU"))
