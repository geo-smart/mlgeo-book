"""Slide figure: GNSS stations rise as the 2013-2014 drought removes water (Borsa et al., 2014).

Rebuilds the lecture-4 literature example from the open archive the course
uses (Nevada Geodetic Laboratory, NGL), not from the paper's figures:
  station list      NGL DataHoldings.txt (coordinates, first and last day)
  daily positions   NGL tenv3 files, IGS20 solution (Blewitt et al., 2018)
  equipment and earthquake steps   NGL steps.txt
  Central Valley outline           USGS principal aquifers of the United States (aquifers_us)
  land and state lines             Natural Earth 10m
Borsa et al. (2014) used 771 stations, mostly from the EarthScope Plate
Boundary Observatory (PBO), processed in 2014. This script uses the PBO
stations (four-character IDs starting with P) as NGL distributes them today.

Method (a simplified version of the paper's):
  1. keep PBO stations west of 109 W, 31-49 N, with data from before 2007 to after mid-2015
  2. exclude, as the paper did, stations in the Central Valley aquifer system (pumping) and around
     Long Valley caldera (unrest and uplift since 2011); also exclude stations subsiding faster
     than 3 mm/yr before 2013 (pumped aquifers elsewhere)
  3. fit offset + rate + annual + semiannual + NGL steps to the vertical over 2006.0-2013.0
  4. residual = vertical minus that model extrapolated forward; drop stations with a step
     between 2013.0 and 2014.5 or a pre-drought residual RMS above 6 mm
  5. uplift in March 2014 = median residual over 1-31 March 2014

The bulk download (about 460 tenv3 files, 1.3 MB each) is reduced to a compact
table of vertical positions cached in pooch.os_cache("mlgeo-figscripts"), so a
rerun does not download again. The featured station's tenv3 is cached whole for
the record table on the slide.

Regenerate: pixi run python book/slides/2026/figscripts/drought_uplift_2014.py
"""
import datetime as dt
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import requests  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold",
    "axes.labelsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
NGL = "https://geodesy.unr.edu"
CACHE = Path(pooch.os_cache("mlgeo-figscripts"))
INK, MUTED = "#26241f", "#6e675c"
MEDIAN_C, STATION_C = "#26241f", "#c0392b"
OUT = Path(__file__).resolve().parent.parent / "figs" / "lit_drought_uplift_2014"
OUT.mkdir(parents=True, exist_ok=True)
FIT = (2006.0, 2013.0)
MARCH14 = (2014 + 59 / 365, 2014 + 90 / 365)      # 1-31 March 2014 in decimal years
LONG_VALLEY = (37.35, 38.05, -119.35, -118.45)    # Long Valley caldera and its surroundings: lat_min, lat_max, lon_min, lon_max


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


def stations():
    rows = [ln.split()[:11] for ln in open(fetch(f"{NGL}/NGLStationPages/DataHoldings.txt",
                                                 "ngl_DataHoldings.txt")).readlines()[1:]]
    d = pd.DataFrame(rows, columns="sta lat lon h x y z beg end mod n".split())
    d[["lat", "lon", "h"]] = d[["lat", "lon", "h"]].astype(float)
    d["lon"] = (d["lon"] + 180) % 360 - 180
    d["beg"] = pd.to_datetime(d["beg"], errors="coerce")
    d["end"] = pd.to_datetime(d["end"], errors="coerce")
    keep = (d["lat"].between(31, 49) & d["lon"].between(-125.5, -109) & d["sta"].str.match(r"^P\d{3}$")
            & (d["beg"] <= "2007-01-01") & (d["end"] >= "2015-06-01"))
    return d[keep].set_index("sta")[["lat", "lon", "h"]]


def vertical(sta):
    """Decimal year and vertical position (m) from one NGL tenv3 file, 2005.5-2016.5."""
    r = requests.get(f"{NGL}/gps_timeseries/IGS20/tenv3/IGS20/{sta}.tenv3", timeout=60)
    r.raise_for_status()
    t = pd.read_csv(io.StringIO(r.text), sep=r"\s+", usecols=[2, 11, 12], header=0,
                    names=None)
    t.columns = ["t", "u0", "up"]
    t = t[(t["t"] > 2005.5) & (t["t"] < 2016.5)]
    return pd.DataFrame({"sta": sta, "t": t["t"].values, "up": (t["u0"] + t["up"]).values})


def all_verticals(sta_list):
    path = CACHE / "lit_drought_2014_verticals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    with ThreadPoolExecutor(4) as ex:
        tabs = list(ex.map(vertical, sta_list))
    df = pd.concat(tabs, ignore_index=True)
    df.to_parquet(path)
    return df


def steps():
    """NGL steps.txt: code 1 = equipment change; code 2 = earthquake within NGL's radius 10^(0.5M-0.79) km.
    That radius is generous (400 km for the M 6.8 offshore Ferndale event of 10 March 2014), so an
    earthquake counts here only if the station lies within 10^(0.5M-1.6) km, about 63 km for M 6.8."""
    s = []
    for ln in open(fetch(f"{NGL}/NGLStationPages/steps.txt", "ngl_steps.txt")):
        p = ln.split()
        if len(p) < 4:
            continue
        if p[2] == "2" and len(p) >= 6 and float(p[4]) > 10 ** (0.5 * float(p[5]) - 1.6):
            continue
        d = dt.datetime.strptime(p[1], "%y%b%d")
        s.append((p[0], d.year + (d.timetuple().tm_yday - 0.5) / 365.25))
    return pd.DataFrame(s, columns=["sta", "t"])


def fit_station(g, st):
    t, u = g["t"].values, g["up"].values * 1000            # mm
    st = np.unique(np.round(st, 4))
    if ((st > FIT[1]) & (st < 2014.5)).any():
        return None, "step"
    st = st[(st > FIT[0]) & (st < FIT[1])]
    w = (t >= FIT[0]) & (t < FIT[1])
    if w.sum() < 1500:
        return None, "short"

    def design(tt):
        cols = [np.ones_like(tt), tt - 2010]
        for k in (1, 2):
            cols += [np.sin(2 * np.pi * k * tt), np.cos(2 * np.pi * k * tt)]
        cols += [(tt >= s).astype(float) for s in st]
        return np.column_stack(cols)

    coef, *_ = np.linalg.lstsq(design(t[w]), u[w], rcond=None)
    res = u - design(t) @ coef
    if np.sqrt(np.mean(res[w] ** 2)) > 6:
        return None, "noisy"
    return pd.DataFrame({"t": t, "res": res}), coef[1]


meta = stations()
raw = all_verticals(list(meta.index))
stp = steps()
lv = (meta["lat"].between(*LONG_VALLEY[:2]) & meta["lon"].between(*LONG_VALLEY[2:]))
aq = gpd.read_file(fetch("https://water.usgs.gov/GIS/dsdl/aquifers_us.zip", "usgs_aquifers_us.zip"))
cv = aq[aq["AQ_NAME"] == "Central Valley aquifer system"].to_crs(4326)
pts = gpd.GeoDataFrame(meta, geometry=gpd.points_from_xy(meta["lon"], meta["lat"]), crs=4326)
in_cv = pts.within(cv.union_all())
res, rate, why = {}, {}, {}
for sta, g in raw.groupby("sta"):
    if in_cv.get(sta, False):
        why[sta] = "Central Valley"; continue
    if lv.get(sta, False):
        why[sta] = "Long Valley"; continue
    out, info = fit_station(g, stp.loc[stp["sta"] == sta, "t"].values)
    if out is None:
        why[sta] = info; continue
    if info < -3:
        why[sta] = "subsiding"; continue
    res[sta], rate[sta] = out, info
meta["mar14"] = np.nan
for sta, r in res.items():
    m = r[(r["t"] >= MARCH14[0]) & (r["t"] < MARCH14[1])]["res"]
    if len(m) >= 10:
        meta.loc[sta, "mar14"] = m.median()
used = meta.dropna(subset=["mar14"])
sierra = used[used["lat"].between(36, 40) & used["lon"].between(-121.2, -117.8)]
mountain = sierra[sierra["h"] > 800]                 # ellipsoidal height above 800 m: Sierra Nevada proper
feat = mountain["mar14"].idxmax()

# weekly median of all residual series
allres = pd.concat([r.assign(sta=s) for s, r in res.items() if s in used.index])
allres["wk"] = (allres["t"] * 52).round() / 52
wk = allres.groupby(["wk", "sta"])["res"].mean().groupby("wk")
med, q1, q3 = (x.rolling(13, center=True, min_periods=7).mean() for x in (wk.median(), wk.quantile(0.25), wk.quantile(0.75)))
fr = res[feat]; fr = fr.assign(wk=(fr["t"] * 52).round() / 52).groupby("wk")["res"].mean()
fr = fr.reindex(np.round(np.arange(fr.index.min() * 52, fr.index.max() * 52 + 1)) / 52)   # gaps stay gaps

# ---- figure: map (left) and series (right) ----
land = gpd.read_file(fetch("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip", "ne_10m_land.zip"))
states = gpd.read_file(fetch(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces_lines.zip",
    "ne_10m_admin_1_lines.zip"))
fig = plt.figure(figsize=(14.5, 9.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45], height_ratios=[1, 1], wspace=0.22, hspace=0.42)
ax = fig.add_subplot(gs[:, 0])
extent = (-125.2, -109.0, 31.2, 49.2)
ax.set_facecolor("#eef3f8")
land.cx[extent[0]:extent[1], extent[2]:extent[3]].plot(ax=ax, color="#f4efe6", edgecolor=MUTED, lw=0.7)
states.cx[extent[0]:extent[1], extent[2]:extent[3]].plot(ax=ax, color=MUTED, lw=0.5, linestyle="--")
cv.plot(ax=ax, color="#d9d4ca", edgecolor="none", zorder=2)
excl = meta.loc[[s for s in meta.index if s not in used.index]]
ax.scatter(excl["lon"], excl["lat"], s=10, color="#b9b2a6", lw=0, zorder=3, label="excluded")
sc = ax.scatter(used["lon"], used["lat"], c=used["mar14"].clip(-12, 12), cmap="RdBu_r", vmin=-12, vmax=12,
                s=42, edgecolor="white", lw=0.4, zorder=4)
for s, dx, dy in ((feat, 0.4, 0.25), ("P395", 0.4, -0.1)):
    if s in meta.index:
        ax.plot(meta.loc[s, "lon"], meta.loc[s, "lat"], marker="^", ms=15, color="none", mec=INK, mew=2.2, zorder=6)
        ax.text(meta.loc[s, "lon"] + dx, meta.loc[s, "lat"] + dy, s, fontsize=17, fontweight="bold", color=INK, zorder=7)
ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
ax.set_aspect(1 / np.cos(np.deg2rad(40)))
ax.set_xticks([-124, -118, -112]); ax.set_yticks([32, 36, 40, 44, 48])
ax.xaxis.set_major_formatter(lambda x, _: f"{-x:.0f}°W")
ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}°N")
ax.tick_params(labelsize=15); ax.set_xlabel(""); ax.set_ylabel("")
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.06, extend="both")
cb.set_label("vertical change, March 2014 (mm)\ngray: excluded; shaded: Central Valley aquifer", fontsize=15); cb.ax.tick_params(labelsize=15)
ax.set_title(f"{len(used)} PBO stations", loc="left")

a1 = fig.add_subplot(gs[0, 1])
a1.plot(fr.index, fr.values, color=STATION_C, lw=1.2)
a1.axvline(2013, color=MUTED, ls="--", lw=1.5); a1.axhline(0, color=MUTED, lw=0.8)
a1.set_xlim(2006, 2016.5); a1.set_ylim(-20, 26); a1.set_ylabel("mm")
a1.set_title(f"{feat}, Sierra Nevada, {meta.loc[feat, 'h']:,.0f} m", loc="left")
a1.text(2006.3, 0.95, "weekly mean; trend and seasons\nfit over 2006–2012 removed", transform=a1.get_xaxis_transform(),
        fontsize=15, color=MUTED, va="top")
a2 = fig.add_subplot(gs[1, 1], sharex=a1)
a2.fill_between(med.index, q1.values, q3.values, color="#d9d4ca", lw=0)
a2.plot(med.index, med.values, color=MEDIAN_C, lw=2.2)
a2.axvline(2013, color=MUTED, ls="--", lw=1.5); a2.axhline(0, color=MUTED, lw=0.8)
a2.set_ylim(-4, 8); a2.set_ylabel("mm")
a2.set_title("Median of all stations", loc="left")
a2.text(2006.3, 0.9, "13-week running mean;\nband: middle 50% of stations", transform=a2.get_xaxis_transform(), fontsize=15, color=MUTED, va="top")
a2.text(2013.1, 0.08, "drought", transform=a2.get_xaxis_transform(), fontsize=16, color=MUTED)
for a in (a1, a2):
    a.set_xticks([2006, 2008, 2010, 2012, 2014, 2016])
plt.setp(a1.get_xticklabels(), visible=False)
fig.text(0.01, 0.0, f"NGL tenv3 daily positions, IGS20 solution, retrieved {dt.date.today().isoformat()}",
         fontsize=13, color=MUTED)
fig.savefig(OUT / "drought_uplift_2014.png", dpi=110, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ---- numbers quoted on the slide ----
print(f"wrote {OUT}")
print(f"candidate PBO stations: {len(meta)}; used: {len(used)}; excluded:",
      pd.Series(why).value_counts().to_dict(), "| no March 2014 data:", len(res) - len(used))
print(f"March 2014 vertical change, all used stations: median {used['mar14'].median():.1f} mm, "
      f"IQR {used['mar14'].quantile(.25):.1f} to {used['mar14'].quantile(.75):.1f}, max {used['mar14'].max():.1f} mm "
      f"({used['mar14'].idxmax()})")
print(f"share of stations rising: {(used['mar14'] > 0).mean():.0%}")
print(f"Sierra box (36-40N, 121.2-117.8W): n={len(sierra)}, median {sierra['mar14'].median():.1f} mm, "
      f"max {sierra['mar14'].max():.1f} mm ({sierra['mar14'].idxmax()}); above 800 m: n={len(mountain)}, "
      f"median {mountain['mar14'].median():.1f} mm, max {mountain['mar14'].max():.1f} mm ({feat})")
print("13-week mean of the weekly median, 2006-2012 range:",
      f"{med.loc[2006:2012.99].min():.1f} to {med.loc[2006:2012.99].max():.1f} mm; peak after 2013:",
      f"{med.loc[2013:].max():.1f} mm at {med.loc[2013:].idxmax():.2f}")
if "P395" in used.index:
    print(f"P395 March 2014: {used.loc['P395', 'mar14']:.1f} mm")
else:
    print("P395:", why.get("P395", "no March data"))

# records for the slide table, from the featured station's file
tenv = fetch(f"{NGL}/gps_timeseries/IGS20/tenv3/IGS20/{feat}.tenv3", f"ngl_tenv3_{feat}.tenv3")
rec = pd.read_csv(tenv, sep=r"\s+")
pick = rec[rec["YYMMMDD"].isin(["13MAR14", "13MAR15", "14MAR14", "14MAR15"])]
for y in ("13", "14"):
    mm = rec[rec["YYMMMDD"].str.startswith(y + "MAR")]["____up(m)"]
    print(f"{feat} raw vertical, March 20{y}: median {mm.median():.4f} m over {len(mm)} days")
print(f"\nrecords from {feat}.tenv3:")
print(pick[["site", "YYMMMDD", "yyyy.yyyy", "u0(m)", "____up(m)", "_ant(m)", "sig_u(m)", "_latitude(deg)", "_longitude(deg)"]].to_string(index=False))
r13 = res[feat]; print(f"{feat} median residual, March 2013: {r13[(r13['t'] >= 2013 + 59/365) & (r13['t'] < 2013 + 90/365)]['res'].median():.1f} mm")
print(f"{feat} at", meta.loc[feat, ["lat", "lon", "h"]].round(3).to_dict(), "pre-drought vertical rate", f"{rate[feat]:.1f} mm/yr")
