"""Slide figure (lecture 8): a river gauge through freeze-up, and how the provider repairs the record.

USGS 15515500, Tanana River at Nenana, Alaska (coordinates and drainage area
from the USGS site service). A stream gauge records stage (water level) every
15 minutes and converts it to discharge through a stage-discharge rating curve.
When the river freezes, ice raises the water level while the flow keeps
falling, so the rating no longer applies. USGS then withholds the computed
discharge and publishes an estimate twice a day, flagged "e". Through the
winter the stage record itself stops.
  (a) gauge height, 15-minute values, water year 2026
  (b) discharge: 15-minute values computed from stage (blue), the USGS
      estimates flagged "e" (orange), and what the open-water rating returns
      if applied to the ice-affected stage (gray)
  (c) the same records in the stage-discharge plane, against the rating
Every value is fetched from USGS at run time: instantaneous values (gauge
height 00065, discharge 00060) from the USGS Water Services, site metadata
from the site service, and the current rating table (exsa) from NWISweb.
Data for this water year are provisional.

Regenerate: pixi run python book/slides/2026/figscripts/lec08_real_river_ice.py
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
INK, MUTED, ICE = "#26241f", "#6e675c", "#e4eef7"
MEAS_C, EST_C = "#2a78d6", "#eb6834"           # dataviz reference slots 1 and 2
CFS_TO_M3S, FT_TO_M = 0.0283168, 0.3048
CACHE = pooch.os_cache("mlgeo-figscripts")
SITE, START, END = "15515500", "2025-10-01", "2026-06-30"


def rdb(text):
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)


url = (f"https://nwis.waterservices.usgs.gov/nwis/iv/?format=rdb&sites={SITE}&startDT={START}"
       f"&endDT={END}&parameterCd=00060,00065")
iv = rdb(requests.get(url, timeout=300).text)
iv["t"] = pd.to_datetime(iv["datetime"])                         # local time (tz_cd AKST/AKDT)
qcol = [c for c in iv.columns if c.endswith("_00060")][0]
gcol = [c for c in iv.columns if c.endswith("_00065")][0]
iv["q"] = pd.to_numeric(iv[qcol], errors="coerce") * CFS_TO_M3S
iv["gh"] = pd.to_numeric(iv[gcol], errors="coerce") * FT_TO_M
iv["qcd"] = iv[qcol + "_cd"].fillna("")
iv = iv.set_index("t")
est = iv[iv["qcd"].str.contains("e") & iv["q"].notna()]
meas = iv[~iv["qcd"].str.contains("e") & iv["q"].notna()]
withheld = iv[iv["q"].isna() & iv["gh"].notna()]                 # stage recorded, discharge not published
no_stage = iv[iv["gh"].isna()]

site = rdb(requests.get(f"https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={SITE}&siteOutput=expanded",
                        timeout=120).text).iloc[0]
lat, lon = float(site["dec_lat_va"]), float(site["dec_long_va"])
rating = rdb(requests.get(f"https://waterdata.usgs.gov/nwisweb/get_ratings?site_no={SITE}&file_type=exsa",
                          timeout=120).text)
r_gh = rating["INDEP"].astype(float).values * FT_TO_M
r_q = rating["DEP"].astype(float).values * CFS_TO_M3S

# open-water rating applied to the ice-affected stage, compared with the USGS estimate at the same time
ice_gh = withheld["gh"]
rated = pd.Series(np.interp(ice_gh.values, r_gh, r_q, left=np.nan, right=np.nan), index=ice_gh.index)
est_at = np.interp(ice_gh.index.values.astype("int64"), est.index.values.astype("int64"), est["q"].values)
ratio = rated.values / est_at
ice0, ice1 = est.index.min(), est.index.max()
gh_t = iv["gh"].dropna().index.to_series()
gh_jump = gh_t.diff()
gh_back = gh_jump.idxmax()                                       # first stage value after the longest stage gap
gh_last = gh_t[gh_t < gh_back].iloc[-1]

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12.5, 7.4))
gs = fig.add_gridspec(2, 2, width_ratios=[3.1, 1.15], hspace=0.36, wspace=0.22)
axg = fig.add_subplot(gs[0, 0])
axq = fig.add_subplot(gs[1, 0], sharex=axg)
axm = fig.add_subplot(gs[0, 1])
axr = fig.add_subplot(gs[1, 1])
for ax in (axg, axq):
    ax.axvspan(ice0, ice1, color=ICE, lw=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axg.plot(iv.index, iv["gh"], color=MEAS_C, lw=1.6)
axg.set_ylabel("gauge height (m)")
axg.set_title("(a) Stage jumps at freeze-up, then stops", loc="left")
mid = gh_last + (gh_back - gh_last) / 2
axg.text(mid, np.nanmean(axg.get_ylim()), f"no stage recorded\nfor {(gh_back - gh_last).days} days",
         ha="center", va="center", fontsize=16, color=INK)
axg.tick_params(labelbottom=False)

axq.plot(meas.index, meas["q"], ".", ms=2, color=MEAS_C, label="computed from stage, every 15 min")
axq.plot(rated.index, rated, ".", ms=1.5, color=MUTED, label="open-water rating applied to ice-affected stage")
axq.plot(est.index, est["q"], "o", ms=4, color=EST_C, label='USGS estimate, flagged "e", twice a day')
axq.set_ylabel("discharge (m³/s)")
axq.set_yscale("log")
axq.set_title(f"(b) Discharge estimated for {(ice1 - ice0).days} days", loc="left")
axq.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=14, markerscale=3)
axq.set_ylim(200, 6000)
axq.set_yticks([200, 500, 1000, 2000, 5000])
axq.set_yticklabels(["200", "500", "1000", "2000", "5000"])
axq.minorticks_off()
axq.xaxis.set_major_locator(mdates.MonthLocator())
axq.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axq.set_xlabel("water year 2026 (Oct 2025 – Jun 2026), Alaska local time")
axq.set_xlim(pd.Timestamp(START), pd.Timestamp(END))

# (c) stage-discharge plane
sel = (r_q > 0.5 * np.nanmin(iv["q"])) & (r_q < 2 * np.nanmax(iv["q"]))
axr.plot(r_q[sel], r_gh[sel], color=INK, lw=2, label="USGS rating")
axr.plot(meas["q"], meas["gh"], ".", ms=2, color=MEAS_C)
both = est[est["gh"].notna()]
axr.plot(both["q"], both["gh"], "o", ms=4, color=EST_C)
axr.set_xscale("log")
axr.set_xticks([300, 1000, 3000])
axr.set_xticklabels(["300", "1000", "3000"])
axr.minorticks_off()
axr.set_xlabel("discharge (m³/s)")
axr.set_ylabel("gauge height (m)")
axr.set_title("(c) Off the rating", loc="left", fontsize=17)
axr.legend(frameon=False, loc="upper left", fontsize=14)
for sp in ("top", "right"):
    axr.spines[sp].set_visible(False)

land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
                                    known_hash=None, fname="ne_10m_land.zip", path=CACHE, progressbar=False))
ext = (-169.0, -137.0, 54.5, 71.8)
axm.set_facecolor("#eef3f8")
land.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=axm, color="#f4efe6", edgecolor=MUTED, lw=0.6)
axm.plot(lon, lat, marker="o", ms=15, color=MEAS_C, mec="white", mew=2)
axm.text(lon, lat - 1.0, f"USGS {SITE}\nTanana River\nat Nenana", ha="center", va="top", fontsize=14,
         fontweight="bold")
axm.set_xlim(ext[:2]); axm.set_ylim(ext[2:])
axm.set_aspect(1 / np.cos(np.deg2rad(63)))
axm.set_xticks([]); axm.set_yticks([])
axm.set_xlabel(""); axm.set_ylabel("")
axm.set_title(f"{lat:.3f}°N, {-lon:.3f}°W\nbasin {float(site['drain_area_va']) * 2.58999:,.0f} km²",
              fontsize=14, fontweight="normal", color=MUTED)

out = Path(__file__).resolve().parent.parent / "figs" / "lec08_real" / "river_ice_tanana.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"site {SITE} {site['station_nm']} {lat} {lon}, drainage {site['drain_area_va']} mi2")
print(f"{len(iv)} 15-min records {iv.index[0]} to {iv.index[-1]}; discharge codes "
      f"{iv['qcd'].replace('', 'blank').value_counts().to_dict()}")
print(f"estimates ('e') {len(est)} from {ice0} to {ice1} ({(ice1 - ice0).days} days); "
      f"stage present but discharge withheld: {len(withheld)} records; no stage: {len(no_stage)} records")
print(f"stage gap: last value {gh_last}, back {gh_back}")
print(f"stage before freeze-up min {iv['gh'][:ice0 + pd.Timedelta(days=30)].min():.2f} m, "
      f"max while discharge withheld {ice_gh.max():.2f} m")
print(f"USGS estimates {est['q'].min():.0f}-{est['q'].max():.0f} m3/s; open-water rating on ice stage "
      f"{np.nanmin(rated):.0f}-{np.nanmax(rated):.0f} m3/s; ratio median {np.nanmedian(ratio):.2f}, "
      f"max {np.nanmax(ratio):.2f}")
print(f"rating table {len(rating)} rows, {r_gh.min():.2f}-{r_gh.max():.2f} m")
