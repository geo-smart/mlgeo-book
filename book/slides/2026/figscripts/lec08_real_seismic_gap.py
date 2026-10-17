"""Slide figure (lecture 8): a genuine telemetry gap at UW.RATT and what a careless merge does to it.

UW.RATT (Rattlesnake Lake, King County, WA; coordinates from the station service) is
the station of book section 2.9. On 2021-01-21 its 100 Hz vertical channel HHZ
lost 40 s of data in transmission. The gap is real; nobody planted it.
  (a) daily largest gap at RATT HHZ through 2021 (EarthScope MUSTANG max_gap),
      with the day we zoom into marked
  (b) the raw record around the gap, in digitizer counts, as two segments
      (gray), and the same record after merge(fill_value=0) (orange): the gap
      becomes a drop from the sensor's DC level to zero
  (c, d) band-pass 1-10 Hz (the band of section 2.9) and 0.1-1 Hz (the ocean
      microseism band) of the zero-filled record
      (orange) and of each segment on its own with the gap left missing (blue);
      the guard interval where the two disagree is shaded
Every value is fetched from its provider at run time: waveforms and station
metadata from the EarthScope FDSN web services, daily gap metrics from the
EarthScope MUSTANG measurements service.

Regenerate: pixi run python book/slides/2026/figscripts/lec08_real_seismic_gap.py
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
from obspy import UTCDateTime  # noqa: E402
from obspy.clients.fdsn import Client  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED, GUARD = "#26241f", "#6e675c", "#eee6d8"
SEG_C, ZERO_C = "#2a78d6", "#eb6834"           # dataviz reference slots 1 and 2
CACHE = pooch.os_cache("mlgeo-figscripts")

NET, STA, LOC, CHA = "UW", "RATT", "", "HHZ"
DAY = UTCDateTime("2021-01-21")
YEAR = ("2021-01-01", "2022-01-01")
BANDS = ((1.0, 10.0),)                          # the band of section 2.9
HALF = 60.0                                     # seconds shown on each side of the gap
PAD = 600.0                                     # seconds processed on each side (keeps window-edge effects off screen)

# ---------------------------------------------------------------- MUSTANG daily gap metrics


def mustang(metric):
    url = (f"https://service.iris.edu/mustang/measurements/1/query?metric={metric}&net={NET}&sta={STA}"
           f"&loc={LOC or '--'}&cha={CHA}&timewindow={YEAR[0]},{YEAR[1]}&format=text")
    lines = requests.get(url, timeout=300).text.splitlines()
    df = pd.read_csv(io.StringIO("\n".join(lines[1:])))
    df["day"] = pd.to_datetime(df["start"], format="%Y/%m/%d %H:%M:%S")
    return df.set_index("day")["value"]


max_gap = mustang("max_gap")
num_gaps = mustang("num_gaps")

# ---------------------------------------------------------------- waveforms and station
client = Client("EARTHSCOPE")
inv = client.get_stations(network=NET, station=STA, channel=CHA, level="channel", starttime=DAY)
sta = inv[0][0]
st_day = client.get_waveforms(NET, STA, LOC or "--", CHA, DAY, DAY + 86400)
gaps = [g for g in st_day.get_gaps() if g[6] > 1.0]
g = max(gaps, key=lambda x: x[6])
g0, g1, gdur = g[4], g[5], g[6]
st = st_day.slice(g0 - PAD, g1 + PAD).copy()
fs = st[0].stats.sampling_rate
dc = float(np.mean(np.concatenate([tr.data for tr in st])))

zero_raw = st.copy().merge(fill_value=0)[0]      # careless merge: zero inside the gap
tz = zero_raw.times() + (zero_raw.stats.starttime - g0)
dist = np.where(tz < 0, -tz, tz - gdur)              # seconds from the nearest gap edge


def rel(tr):
    return tr.times() + (tr.stats.starttime - g0)


def repair_test(band):
    """Filter across the zero-filled gap, and segment by segment with the gap left missing."""
    zero = zero_raw.copy()
    zero.detrend("demean")
    zero.filter("bandpass", freqmin=band[0], freqmax=band[1], corners=4, zerophase=True)
    segs = []
    for tr in st.copy():
        tr.detrend("demean")
        tr.taper(max_percentage=None, max_length=1.0)
        tr.filter("bandpass", freqmin=band[0], freqmax=band[1], corners=4, zerophase=True)
        segs.append(tr)
    seg = np.full(zero.stats.npts, np.nan)
    for tr in segs:
        i0 = int(round((tr.stats.starttime - zero.stats.starttime) * fs))
        seg[i0:i0 + tr.stats.npts] = tr.data
    outside = ~np.isnan(seg)
    local_rms = float(np.sqrt(np.mean(seg[outside & (dist > 60) & (dist < 300)] ** 2)))
    diff = np.abs(zero.data - seg)
    bad = outside & (diff > local_rms) & (dist < 120)
    guard = (float(-tz[bad & (tz < 0)].min()) if (bad & (tz < 0)).any() else 0.0,
             float(tz[bad & (tz > gdur)].max() - gdur) if (bad & (tz > gdur)).any() else 0.0)
    return dict(band=band, zero=zero, segs=segs, local_rms=local_rms, guard=guard,
                edge_peak=float(np.nanmax(diff[outside & (dist < 5)])),
                edge_rms=float(np.sqrt(np.mean(diff[outside & (dist < 2)] ** 2))),   # same metric as the planted gap
                inside_peak=float(np.abs(zero.data[(tz >= 0) & (tz <= gdur)]).max()))


tests = [repair_test(b) for b in BANDS]

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12.5, 7.4))
gs = fig.add_gridspec(3, 2, width_ratios=[3.6, 1], height_ratios=[0.75, 1, 1.15], hspace=0.62, wspace=0.12)
axy = fig.add_subplot(gs[0, 0])
axr = fig.add_subplot(gs[1, 0])
axf = [fig.add_subplot(gs[2 + i, 0], sharex=axr) for i in range(len(tests))]
axm = fig.add_subplot(gs[0:2, 1])
axz = fig.add_subplot(gs[2, 1])

# (a) MUSTANG
axy.bar(max_gap.index, max_gap.clip(lower=0.01), width=1.0, color=MUTED)
axy.set_yscale("log")
axy.set_ylim(0.3, 1e4)
axy.set_yticks([1, 10, 100, 1000])
axy.set_yticklabels(["1", "10", "100", "1000"])
axy.set_ylabel("largest\ngap (s)", fontsize=15)
axy.plot([pd.Timestamp(DAY.datetime)], [gdur * 4], marker="v", ms=13, color=ZERO_C)
axy.xaxis.set_major_locator(mdates.MonthLocator())
axy.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axy.set_xlim(pd.Timestamp(YEAR[0]), pd.Timestamp(YEAR[1]))
n_gap_days = int((num_gaps > 0).sum())
axy.set_title(f"(a) {NET}.{STA} {CHA} in 2021: {n_gap_days} of {len(num_gaps)} archived days have gaps",
              loc="left")

# (b) raw counts
axr.axvspan(0, gdur, color=GUARD, lw=0)
axr.plot(tz, zero_raw.data / 1e3, color=ZERO_C, lw=1.4, label="merged with zeros")
for i, s in enumerate(st):
    axr.plot(rel(s), s.data / 1e3, color=INK, lw=0.8, label="as delivered" if i == 0 else None)
axr.text(gdur / 2, 0.22, f"{gdur:.0f} s missing", ha="center", va="center", fontsize=15, color=INK)
axr.set_ylabel("raw\n(10³ counts)")
axr.set_ylim(-1.0, 0.45)
axr.set_title(f"(b) Raw record around the gap, {g0.strftime('%Y-%m-%d')}", loc="left")
axr.legend(frameon=False, loc="lower left", fontsize=14, ncol=2)

# (c, d) filtered
for ax, t, tag in zip(axf, tests, "cd"):
    gb, ga = t["guard"]
    ax.axvspan(-gb, 0, color=GUARD, lw=0)
    ax.axvspan(gdur, gdur + ga, color=GUARD, lw=0)
    ax.plot(tz, t["zero"].data, color=ZERO_C, lw=1.2, label="zero-filled, then filtered")
    for i, s in enumerate(t["segs"]):
        ax.plot(rel(s), s.data, color=SEG_C, lw=1.0,
                label="each segment filtered, gap left missing" if i == 0 else None)
    lim = 1.25 * max(np.abs(t["zero"].data[np.abs(tz - gdur / 2) < HALF + gdur]).max(),
                     np.abs(np.concatenate([s.data for s in t["segs"]])).max())
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("counts")
    ax.set_title(f"({tag}) {t['band'][0]:g}–{t['band'][1]:g} Hz: edge error "
                 f"{t['edge_peak'] / t['local_rms']:,.0f}× the local signal, for {max(gb, ga):.0f} s",
                 loc="left")
axf[0].legend(frameon=False, loc="lower left", fontsize=14, ncol=2)
t = tests[0]
zoom = np.abs(tz) < 2.5
axz.plot(tz[zoom], t["zero"].data[zoom], color=ZERO_C, lw=1.6)
for s_ in t["segs"]:
    m = np.abs(rel(s_)) < 2.5
    axz.plot(rel(s_)[m], s_.data[m], color=SEG_C, lw=1.6)
axz.axvspan(0, 2.5, color=GUARD, lw=0)
axz.set_xlim(-2.5, 2.5); axz.set_xticks([-2, 0, 2]); axz.set_yticklabels([])
axz.set_xlabel("seconds")
axz.set_title("zoom, first edge", fontsize=15, loc="right", pad=14)
for sp in ("top", "right"):
    axz.spines[sp].set_visible(False)
axf[-1].set_xlabel(f"seconds after the start of the gap ({g0.strftime('%H:%M:%S')} UTC)")
axr.set_xlim(-HALF, gdur + HALF)
for ax in [axy, axr, *axf]:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# location
land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
                                    known_hash=None, fname="ne_10m_land.zip", path=CACHE, progressbar=False))
states = gpd.read_file(pooch.retrieve(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces_lines.zip",
    known_hash=None, fname="ne_10m_admin_1_lines.zip", path=CACHE, progressbar=False))
ext = (-125.5, -116.0, 41.5, 49.5)
axm.set_facecolor("#eef3f8")
land.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=axm, color="#f4efe6", edgecolor=MUTED, lw=0.6)
states.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=axm, color=MUTED, lw=0.6, linestyle="--")
axm.plot(sta.longitude, sta.latitude, marker="^", ms=18, color=SEG_C, mec="white", mew=2)
axm.text(sta.longitude, sta.latitude - 0.55, f"{NET}.{STA}", ha="center", va="top", fontsize=15, fontweight="bold")
axm.set_xlim(ext[:2]); axm.set_ylim(ext[2:])
axm.set_aspect(1 / np.cos(np.deg2rad(45.5)))
axm.set_xticks([]); axm.set_yticks([])
axm.set_xlabel(""); axm.set_ylabel("")
axm.set_title(f"{sta.latitude:.3f}°N, {-sta.longitude:.3f}°W", fontsize=14, fontweight="normal", color=MUTED)

out = Path(__file__).resolve().parent.parent / "figs" / "lec08_real" / "seismic_gap_ratt.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"station {NET}.{STA} {sta.latitude:.4f} {sta.longitude:.4f} {sta.elevation:.0f} m '{sta.site.name}'")
print(f"MUSTANG 2021: {len(num_gaps)} days, {n_gap_days} with >=1 gap, {int((max_gap > 10).sum())} with a gap > 10 s, "
      f"total gaps {int(num_gaps.sum())}; max_gap on {DAY.date} = {max_gap.get(pd.Timestamp(DAY.datetime), np.nan)} s")
print(f"gaps > 1 s on {DAY.date}: {[(str(x[4])[:19], round(x[6], 2)) for x in gaps]}")
print(f"chosen gap {g0} to {g1}, {gdur:.2f} s; segments {[(str(tr.stats.starttime)[:22], tr.stats.npts) for tr in st]}")
print(f"mean raw level {dc:.0f} counts in the window (zero fill steps by that much at both edges)")
for t in tests:
    print(f"band {t['band']} Hz, 4 poles, zero-phase: local in-band RMS {t['local_rms']:.1f} counts (60-300 s from gap); "
          f"peak edge error {t['edge_peak']:.0f} counts = {t['edge_peak'] / t['local_rms']:.1f}x; RMS edge error "
          f"0-2 s from the edges {t['edge_rms']:.1f} counts = {t['edge_rms'] / t['local_rms']:.1f}x (graded against "
          f"the segment-wise result: no intact record exists); peak filtered value "
          f"inside the gap {t['inside_peak']:.0f} counts = {t['inside_peak'] / t['local_rms']:.1f}x; guard interval "
          f"(error > local RMS) {t['guard'][0]:.1f} s before, {t['guard'][1]:.1f} s after")
