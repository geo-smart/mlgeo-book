"""Slide figure (lecture 8): a real clock error, flagged by the station and measured by cross-correlation.

US.HDIL (Hopedale, Illinois; USGS/EarthScope broadband station, coordinates from
the station service) lost GPS timing on 2022-10-03 and regained it on
2023-03-07, according to the daily timing-quality metric the archive computes
from the station's own MiniSEED headers (last day without lock: 2023-03-06).
While the clock ran without GPS, data arrived intermittently in December 2022
and January 2023, and the clock error grew from near 0 to about +118 s, +230 s
and +270 s across those interruptions; it eased to about +267 s by early March.
The archived waveforms still carry that error.
  (a) EarthScope MUSTANG daily timing_quality (%) and percent_availability (%)
      for US.HDIL.00.LHZ
  (b) daily ambient-noise cross-correlation HDIL x N4.N41A (133 km west,
      0.05-0.2 Hz, spectrally whitened, 1-hour windows stacked per day); a
      clock error moves the whole correlation, both sides, by the same lag.
      Days with correlation 0.3 to 0.5 can lock onto the wrong side of the
      reference (about 2 x the 35-38 s surface-wave lag between the stations),
      which gives the -69 to -78 s values of mid-March 2023 and the +200 s
      values of some February days; the printed weekly table lists them.
  (c) the M7.8 Kahramanmaras, Turkiye, earthquake of 2023-02-06 at N4.N41A,
      N4.P43A and US.HDIL, each aligned on its own predicted P time (iasp91)
Every value is fetched at run time: waveforms and station metadata from the
EarthScope FDSN web services, daily metrics from EarthScope MUSTANG, the event
from USGS ComCat. The lags are measured here by cross-correlation.

Regenerate: pixi run python book/slides/2026/figscripts/lec08_real_clock_error.py   (about 5 minutes)
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
from obspy.geodetics import gps2dist_azimuth, locations2degrees  # noqa: E402
from obspy.taup import TauPyModel  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED = "#26241f", "#6e675c"
HDIL_C, REF_C, CTRL_C = "#eb6834", "#2a78d6", "#1baf7a"      # dataviz reference slots 2, 1, 3
CACHE = pooch.os_cache("mlgeo-figscripts")
T0, T1 = UTCDateTime("2022-09-15"), UTCDateTime("2023-04-01")
REF_END = UTCDateTime("2022-12-10")             # reference stack: days before the December outage
MAXLAG, BAND = 600, (0.05, 0.2)
EVENT = "us6000jllz"                            # USGS ComCat id, M7.8 Kahramanmaras 2023-02-06
STATIONS = {"HDIL": ("US", "00"), "N41A": ("N4", "00"), "P43A": ("N4", "00")}
es = Client("EARTHSCOPE")

# ---------------------------------------------------------------- station metadata and MUSTANG
coords = {}
for sta, (net, loc) in STATIONS.items():
    s = es.get_stations(network=net, station=sta, location=loc, channel="LHZ", starttime=T0, level="station")[0][0]
    coords[sta] = (s.latitude, s.longitude, s.site.name)
d_ref_km = gps2dist_azimuth(*coords["HDIL"][:2], *coords["N41A"][:2])[0] / 1e3


def mustang(metric):
    url = (f"https://service.iris.edu/mustang/measurements/1/query?metric={metric}&net=US&sta=HDIL&loc=00"
           f"&cha=LHZ&timewindow={T0.date},{T1.date}&format=text")
    lines = requests.get(url, timeout=300).text.splitlines()
    df = pd.read_csv(io.StringIO("\n".join(lines[1:])))
    return df.set_index(pd.to_datetime(df["start"], format="%Y/%m/%d %H:%M:%S"))["value"]


tq, avail = mustang("timing_quality"), mustang("percent_availability")
low = tq[tq < 50]
low0, low1 = low.index.min(), low.index.max() + pd.Timedelta(days=1)

# ---------------------------------------------------------------- daily noise correlations
ndays = int((T1 - T0) / 86400)


def daily_matrix(net, sta, loc):
    out = np.zeros((ndays, 86400))
    t = T0
    while t < T1:
        try:
            st = es.get_waveforms(net, sta, loc, "LHZ", t, min(t + 10 * 86400, T1))
        except Exception:                                        # no data in this chunk
            st = []
        for tr in st:
            i0 = int(round(tr.stats.starttime - T0))
            idx = np.arange(tr.stats.npts) + i0
            ok = (idx >= 0) & (idx < ndays * 86400)
            out.reshape(-1)[idx[ok]] = tr.data[ok]
        t += 10 * 86400
    return out


def day_ncf(a, b):
    acc, n = np.zeros(2 * MAXLAG + 1), 0
    for x, y in zip(a.reshape(24, -1), b.reshape(24, -1)):
        if (x == 0).mean() > 0.2 or (y == 0).mean() > 0.2:      # skip hours with missing data
            continue
        nf = 2 * len(x)
        X, Y = np.fft.rfft(x - x.mean(), nf), np.fft.rfft(y - y.mean(), nf)
        f = np.fft.rfftfreq(nf, 1.0)
        w = (f > BAND[0]) & (f < BAND[1])
        X = np.where(w, X / (np.abs(X) + 1e-12), 0)
        Y = np.where(w, Y / (np.abs(Y) + 1e-12), 0)
        cc = np.fft.irfft(X * np.conj(Y), nf)
        acc += np.concatenate([cc[-MAXLAG:], cc[:MAXLAG + 1]])
        n += 1
    return acc / n if n >= 12 else np.full(2 * MAXLAG + 1, np.nan)


H = daily_matrix("US", "HDIL", "00")
R = daily_matrix("N4", "N41A", "00")
C = np.array([day_ncf(H[i], R[i]) for i in range(ndays)])
days = pd.date_range(T0.datetime, periods=ndays, freq="D")
lags = np.arange(-MAXLAG, MAXLAG + 1)
ref_rows = (days < pd.Timestamp(REF_END.datetime)) & ~np.isnan(C).any(1)
ref = C[ref_rows].mean(0)
shift, coef = np.full(ndays, np.nan), np.full(ndays, np.nan)
for i in range(ndays):
    if np.isnan(C[i]).any():
        continue
    x = np.correlate(np.r_[np.zeros(MAXLAG), C[i], np.zeros(MAXLAG)], ref, "valid")
    x /= np.linalg.norm(C[i]) * np.linalg.norm(ref)
    k = int(np.argmax(x))
    sub = 0.0
    if 0 < k < len(x) - 1:                                       # parabola through the peak
        sub = 0.5 * (x[k - 1] - x[k + 1]) / (x[k - 1] - 2 * x[k] + x[k + 1])
    shift[i], coef[i] = k - MAXLAG + sub, x[k]

# ---------------------------------------------------------------- the earthquake
ev = Client("USGS").get_events(eventid=EVENT)[0]
org, mag = ev.preferred_origin(), ev.preferred_magnitude()
model = TauPyModel("iasp91")
quake = {}
for sta, (net, loc) in STATIONS.items():
    dist = locations2degrees(org.latitude, org.longitude, *coords[sta][:2])
    tp = org.time + model.get_travel_times(org.depth / 1e3, dist, ["P"])[0].time
    tr = es.get_waveforms(net, sta, loc, "LHZ", tp - 900, tp + 1500).merge(fill_value=0)[0]
    tr.detrend("demean")
    tr.taper(0.05)
    tr.filter("bandpass", freqmin=0.05, freqmax=0.3, zerophase=True)
    quake[sta] = (tr, tp, dist)


def quake_lag(sta, ref="N41A"):
    a, tpa, _ = quake[ref]
    b, tpb, _ = quake[sta]
    xa = a.slice(tpa - 20, tpa + 100).data.astype(float)
    xb = b.slice(tpb - 20 - 400, tpb + 100 + 400).data.astype(float)
    xa -= xa.mean()
    xb -= xb.mean()
    x = np.correlate(xb, xa, "valid") / np.sqrt(np.convolve(xb ** 2, np.ones(len(xa)), "valid") * (xa ** 2).sum())
    k = int(np.argmax(x))
    sub = 0.5 * (x[k - 1] - x[k + 1]) / (x[k - 1] - 2 * x[k] + x[k + 1])
    return (k - 400 + sub) / a.stats.sampling_rate, x[k]


q_lag = {s: quake_lag(s) for s in ("HDIL", "P43A")}

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12.5, 7.8))
gs = fig.add_gridspec(3, 2, width_ratios=[3.3, 1], height_ratios=[0.7, 1.25, 1.15], hspace=0.6, wspace=0.14)
axq = fig.add_subplot(gs[0, 0])
axc = fig.add_subplot(gs[1, 0], sharex=axq)
axw = fig.add_subplot(gs[2, 0])
axm = fig.add_subplot(gs[0:2, 1])

axq.bar(avail.index, avail.values, width=1.0, color="#e2ddd3")
axq.plot(tq.index, tq.values, color=HDIL_C, lw=2.5)
axq.text(pd.Timestamp("2022-10-20"), 14, "timing quality", color=INK, fontsize=15, va="bottom")
axq.text(pd.Timestamp("2022-10-20"), 60, "data received", color=MUTED, fontsize=15, va="center")
axq.set_ylim(0, 105)
axq.set_yticks([0, 50, 100])
axq.set_ylabel("%")
last_bad = low1 - pd.Timedelta(days=1)
axq.set_title(f"(a) US.HDIL timing quality: GPS lost {low0:%Y-%m-%d} to {last_bad:%Y-%m-%d}",
              loc="left")
axq.tick_params(labelbottom=False)

axc.set_facecolor("#e2ddd3")                                      # days without a correlation
M = C / np.nanmax(np.abs(C), axis=1, keepdims=True)
pcm = axc.pcolormesh(days, lags, M.T, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
weak = (coef > 0.3) & (coef <= 0.5)
axc.plot(days[weak], shift[weak], "o", ms=4, mfc="none", mec=INK, mew=0.8)
axc.plot(days[coef > 0.5], shift[coef > 0.5], "o", ms=5, color=HDIL_C, mec="white", mew=0.5)
axc.set_ylim(-450, 450)
axc.set_ylabel("lag (s)")
strong_err = shift[(coef > 0.5) & (shift > 100)]
lo_err, hi_err = 10 * np.round(np.nanmin(strong_err) / 10), 10 * np.round(np.nanmax(strong_err) / 10)
axc.set_title(f"(b) Noise correlation with N41A: {lo_err:+.0f} to {hi_err:+.0f} s", loc="left")
cax = axc.inset_axes([0.02, 0.80, 0.16, 0.07])                    # empty corner: Oct-Nov, lags > 300 s
cb = fig.colorbar(pcm, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
cax.xaxis.tick_top()
cb.ax.tick_params(labelsize=11, pad=1)
cax.text(1.06, 0.5, "normalized correlation", transform=cax.transAxes, fontsize=12, va="center", color=INK)
axc.xaxis.set_major_locator(mdates.MonthLocator())
axc.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

off = 0
for sta, col in (("N41A", REF_C), ("P43A", CTRL_C), ("HDIL", HDIL_C)):
    tr, tp, dist = quake[sta]
    tr = tr.slice(tp - 120, tp + 480)
    t = tr.times() + (tr.stats.starttime - tp)
    axw.plot(t, tr.data / np.abs(tr.data).max() * 0.45 + off, color=col, lw=1.2)
    lbl = f"{STATIONS[sta][0]}.{sta}" + ("" if sta == "N41A" else f"  {q_lag[sta][0]:+.1f} s")
    axw.text(482, off, lbl, fontsize=15, va="center", color=INK)
    off -= 1
axw.axvline(0, color=INK, lw=1)
axw.set_yticks([])
axw.set_xlim(-120, 480)
axw.set_xlabel("seconds after the predicted P arrival at each station")
axw.set_title(f"(c) M{mag.mag:.1f} Kahramanmaraş, {org.time.strftime('%Y-%m-%d')}, "
              f"{quake['HDIL'][2]:.0f}°: P {q_lag['HDIL'][0]:.0f} s late at HDIL only", loc="left")
for ax in (axq, axc, axw):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axw.spines["left"].set_visible(False)

land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
                                    known_hash=None, fname="ne_10m_land.zip", path=CACHE, progressbar=False))
states = gpd.read_file(pooch.retrieve(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces_lines.zip",
    known_hash=None, fname="ne_10m_admin_1_lines.zip", path=CACHE, progressbar=False))
ext = (-93.0, -86.0, 37.0, 43.0)
axm.set_facecolor("#eef3f8")
land.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=axm, color="#f4efe6", edgecolor=MUTED, lw=0.6)
states.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=axm, color=MUTED, lw=0.8, linestyle="--")
for sta, col in (("N41A", REF_C), ("P43A", CTRL_C), ("HDIL", HDIL_C)):
    la, lo, _ = coords[sta]
    axm.plot(lo, la, marker="^", ms=17, color=col, mec="white", mew=2)
    west = sta == "N41A"                                          # N41A label to the west, clear of HDIL
    axm.text(lo - 0.3 if west else lo + 0.3, la, sta, ha="right" if west else "left", va="center",
             fontsize=14, fontweight="bold")
axm.text(-89.2, 37.6, "Illinois", fontsize=14, color=MUTED, style="italic", ha="center")
axm.set_xlim(ext[:2]); axm.set_ylim(ext[2:])
axm.set_aspect(1 / np.cos(np.deg2rad(40)))
axm.set_xticks([]); axm.set_yticks([])
axm.set_xlabel(""); axm.set_ylabel("")
la, lo, name = coords["HDIL"]
axm.set_title(f"HDIL {la:.3f}°N, {-lo:.3f}°W", fontsize=14, fontweight="normal", color=MUTED)

out = Path(__file__).resolve().parent.parent / "figs" / "lec08_real" / "clock_error_hdil.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
for sta in STATIONS:
    print(f"{STATIONS[sta][0]}.{sta} {coords[sta][0]:.4f} {coords[sta][1]:.4f} '{coords[sta][2]}'")
print(f"HDIL-N41A distance {d_ref_km:.1f} km")
print(f"MUSTANG timing_quality < 50 from {low0.date()} to {last_bad.date()} "
      f"({len(low)} days, median {low.median():.0f} %); median before {tq[:low0].iloc[:-1].median():.1f} %, "
      f"after {tq[low1:].median():.1f} %")
runs = (low != low.shift()).cumsum()
print("timing_quality while unlocked, runs of equal value: " + "; ".join(
    f"{g.iloc[0]:.0f} % {g.index.min().date()} to {g.index.max().date()} ({len(g)} d)" for _, g in low.groupby(runs)))
full = avail.reindex(low.index)
print(f"days without lock but 100 % availability: {len(full[full >= 100])} of {full.notna().sum()}; "
      f"availability in December-January: " + ", ".join(
          f"{d_:%d %b} {v:.0f}" for d_, v in avail["2022-12-01":"2023-01-31"].items() if v < 100 or d_.day in (10, 11, 16)))
print(f"days with < 50 % availability in the window: {int((avail < 50).sum())}; "
      f"daily noise correlations computed: {int((~np.isnan(C).any(1)).sum())} of {ndays}; reference: "
      f"{int(ref_rows.sum())} days before {REF_END.date}")
print("weekly table of daily lags (s) and correlations, days with correlation > 0.2 (week starting Monday):")
wk = pd.DataFrame({"lag": shift, "cc": coef}, index=days).dropna()
wk = wk[(wk["cc"] > 0.2) & (wk.index >= pd.Timestamp("2022-12-01"))]
for k, v in wk.groupby(wk.index.to_period("W-SUN").start_time):
    print(f"  {k.date()}: " + ", ".join(f"{d_:%d %b} {r.lag:+.1f} ({r.cc:.2f})" for d_, r in v.iterrows()))
good = coef > 0.5
seg = pd.Series(shift[good], index=days[good])
print("daily lag (s, corr > 0.5) by month: " + "; ".join(
    f"{k:%Y-%m}: n={len(v)} median {v.median():+.1f}" for k, v in seg.groupby(seg.index.to_period('M').to_timestamp())))
big = seg[seg > 100]
if len(big):
    print(f"lag > 100 s from {big.index.min().date()} to {big.index.max().date()} ({len(big)} days), "
          f"range {big.min():+.1f} to {big.max():+.1f} s; January {big[:'2023-01-20'].median():+.1f} s, "
          f"after 20 January {big['2023-01-20':].median():+.1f} s")
print(f"event {EVENT} M{mag.mag} {mag.magnitude_type} {org.time} depth {org.depth / 1e3:.0f} km; "
      + "; ".join(f"{s} dist {quake[s][2]:.1f} deg" for s in STATIONS))
print("P-wave lag relative to N41A after removing predicted travel-time difference: "
      + "; ".join(f"{s} {q_lag[s][0]:+.2f} s (cc {q_lag[s][1]:.2f})" for s in q_lag))
