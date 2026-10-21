"""Slide figure (lecture 9): one day of ground motion in Seattle, and which part of it is "noise".

UW.SP2 (Seward Park, Seattle, WA; broadband HHZ at 100 Hz, 1 m depth, coordinates from
the station service) recorded Thursday 2024-02-22, local time (PST = UTC-8). One local
day, 08:00 UTC to 08:00 UTC the next day, is cut into 5-minute windows; each window
gets a Welch power spectral density (PSD) of ground acceleration after division by the
instrument response from the StationXML.
  (a) spectrogram, PSD in dB relative to 1 (m/s^2)^2/Hz, local time on the x axis;
      the ocean microseism band (0.1-0.5 Hz), the cultural band (1-10 Hz), the
      earthquakes ComCat lists that day within 100 km (M >= 2.5) and anywhere
      at M >= 6 are marked
  (b) mean PSD in the cultural band (1-10 Hz) and in the microseism band
      (0.1-0.5 Hz), per 5-minute window
  (c) median daytime (08-18 PST) and nighttime (00-05 PST) PSD against the
      Peterson (1993) new low and high noise models (NLNM, NHNM)
  (d) station and epicenters (coastlines: Natural Earth 1:10m land)
Printed besides the figure: the 5-minute maximum over minimum of the cultural
band with and without the windows of the local earthquake, its 99th over 1st
percentile, and the excess kurtosis of 1-10 Hz ground velocity by day (10-16 h)
and at night (00:00-04:54, before the local earthquake).
Every value is fetched at run time: waveforms and response from the EarthScope FDSN
web services, earthquakes from the USGS ComCat event service.

Regenerate: pixi run python book/slides/2026/figscripts/lec09_day_spectrogram.py   (about 2 minutes)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pooch  # noqa: E402
from obspy import UTCDateTime  # noqa: E402
from obspy.clients.fdsn import Client  # noqa: E402
from obspy.geodetics import gps2dist_azimuth  # noqa: E402
from obspy.signal.spectral_estimation import get_nhnm, get_nlnm  # noqa: E402
from scipy.signal import welch  # noqa: E402
from scipy.stats import kurtosis  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED = "#26241f", "#6e675c"
CULT_C, MS_C, EQ_C = "#eb6834", "#2a78d6", "#26241f"     # dataviz reference slots 2, 1; ink
CACHE = pooch.os_cache("mlgeo-figscripts")

NET, STA, LOC, CHA = "UW", "SP2", "--", "HHZ"
UTC_OFFSET_H = -8                                    # PST on 2024-02-22
T0 = UTCDateTime("2024-02-22T08:00:00")              # local midnight
WIN, SEG = 300.0, 60.0                               # spectrogram window and Welch segment (s)
CULT, MS = (1.0, 10.0), (0.1, 0.5)
DAY_H, NIGHT_H = (8, 18), (0, 5)                     # local hours

# ---------------------------------------------------------------- data
fdsn = Client("EARTHSCOPE")
inv = fdsn.get_stations(network=NET, station=STA, channel=CHA, starttime=T0, endtime=T0 + 86400, level="response")
sta = inv[0][0]
st = fdsn.get_waveforms(NET, STA, LOC, CHA, T0, T0 + 86400)
n_traces = len(st)
st.merge(fill_value=None)
tr = st[0]
fs = tr.stats.sampling_rate
data = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(tr.data, dtype=float)), np.nan)

nseg = int(SEG * fs)
f = np.fft.rfftfreq(nseg, 1 / fs)
resp = inv.get_response(tr.id, T0).get_evalresp_response_for_frequencies(f[1:], output="VEL")
gain = np.r_[np.inf, np.abs(resp) ** 2]                # counts^2 per (m/s)^2
to_acc = (2 * np.pi * f) ** 2

nwin = int(86400 / WIN)
P = np.full((nwin, f.size), np.nan)
for i in range(nwin):
    i0 = int(round((T0 + i * WIN - tr.stats.starttime) * fs))
    x = data[max(i0, 0):i0 + int(WIN * fs)]
    if i0 < 0 or x.size < WIN * fs * 0.99 or np.isnan(x).any():
        continue
    _, p = welch(x - x.mean(), fs, nperseg=nseg, window="hann", detrend="linear")
    P[i] = p / gain * to_acc
hours = (np.arange(nwin) + 0.5) * WIN / 3600          # local hours since midnight
keep = (f >= 0.02) & (f <= 0.8 * fs / 2)
fk, Pk = f[keep], P[:, keep]
dB = 10 * np.log10(Pk)


def band_mean(lo, hi):
    m = (fk >= lo) & (fk <= hi)
    return np.nanmean(Pk[:, m], axis=1)


cult, ms = band_mean(*CULT), band_mean(*MS)
day = (hours >= DAY_H[0]) & (hours < DAY_H[1])
night = (hours >= NIGHT_H[0]) & (hours < NIGHT_H[1])
ratio_cult = np.nanmedian(cult[day]) / np.nanmedian(cult[night])
ratio_ms = np.nanmedian(ms[day]) / np.nanmedian(ms[night])
psd_day, psd_night = np.nanmedian(Pk[day], axis=0), np.nanmedian(Pk[night], axis=0)
psd_all = np.nanmedian(Pk, axis=0)
mb = (fk >= 0.05) & (fk <= 1.0)
ms_peak = fk[mb][np.argmax(psd_all[mb])]
ms_peak_db = 10 * np.log10(psd_all[mb].max())

# earthquakes within 100 km that day
cat = Client("USGS").get_events(starttime=T0, endtime=T0 + 86400, latitude=sta.latitude, longitude=sta.longitude,
                                maxradius=0.9, minmagnitude=2.5)
quakes = []
for ev in cat:
    o = ev.preferred_origin() or ev.origins[0]
    m = ev.preferred_magnitude() or ev.magnitudes[0]
    d_km = gps2dist_azimuth(sta.latitude, sta.longitude, o.latitude, o.longitude)[0] / 1e3
    if d_km <= 100:
        quakes.append(dict(t=o.time, h=(o.time - T0) / 3600, mag=m.mag, mtype=m.magnitude_type,
                           lat=o.latitude, lon=o.longitude, dep=o.depth / 1e3, d_km=d_km,
                           desc=ev.event_descriptions[0].text if ev.event_descriptions else ""))
for ev in Client("USGS").get_events(starttime=T0, endtime=T0 + 86400, minmagnitude=6.0):
    o = ev.preferred_origin() or ev.origins[0]
    m = ev.preferred_magnitude() or ev.magnitudes[0]
    d_km = gps2dist_azimuth(sta.latitude, sta.longitude, o.latitude, o.longitude)[0] / 1e3
    if d_km > 100:
        quakes.append(dict(t=o.time, h=(o.time - T0) / 3600, mag=m.mag, mtype=m.magnitude_type,
                           lat=o.latitude, lon=o.longitude, dep=o.depth / 1e3, d_km=d_km,
                           desc=ev.event_descriptions[0].text if ev.event_descriptions else ""))
quakes.sort(key=lambda q: q["t"])

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16, 7.2))
gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 1.1], height_ratios=[1.75, 1.25], hspace=0.42, wspace=0.45)
axs = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[1, 0], sharex=axs)
axp = fig.add_subplot(gs[0, 1])
axm = fig.add_subplot(gs[1, 1])

edges_t = np.arange(nwin + 1) * WIN / 3600
lf = np.log10(fk)
edges_f = 10 ** np.r_[lf[0] - (lf[1] - lf[0]) / 2, (lf[1:] + lf[:-1]) / 2, lf[-1] + (lf[-1] - lf[-2]) / 2]
vmin, vmax = -168, np.nanpercentile(dB, 99.5)
pc = axs.pcolormesh(edges_t, edges_f, dB.T, cmap="magma", vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
axs.set_yscale("log")
axs.set_ylim(fk[0], fk[-1])
axs.set_ylabel("frequency (Hz)")
for (lo, hi), c, lab in ((MS, MS_C, "microseism"), (CULT, CULT_C, "cultural")):
    axs.annotate("", xy=(24.25, lo), xytext=(24.25, hi), annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", color=c, lw=6))
    axs.text(24.6, np.sqrt(lo * hi), lab, color=INK, fontsize=15, va="center", ha="left", rotation=90,
             clip_on=False)
for q in quakes:
    lab = f"M{q['mag']:.1f}, {q['d_km']:.0f} km" if q["d_km"] < 1000 else f"M{q['mag']:.1f}, {q['d_km'] / 111.19:.0f}°"
    axs.annotate(lab, xy=(q["h"], fk[-1]), xytext=(q["h"], fk[-1] * 1.9),
                 ha="center", va="bottom", fontsize=14, color=INK, annotation_clip=False,
                 arrowprops=dict(arrowstyle="-|>", color=EQ_C, lw=1.8))
cb = fig.colorbar(pc, cax=axs.inset_axes([1.085, 0.0, 0.018, 1.0]))
cb.ax.set_title("dB", fontsize=14, loc="left")
axs.set_title(f"(a) {NET}.{STA} {CHA}, {(T0 + 1).strftime('%a %Y-%m-%d')}, vertical acceleration",
              loc="left", pad=62)
plt.setp(axs.get_xticklabels(), visible=False)

axb.axvspan(*DAY_H, color="#f4efe6", lw=0)
axb.axvspan(*NIGHT_H, color="#eef3f8", lw=0)
axb.plot(hours, 10 * np.log10(cult), color=CULT_C, lw=2, label=f"{CULT[0]:g}–{CULT[1]:g} Hz")
axb.plot(hours, 10 * np.log10(ms), color=MS_C, lw=2, label=f"{MS[0]:g}–{MS[1]:g} Hz")
axb.text(np.mean(DAY_H), axb.get_ylim()[1], "daytime", ha="center", va="top", color=MUTED, fontsize=14)
axb.text(np.mean(NIGHT_H), axb.get_ylim()[1], "night", ha="center", va="top", color=MUTED, fontsize=14)
axb.set_ylabel("dB")
axb.set_xlim(0, 24)
axb.set_xticks(range(0, 25, 3))
axb.set_xlabel("local time, PST (hours)")
axb.legend(frameon=False, loc="upper right", fontsize=14, ncol=1)
axb.set_title(f"(b) cultural band {10 * np.log10(ratio_cult):.1f} dB louder by day "
              f"({ratio_cult:.1f}×)", loc="left")

pf, nlnm = get_nlnm()
_, nhnm = get_nhnm()
axp.fill_between(1 / pf, nlnm, nhnm, color="#efebe4", lw=0)
axp.plot(1 / pf, nlnm, color=MUTED, lw=1.2)
axp.plot(1 / pf, nhnm, color=MUTED, lw=1.2)
i_lab = np.argmin(np.abs(1 / pf - 0.5))
i_low = np.argmin(np.abs(1 / pf - 0.3))
axp.text(0.3, nlnm[i_low] - 4, "NLNM", color=MUTED, fontsize=13, va="top", ha="center")
axp.text(0.5, nhnm[i_lab] + 3, "NHNM", color=MUTED, fontsize=13, va="bottom", ha="center")
axp.plot(fk, 10 * np.log10(psd_day), color=CULT_C, lw=2.2, label="08–18 h")
axp.plot(fk, 10 * np.log10(psd_night), color=MS_C, lw=2.2, label="00–05 h")
axp.axvline(ms_peak, color=INK, lw=1, ls=":")
axp.text(ms_peak * 1.1, -186, f"{ms_peak:.2f} Hz", fontsize=13, color=INK)
axp.set_xscale("log")
axp.set_xlim(fk[0], fk[-1])
axp.set_ylim(-190, -75)
axp.set_xlabel("frequency (Hz)")
axp.text(0, 1.03, "dB rel. 1 (m/s²)²/Hz", transform=axp.transAxes, fontsize=13, color=MUTED)
axp.legend(frameon=False, fontsize=13, loc="lower right", title="median PSD", title_fontsize=12,
           borderaxespad=0.2, handlelength=1.2)
axp.set_title("(c) day and night", loc="left", pad=62)

land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
                                    known_hash=None, fname="ne_10m_land.zip", path=CACHE, progressbar=False))
ext = (-123.4, -121.2, 46.9, 48.2)
axm.set_facecolor("#eef3f8")
land.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=axm, color="#f4efe6", edgecolor=MUTED, lw=0.6)
axm.plot(sta.longitude, sta.latitude, marker="^", ms=18, color=CULT_C, mec="white", mew=2, zorder=3)
axm.text(sta.longitude - 0.08, sta.latitude, STA, fontsize=14, fontweight="bold", ha="right", va="center")
for q in [q for q in quakes if q["d_km"] <= 100]:
    axm.plot(q["lon"], q["lat"], marker="*", ms=18, color=EQ_C, mec="white", mew=1, zorder=3)
axm.set_xlim(ext[:2]); axm.set_ylim(ext[2:])
axm.set_aspect(1 / np.cos(np.deg2rad(47.5)))
axm.set_xticks([]); axm.set_yticks([])
axm.set_xlabel(""); axm.set_ylabel("")
axm.set_title(f"(d) {sta.latitude:.3f}°N, {-sta.longitude:.3f}°W", fontsize=14, fontweight="normal",
              color=MUTED, loc="left")
for ax in (axb, axp):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

out = Path(__file__).resolve().parent.parent / "figs" / "lec09_noise" / "day_spectrogram_sp2.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"station {NET}.{STA} {sta.latitude:.5f} {sta.longitude:.5f} {sta.elevation:.0f} m '{sta.site.name}'; "
      f"channel {tr.id} {fs:g} Hz; {n_traces} trace(s) delivered, {tr.stats.starttime} to {tr.stats.endtime}")
print(f"local day {T0} to {T0 + 86400} UTC (PST = UTC{UTC_OFFSET_H:+d}); {nwin} windows of {WIN:.0f} s, "
      f"Welch {SEG:.0f} s Hann; {int(np.isnan(P[:, 1]).sum())} windows skipped for gaps")
print(f"cultural band {CULT} Hz: median day ({DAY_H} h) / night ({NIGHT_H} h) power ratio "
      f"{ratio_cult:.2f} = {10 * np.log10(ratio_cult):.1f} dB; "
      f"max/min over windows {np.nanmax(cult) / np.nanmin(cult):.1f}")
eq_win = np.zeros(nwin, bool)
for q in quakes:
    if q["d_km"] <= 100:
        eq_win |= np.abs(hours - q["h"]) < 10 / 60          # windows within 10 min of a local earthquake
print(f"cultural band max/min without the {int(eq_win.sum())} windows within 10 min of the local earthquake(s): "
      f"{np.nanmax(cult[~eq_win]) / np.nanmin(cult[~eq_win]):.1f}; 99th/1st percentile over all windows "
      f"{np.nanpercentile(cult, 99) / np.nanpercentile(cult, 1):.1f}; hour of the maximum {hours[np.nanargmax(cult)]:.2f} PST")
vel = tr.copy()
vel.detrend("demean")
vel.remove_response(inventory=inv, output="VEL", pre_filt=(0.03, 0.05, 40, 45))
vel.filter("bandpass", freqmin=CULT[0], freqmax=CULT[1], corners=4, zerophase=True)
for lab, (a, b) in {"day 10-16 h": (10, 16), "night 00:00-04:54": (0, 4.9)}.items():
    seg = vel.data[int(a * 3600 * fs):int(b * 3600 * fs)]
    print(f"1-10 Hz velocity, {lab}: excess kurtosis {kurtosis(seg):.1f}")
print(f"microseism band {MS} Hz: day/night ratio {ratio_ms:.2f} = {10 * np.log10(ratio_ms):.1f} dB")
print(f"microseism peak of the daily median PSD (0.05-1 Hz): {ms_peak:.3f} Hz (period {1 / ms_peak:.1f} s), "
      f"{ms_peak_db:.1f} dB; frequency resolution {fk[1] - fk[0]:.4f} Hz")
for q in quakes:
    print(f"earthquake {q['t']} UTC ({q['h']:.2f} h PST), {q['mtype']} {q['mag']:.2f}, {q['dep']:.1f} km deep, "
          f"{q['d_km']:.1f} km from {STA}, {q['desc']}")
