"""Slide figure (lecture 9): two hours of "noise" at UW.RATT, measured against the global noise models.

UW.RATT (Rattlesnake Lake, King County, WA), channel HHZ, 2021-07-29 04:15 to
06:15 UTC (21:15 to 23:15 PDT on July 28), the two hours that end 49 s before
the Mww 8.2 Chignik, Alaska, earthquake. Course copies of the raw counts and of
the StationXML (book/Chapter2-DataManipulation/snapshots/, see
ratt_chignik_2021-07-29.provenance.json) are read; nothing is fetched.
  (a) the record as ground velocity (response removed, 0.05-20 Hz pre-filter)
  (b) acceleration power spectral density, Welch, 327.68 s Hann segments with
      50 % overlap over the two hours (bold), and the 5th-95th percentile of
      the 12 PSDs of the 10-minute windows (shaded), against the Peterson (1993)
      new low and high noise models (NLNM, NHNM, from obspy); the maxima of the
      two-hour PSD in the primary microseism band (0.04-0.1 Hz) and in the
      long-period (0.1-0.18 Hz) and short-period (0.18-0.5 Hz) parts of the
      secondary microseism band are marked
  (c) station location
  (d) distribution of the 1-10 Hz band-passed velocity samples against a
      Gaussian with the same standard deviation, on a log count axis
ComCat lists one M >= 5 earthquake in the window, Mww 5.4 Macquarie Island
region at 05:00:54 UTC (us6000f02b), about 115 degrees away; nothing from it
stands out in the record.

Regenerate: pixi run python book/slides/2026/figscripts/lec09_ratt_noise_psd.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pooch  # noqa: E402
from obspy import read, read_inventory  # noqa: E402
from obspy.signal.spectral_estimation import get_nhnm, get_nlnm  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.signal import welch  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED = "#26241f", "#6e675c"
DATA_C, P_C, S_C = "#2a78d6", "#1baf7a", "#eb6834"      # dataviz reference slots 1, 3, 2
CACHE = pooch.os_cache("mlgeo-figscripts")
SNAP = Path(__file__).resolve().parents[3] / "Chapter2-DataManipulation" / "snapshots"
MSEED = SNAP / "ratt_chignik_2021-07-29_UW.RATT.HHZ_0415-0615.mseed"
SXML = SNAP / "ratt_chignik_2021-07-29_UW.RATT.stationxml.xml"
NSEG = 2 ** 15                                         # 327.68 s at 100 Hz
PRIMARY, SECONDARY_LP, SECONDARY_SP = (0.04, 0.1), (0.1, 0.18), (0.18, 0.5)

st = read(str(MSEED))
inv = read_inventory(str(SXML))
tr = st[0]
fs = tr.stats.sampling_rate
sta = inv.select(station="RATT", channel="HHZ")[0][0]
cha = sta.select(channel="HHZ")[0]
counts = tr.data.astype(float)

f = np.fft.rfftfreq(NSEG, 1 / fs)
resp = inv.get_response(tr.id, tr.stats.starttime).get_evalresp_response_for_frequencies(f[1:], output="VEL")
gain = np.abs(resp) ** 2


def acc_psd(x, nseg=NSEG):
    fw, p = welch(x - x.mean(), fs, nperseg=nseg, noverlap=nseg // 2, window="hann", detrend="linear")
    return p[1:] / gain * (2 * np.pi * fw[1:]) ** 2


ff = f[1:]
P = acc_psd(counts)
WIN = int(600 * fs)
P_win = np.array([acc_psd(counts[i:i + WIN]) for i in range(0, counts.size - WIN + 1, WIN)])
lo5, hi95 = np.percentile(10 * np.log10(P_win), [5, 95], axis=0)
PdB = 10 * np.log10(P)
keep = (ff >= 0.01) & (ff <= 40)


def peak(band):
    m = (ff >= band[0]) & (ff <= band[1])
    i = np.argmax(PdB[m])
    return ff[m][i], PdB[m][i]


f_p, db_p = peak(PRIMARY)
f_l, db_l = peak(SECONDARY_LP)
f_s, db_s = peak(SECONDARY_SP)
pf, nlnm = get_nlnm()
_, nhnm = get_nhnm()
def nlnm_at(fr):                                        # obspy lists periods in decreasing order
    return np.interp(np.log10(1 / fr), np.log10(pf[::-1]), nlnm[::-1])


def nhnm_at(fr):
    return np.interp(np.log10(1 / fr), np.log10(pf[::-1]), nhnm[::-1])

vel = tr.copy()
vel.detrend("demean")
vel.remove_response(inventory=inv, output="VEL", pre_filt=(0.03, 0.05, 20, 30), water_level=None)
band = vel.copy().filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True)
v = band.data * 1e9                                     # nm/s
v = v[int(60 * fs):-int(60 * fs)]                       # drop filter edges
kurt = stats.kurtosis(v)
sd = v.std()
frac4 = np.mean(np.abs(v) > 4 * sd)

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16, 9.2))
gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[0.8, 2.2], hspace=0.42, wspace=0.18)
axw = fig.add_subplot(gs[0, 0])
axp = fig.add_subplot(gs[1, 0])
axm = fig.add_subplot(gs[0, 1])
axh = fig.add_subplot(gs[1, 1])

tmin = vel.times() / 60
axw.plot(tmin, vel.data * 1e6, color=DATA_C, lw=0.5)
axw.set_xlim(0, tmin[-1])
axw.set_xlabel(f"minutes after {tr.stats.starttime.strftime('%Y-%m-%d %H:%M')} UTC")
axw.set_ylabel("µm/s")
axw.set_title("(a) UW.RATT HHZ, the two hours before the Chignik M8.2", loc="left")

axp.fill_between(1 / pf, nlnm, nhnm, color="#efebe4", lw=0)
axp.plot(1 / pf, nlnm, color=MUTED, lw=1.2)
axp.plot(1 / pf, nhnm, color=MUTED, lw=1.2)
axp.text(0.012, nlnm_at(0.012) - 3, "NLNM", color=MUTED, fontsize=15, va="top")
axp.text(2.0, nhnm_at(2.0) + 2, "NHNM", color=MUTED, fontsize=15, va="bottom", ha="center")
axp.fill_between(ff[keep], lo5[keep], hi95[keep], color=DATA_C, alpha=0.25, lw=0, label="10-min windows, 5–95 %")
axp.plot(ff[keep], PdB[keep], color=DATA_C, lw=2.4, label="two-hour PSD")
for fr, db, c in ((f_p, db_p, P_C), (f_l, db_l, S_C), (f_s, db_s, S_C)):
    axp.plot(fr, db, marker="o", ms=11, color=c, mec="white", mew=1.5, zorder=4)
axp.annotate(f"primary\n{f_p:.3f} Hz ({1 / f_p:.1f} s)", xy=(f_p, db_p), xytext=(f_p * 0.75, db_p + 22),
             ha="center", fontsize=14, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=1))
xt = (np.sqrt(f_l * f_s) * 1.25, max(db_l, db_s) + 24)
axp.annotate(f"secondary\n{f_l:.3f} Hz ({1 / f_l:.1f} s) and\n{f_s:.3f} Hz ({1 / f_s:.1f} s)", xy=(f_l, db_l),
             xytext=xt, ha="center", va="bottom", fontsize=14, color=INK,
             arrowprops=dict(arrowstyle="-", color=INK, lw=1))
axp.annotate("", xy=(f_s, db_s), xytext=xt, arrowprops=dict(arrowstyle="-", color=INK, lw=1))
axp.set_xscale("log")
axp.set_xlim(0.01, 40)
axp.set_ylim(-195, -85)
axp.set_xlabel("frequency (Hz)")
axp.set_ylabel("dB rel. 1 (m/s²)²/Hz")
axp.legend(frameon=False, fontsize=14, loc="lower right")
axp.set_title("(b) ground acceleration power spectral density", loc="left")
sec = axp.secondary_xaxis("top", functions=(lambda x: 1 / np.maximum(x, 1e-9), lambda x: 1 / np.maximum(x, 1e-9)))
sec.set_xticks([100, 10, 1, 0.1])
sec.set_xticklabels(["100 s", "10 s", "1 s", "0.1 s"], fontsize=14)

land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
                                    known_hash=None, fname="ne_10m_land.zip", path=CACHE, progressbar=False))
ext = (-125.0, -119.5, 45.8, 49.2)
axm.set_facecolor("#eef3f8")
land.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=axm, color="#f4efe6", edgecolor=MUTED, lw=0.6)
axm.plot(sta.longitude, sta.latitude, marker="^", ms=16, color=DATA_C, mec="white", mew=2)
axm.text(sta.longitude + 0.25, sta.latitude, "RATT", fontsize=14, fontweight="bold", va="center")
axm.plot(-122.33, 47.61, marker="o", ms=6, color=INK)
axm.text(-122.45, 47.61, "Seattle", fontsize=13, ha="right", va="center")
axm.set_xlim(ext[:2]); axm.set_ylim(ext[2:])
axm.set_aspect(1 / np.cos(np.deg2rad(47.5)))
axm.set_xticks([]); axm.set_yticks([])
axm.set_xlabel(""); axm.set_ylabel("")
axm.set_title(f"(c) {sta.latitude:.3f}°N, {-sta.longitude:.3f}°W", fontsize=14, fontweight="normal", color=MUTED,
              loc="left")

edges = np.linspace(-6 * sd, 6 * sd, 61)
h, _ = np.histogram(v, edges)
centers = 0.5 * (edges[1:] + edges[:-1])
axh.bar(centers / sd, h, width=(edges[1] - edges[0]) / sd, color=DATA_C, edgecolor="white", lw=0.3)
g = v.size * (edges[1] - edges[0]) * stats.norm.pdf(centers, 0, sd)
axh.plot(centers / sd, g, color=INK, lw=2, label="Gaussian, same std")
axh.set_yscale("log")
axh.set_ylim(0.5, h.max() * 3)
axh.set_xlabel("1–10 Hz velocity (standard deviations)")
axh.set_ylabel("count")
axh.legend(frameon=False, fontsize=13, loc="upper left")
axh.set_title(f"(d) excess kurtosis {kurt:.1f}", loc="left", fontsize=16)
for ax in (axw, axp, axh):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

out = Path(__file__).resolve().parent.parent / "figs" / "lec09_noise" / "ratt_noise_psd.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"{tr.id} {tr.stats.starttime} to {tr.stats.endtime}, {fs:g} Hz, {tr.stats.npts} samples; sensor "
      f"'{cha.sensor.description}'; sensitivity {cha.response.instrument_sensitivity.value:.4g} counts/(m/s) at "
      f"{cha.response.instrument_sensitivity.frequency:g} Hz; station {sta.latitude:.5f} {sta.longitude:.5f}")
print(f"Welch: {NSEG} samples ({NSEG / fs:.2f} s) Hann, 50 % overlap, resolution {ff[0]:.5f} Hz; "
      f"{P_win.shape[0]} ten-minute windows for the percentile band")
print(f"primary microseism peak (search {PRIMARY} Hz): {f_p:.4f} Hz = {1 / f_p:.1f} s, {db_p:.1f} dB "
      f"(NLNM there {nlnm_at(f_p):.1f}, NHNM {nhnm_at(f_p):.1f})")
print(f"long-period secondary peak (search {SECONDARY_LP} Hz): {f_l:.4f} Hz = {1 / f_l:.1f} s, {db_l:.1f} dB "
      f"(NLNM there {nlnm_at(f_l):.1f}, NHNM {nhnm_at(f_l):.1f}); ratio to primary frequency {f_l / f_p:.2f}")
print(f"short-period secondary peak (search {SECONDARY_SP} Hz): {f_s:.4f} Hz = {1 / f_s:.1f} s, {db_s:.1f} dB "
      f"(NLNM there {nlnm_at(f_s):.1f}, NHNM {nhnm_at(f_s):.1f}); ratio to primary frequency {f_s / f_p:.2f}")
for fr in (0.02, 1.0, 5.0, 10.0):
    i = np.argmin(np.abs(ff - fr))
    print(f"PSD at {ff[i]:.3f} Hz: {PdB[i]:.1f} dB; NLNM {nlnm_at(ff[i]):.1f}, NHNM {nhnm_at(ff[i]):.1f}; "
          f"10-min spread 5-95 % {lo5[i]:.1f} to {hi95[i]:.1f}")
print(f"1-10 Hz velocity: std {sd:.2f} nm/s, excess kurtosis {kurt:.2f}, skewness {stats.skew(v):.3f}, "
      f"fraction beyond 4 std {100 * frac4:.4f} % (Gaussian {100 * 2 * stats.norm.sf(4):.4f} %), "
      f"max |v| {np.abs(v).max() / sd:.1f} std; {v.size} samples")
print(f"broadband velocity (0.05-20 Hz pre-filter): std {vel.data.std() * 1e6:.3f} um/s, peak {np.abs(vel.data).max() * 1e6:.3f} um/s")
