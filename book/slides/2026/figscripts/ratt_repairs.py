"""Slide figures for lecture 8: three repairs on the real UW.RATT record.

Reproduces the repair demonstrations of book notebook 2.9 on the real
UW.RATT..HHZ record (100 samples/s, 2021-07-29 06:15 to 08:15 UTC): the M8.2
Chignik, Alaska earthquake (origin 06:15:49 UTC) recorded 2,620 km away at
Rattlesnake Lake, Washington. Sized for a two-column slide (figure plus text):

  figs/2.9_filtering_data/causal_vs_zerophase_slide.png
  figs/2.9_filtering_data/gap_ringing_slide.png
  figs/2.9_filtering_data/clock_shift_slide.png

Correction made here (October 2026): the notebook's "zoom on the P-wave onset"
(750-762 s after 06:15) is not the P wave. The iasp91 P arrival is at 357 s
after 06:15 (308 s travel time, 23.6 degrees). Between 755 and 832 s, during
the largest surface waves, the raw record has 8 steps, each completed within two
or three samples; the sharp pulse at 755 s is the first of them. All eight sit
on steep swings (raw values -367,000 to -635,000 counts, slopes 200,000 to
540,000 counts/s), each jump opposite in sign to the local slope; their sizes,
measured here from a cubic fit on each side, are printed at the end. Cause not
identified, likely instrument or digitizer.
The filter-phase figure now shows the P arrival in context and uses the 755 s
step, whose true time is known from the raw record, to measure what each filter
does. The clock figure correlates a window on the P wave train (355-415 s).

The waveform is fetched once from the EarthScope FDSN service and cached as
miniSEED in book/Chapter2-DataManipulation/data/, so re-runs are offline.

Regenerate: pixi run python book/slides/2026/figscripts/ratt_repairs.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import scipy.signal as signal  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 17,
})

BOOK = Path(__file__).resolve().parents[3]  # .../curriculum-book/book
CACHE = BOOK / "Chapter2-DataManipulation" / "data" / "UW.RATT..HHZ_2021-07-29.mseed"
OUT = Path(__file__).resolve().parent.parent / "figs" / "2.9_filtering_data"
TAG = ("UW.RATT HHZ, Washington · M8.2 Chignik, Alaska, 2021-07-29, 2,620 km away · "
       "raw counts · EarthScope FDSN")
BLUE, ORANGE, RED, GRAY = "#1f77b4", "#ff7f0e", "#b3402a", "#8a8378"
T_P, T_S = 357.0, 609.0      # iasp91 P and S, seconds after 06:15:00 (obspy.taup, depth 32 km)


def load_ratt():
    import obspy

    if CACHE.exists():
        st = obspy.read(str(CACHE))
    else:
        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client

        t0 = UTCDateTime(2021, 7, 29, 6, 15)
        st = Client("EARTHSCOPE").get_waveforms(network="UW", station="RATT", location="--",
                                                channel="HHZ", starttime=t0, endtime=t0 + 7200)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        st.write(str(CACHE), format="MSEED")
    raw = np.asarray(st[0].data, dtype=float).copy()
    st.merge(); st.detrend(type="linear"); st[0].taper(max_percentage=0.05)
    fs = st[0].stats.sampling_rate
    z = np.asarray(st[0].data, dtype=float)
    return raw, z, fs


raw, z, fs = load_ratt()
t = np.arange(len(z)) / fs
sos = signal.butter(2, [1, 10], "bandpass", fs=fs, output="sos")
zf = signal.sosfiltfilt(sos, z)          # zero-phase reference, used by all three


def stamp(fig):
    fig.suptitle(TAG, x=0.01, ha="left", fontsize=15, fontweight="normal", color="#6e675c")


# ---------------------------------------------------------------- figure 1
# Where the P wave is, and what each filter does to a step of known time.
zf_causal = signal.sosfilt(sos, z)

# locate the 8 steps (second difference removes the surface-wave slope)
d2 = np.abs(np.diff(raw, 2))
steps = []
for tc in [755, 767, 783, 792, 805, 814, 825, 832]:
    s = np.flatnonzero((t[:-2] > tc - 1.5) & (t[:-2] < tc + 1.5))
    steps.append(t[s[np.argmax(d2[s])] + 1])
# step sizes: cubic fit to 0.3 s of raw samples on each side of the jump, both
# evaluated at the middle of the jump; the jump samples themselves are excluded
draw = np.diff(raw)
step_info = []
for ts in steps:
    k = int(round(ts * fs))
    slope = np.median(draw[k - 20:k + 20])                       # counts per sample
    near = np.arange(k - 5, k + 6)
    jump = near[np.abs(draw[near] - slope) > 400]               # samples off the local slope
    a, b = jump.min(), jump.max()
    left, right = np.arange(a - 29, a + 1), np.arange(b + 1, b + 31)
    tm = 0.5 * (t[a] + t[b + 1])
    size = (np.polyval(np.polyfit(t[right] - tm, raw[right], 3), 0)
            - np.polyval(np.polyfit(t[left] - tm, raw[left], 3), 0))
    step_info.append((ts, size, raw[a], slope * fs, b - a + 1))
t_step = steps[0]
k = int(round(t_step * fs))
w = slice(k - 150, k + 150)
p = np.polyfit(t[k - 150:k - 2], z[k - 150:k - 2], 2)
raw_local = z[w] - np.polyval(p, t[w])   # raw minus the pre-step surface-wave trend

pre = (t >= t_step - 0.5) & (t < t_step - 0.01)
bg = (t >= t_step - 3) & (t < t_step - 1)
zp_pre, zp_bg = np.abs(zf[pre]).max(), zf[bg].std()
zc_pre = np.abs(zf_causal[pre]).max()
win = (t >= t_step - 1) & (t < t_step + 0.5)
first_zp = t[win][np.argmax(np.abs(zf[win]) > 0.1 * np.abs(zf[win]).max())] - t_step
first_zc = t[win][np.argmax(np.abs(zf_causal[win]) > 0.1 * np.abs(zf_causal[win]).max())] - t_step
post = (t >= t_step) & (t < t_step + 0.5)
pk_zp = t[post][np.argmax(np.abs(zf[post]))] - t_step
pk_zc = t[post][np.argmax(np.abs(zf_causal[post]))] - t_step

fig, ax = plt.subplots(2, 1, figsize=(11.5, 8.6), gridspec_kw={"height_ratios": [1, 1.25]},
                       layout="constrained")
ax[0].plot(t, z / 1e6, color="#26241f", lw=0.9)
ax[0].axvline(T_P, color=BLUE, lw=2); ax[0].axvline(T_S, color=BLUE, lw=2, ls="--")
ax[0].text(T_P + 8, 1.55, "P", color=BLUE, fontsize=20, fontweight="bold")
ax[0].text(T_S + 8, 1.55, "S", color=BLUE, fontsize=20, fontweight="bold")
for ts in steps:
    ax[0].plot(ts, -2.15, marker="^", color=RED, ms=11, clip_on=False)
ax[0].text(845, -1.95, "8 steps", color=RED, fontsize=18, fontweight="bold")
ax[0].set_xlim(300, 1000); ax[0].set_ylim(-2.3, 2.2)
ax[0].set_ylabel("10⁶ counts"); ax[0].set_xlabel("seconds after 06:15:00 UTC")
ax[0].set_title("(a) Raw record: P at 357 s, surface waves after 650 s", loc="left")

ax[1].plot(t[w], raw_local, color=GRAY, lw=2.2, ls="--", label="raw, trend removed (clipped)")
ax[1].plot(t, zf, color=BLUE, lw=2.2, label="zero-phase 1–10 Hz")
ax[1].plot(t, zf_causal, color=ORANGE, lw=2.2, alpha=0.9, label="causal 1–10 Hz")
ax[1].axvline(t_step, color="k", lw=1)
ax[1].set_xlim(t_step - 0.9, t_step + 0.9); ax[1].set_ylim(-2600, 4200)
ax[1].set_ylabel("counts"); ax[1].set_xlabel("seconds after 06:15:00 UTC")
ax[1].set_title(f"(b) Step at {t_step:.2f} s: zero-phase output starts {abs(first_zp):.1f} s early",
                loc="left")
ax[1].legend(loc="lower right", framealpha=0.95, fontsize=15)
for a in ax:
    a.grid(True, alpha=0.35)
stamp(fig)
fig.savefig(OUT / "causal_vs_zerophase_slide.png", dpi=130, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Zero-filled dropout: filtering across it rings; segment-wise filtering does not.
gap_start, gap_end = 850.0, 870.0
i0, i1 = int(gap_start * fs), int(gap_end * fs)
z_gap = z.copy(); z_gap[i0:i1] = 0.0
zf_gap = signal.sosfiltfilt(sos, z_gap)

bad = np.zeros(len(z), dtype=bool); bad[i0:i1] = True
zf_seg = np.full(len(z), np.nan)
good_idx = np.flatnonzero(~bad)
for seg in np.split(good_idx, np.flatnonzero(np.diff(good_idx) > 1) + 1):
    zf_seg[seg] = signal.sosfiltfilt(sos, z[seg])

# grade as the notebook does: 0-2 s from either edge, outside the gap
near = np.zeros(len(z), dtype=bool)
near[int((gap_start - 2) * fs):i0] = True
near[i1:int((gap_end + 2) * fs)] = True
sig_rms = np.sqrt(np.mean(zf[near] ** 2))
err_naive = np.sqrt(np.mean((zf_gap[near] - zf[near]) ** 2))
err_seg = np.sqrt(np.mean((zf_seg[near] - zf[near]) ** 2))
step_lo, step_hi = z[i0 - 1], z[i1]                 # detrended and tapered record
raw_lo, raw_hi = raw[i0 - 1], raw[i1]                # raw counts as archived
far = np.zeros(len(z), dtype=bool)                   # 2-5 s from either edge
far[int((gap_start - 5) * fs):int((gap_start - 2) * fs)] = True
far[int((gap_end + 2) * fs):int((gap_end + 5) * fs)] = True
err_seg_far = np.sqrt(np.mean((zf_seg[far] - zf[far]) ** 2))

fig, ax = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True, layout="constrained")
ax[0].plot(t, z_gap / 1e3, color=BLUE, lw=1.2)
ax[0].set_title("(a) Raw record with 20 s set to zero", loc="left")
ax[0].set_ylabel("10³ counts"); ax[0].set_ylim(-900, 900)
ax[1].plot(t, zf, color=GRAY, lw=1.4, label="complete record, filtered")
ax[1].plot(t, zf_gap, color=RED, lw=1.4, alpha=0.9, label="filtered across the gap")
ax[1].set_title(f"(b) Filtered across the gap: error {err_naive / sig_rms:.0f}× the signal",
                loc="left")
ax[1].legend(loc="upper left", framealpha=0.95, fontsize=15)
ax[2].plot(t, zf, color=GRAY, lw=1.4, label="complete record, filtered")
ax[2].plot(t, zf_seg, color=BLUE, lw=1.4, alpha=0.9, label="each segment filtered alone")
ax[2].set_title(f"(c) Segments filtered alone: error {err_seg / sig_rms:.2f}× the signal",
                loc="left")
ax[2].legend(loc="upper left", framealpha=0.95, fontsize=15)
ax[2].set_xlabel("seconds after 06:15:00 UTC")
for a in ax:
    a.axvspan(gap_start, gap_end, color="k", alpha=0.08)
    a.set_xlim(835, 885); a.grid(True, alpha=0.35)
ax[1].set_ylabel("counts"); ax[1].set_ylim(-75000, 75000)
ax[2].set_ylabel("counts (zoomed)"); ax[2].set_ylim(-400, 400)
stamp(fig)
fig.savefig(OUT / "gap_ringing_slide.png", dpi=130, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
# A 0.5 s clock error, recovered from the cross-correlation peak on the P wave train.
offset_true = 0.5
z_late = np.roll(z, int(offset_true * fs))
zf_late = signal.sosfiltfilt(sos, z_late)
w0, w1 = int(355 * fs), int(415 * fs)
a_, b_ = zf[w0:w1], zf_late[w0:w1]
cc = signal.correlate(b_, a_, mode="full")
lags = signal.correlation_lags(len(b_), len(a_), mode="full") / fs
j = np.argmax(cc)
lag_best = lags[j]
y0, y1, y2 = cc[j - 1], cc[j], cc[j + 1]
lag_sub = lag_best + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) / fs

fig, ax = plt.subplots(2, 1, figsize=(11.5, 8.0), layout="constrained")
ax[0].plot(t, zf, color=BLUE, lw=2.0, label="correct clock")
ax[0].plot(t, zf_late, color=ORANGE, lw=2.0, alpha=0.85, label="clock 0.5 s late")
ax[0].set_xlim(366, 374); ax[0].set_ylim(-900, 900)
ax[0].set_title("(a) P wave train, 1–10 Hz, under two clocks", loc="left")
ax[0].legend(loc="upper left", framealpha=0.95, fontsize=15)
ax[0].set_ylabel("counts"); ax[0].set_xlabel("seconds after 06:15:00 UTC")
ax[1].plot(lags, cc / cc.max(), color=BLUE, lw=2.0)
ax[1].axvline(lag_best, color=RED, ls="--", lw=2.2)
ax[1].set_xlim(-2, 2)
ax[1].set_title(f"(b) Correlation of 355–415 s: peak at {lag_best:+.2f} s", loc="left")
ax[1].set_xlabel("lag (s)"); ax[1].set_ylabel("normalized")
for a in ax:
    a.grid(True, alpha=0.35)
stamp(fig)
fig.savefig(OUT / "clock_shift_slide.png", dpi=130, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print("steps (s after 06:15):", ", ".join(f"{s:.2f}" for s in steps))
print("step sizes (cubic fit, 0.3 s each side): " + "; ".join(
    f"{ts:.2f} s {size:+,.0f} counts over {n} samples at raw {r0:,.0f}, slope {sl:+,.0f} counts/s"
    for ts, size, r0, sl, n in step_info))
print(f"step at {t_step:.2f} s; zero-phase: max {zp_pre:.0f} counts in the 0.5 s before "
      f"(background RMS {zp_bg:.0f}), first >10% of peak at {first_zp:+.2f} s, peak at {pk_zp:+.2f} s")
print(f"causal: max {zc_pre:.0f} counts in the 0.5 s before, first >10% of peak at {first_zc:+.2f} s, "
      f"peak at {pk_zc:+.2f} s")
print(f"gap: level at edges {step_lo:.0f} and {step_hi:.0f} counts detrended, {raw_lo:.0f} and {raw_hi:.0f} raw; "
      f"segment-wise error 2-5 s from the edges {err_seg_far:.3f} counts ({err_seg_far / sig_rms:.4f}x); "
      f"max |filtered| in 835-885 s {np.abs(zf[int(835 * fs):int(885 * fs)]).max():.0f}")
print(f"gap: local signal RMS {sig_rms:.0f}; "
      f"naive error {err_naive:.0f} ({err_naive / sig_rms:.1f}x); segment-wise {err_seg:.1f} "
      f"({err_seg / sig_rms:.2f}x)")
print(f"clock: recovered lag {lag_best:+.2f} s, parabola {lag_sub:+.3f} s; wrote 3 figures -> {OUT}")
