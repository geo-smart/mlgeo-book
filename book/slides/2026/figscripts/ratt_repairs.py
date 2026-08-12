"""Slide figures for lecture 8: the pathology curriculum on the real RATT record.

Reproduces the three repair demonstrations of book notebook 2.9 on the real
UW.RATT..HHZ record (100 Hz, 2021-07-29): the M8.2 Chignik, Alaska earthquake
recorded in Washington State. Re-plotted at lecture scale (20 pt base font,
concept-named panel titles):

  figs/2.9_filtering_data/causal_vs_zerophase_slide.png
  figs/2.9_filtering_data/gap_ringing_slide.png
  figs/2.9_filtering_data/clock_shift_slide.png

The waveform is fetched once from the EarthScope DMC and cached as miniSEED in
book/Chapter2-DataManipulation/data/, so re-runs are offline.

Regenerate: pixi run python book/slides/2026/figscripts/ratt_repairs.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import scipy.signal as signal  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "legend.fontsize": 17,
})

BOOK = Path(__file__).resolve().parents[3]  # .../curriculum-book/book
CACHE = BOOK / "Chapter2-DataManipulation" / "data" / "UW.RATT..HHZ_2021-07-29.mseed"
OUT = Path(__file__).resolve().parent.parent / "figs" / "2.9_filtering_data"
TAG = ("Station UW.RATT (Washington) — M8.2 Chignik, Alaska earthquake, 2021-07-29 · "
       "100 Hz vertical, raw counts · EarthScope DMC")
BLUE, ORANGE, RED, GRAY = "#1f77b4", "#ff7f0e", "#b3402a", "#8a8378"


def load_ratt():
    import obspy

    if CACHE.exists():
        st = obspy.read(str(CACHE))
    else:
        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client

        t0 = UTCDateTime(2021, 7, 29, 6, 15)
        st = Client("IRIS").get_waveforms(network="UW", station="RATT", location="--",
                                          channel="HHZ", starttime=t0, endtime=t0 + 7200)
        st.write(str(CACHE), format="MSEED")
    st.merge(); st.detrend(type="linear"); st[0].taper(max_percentage=0.05)
    fs = st[0].stats.sampling_rate
    z = np.asarray(st[0].data, dtype=float)
    return z, fs


z, fs = load_ratt()
t = np.arange(len(z)) / fs
sos = signal.butter(2, [1, 10], "bandpass", fs=fs, output="sos")
zf = signal.sosfiltfilt(sos, z)          # zero-phase reference, used by all three


def stamp(fig):
    fig.suptitle(TAG, x=0.01, ha="left", fontsize=18, fontweight="normal", color="#6e675c")


# ---------------------------------------------------------------- figure 1
# Causal vs zero-phase filtering around the P-wave onset.
zf_causal = signal.sosfilt(sos, z)

fig, ax = plt.subplots(2, 1, figsize=(15, 8.2), sharey=True)
ax[0].plot(t, zf, color=BLUE, lw=1.2, label="zero-phase (filtered twice)")
ax[0].plot(t, zf_causal, color=ORANGE, lw=1.2, alpha=0.85, label="causal (filtered once, forward)")
ax[0].set_xlim(700, 1000)
ax[0].set_title("The earthquake, band-passed 1–10 Hz — two filters, one record", loc="left")
ax[0].legend(loc="upper right", framealpha=0.95)
ax[1].plot(t, zf, color=BLUE, lw=1.6, label="zero-phase: onset time kept")
ax[1].plot(t, zf_causal, color=ORANGE, lw=1.6, alpha=0.85, label="causal: onset pushed later")
ax[1].set_xlim(750, 762)
ax[1].set_title("Zoom on the P-wave onset — the pick moves with the filter", loc="left")
ax[1].legend(loc="upper right", framealpha=0.95)
ax[1].set_xlabel("time (s)")
for a in ax:
    a.grid(True, alpha=0.4); a.set_ylabel("counts")
stamp(fig)
fig.tight_layout(rect=(0, 0, 1, 0.95))
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

fig, ax = plt.subplots(3, 1, figsize=(15, 9.2), sharex=True)
ax[0].plot(t, z_gap, color=BLUE, lw=1.0)
ax[0].set_title("A 20 s telemetry dropout, zero-filled — looks harmless", loc="left")
ax[1].plot(t, zf, color=GRAY, lw=1.0, label="filtered complete record (truth)")
ax[1].plot(t, zf_gap, color=RED, lw=1.0, alpha=0.9, label="filtered straight across the gap")
ax[1].set_title("Filtering across the gap: edge ringing 185× the local signal", loc="left")
ax[1].legend(loc="upper left", framealpha=0.95)
ax[2].plot(t, zf, color=GRAY, lw=1.0, label="filtered complete record (truth)")
ax[2].plot(t, zf_seg, color=BLUE, lw=1.0, alpha=0.9, label="each segment filtered on its own")
ax[2].set_title("Segment-wise filtering: the gap stays a gap", loc="left")
ax[2].legend(loc="upper left", framealpha=0.95)
ax[2].set_xlabel("time (s)")
for a in ax:
    a.axvspan(gap_start, gap_end, color="k", alpha=0.08)
    a.set_xlim(830, 890); a.grid(True, alpha=0.4); a.set_ylabel("counts")
ax[1].set_ylim(-75000, 75000); ax[2].set_ylim(-75000, 75000)
stamp(fig)
fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.8)
fig.savefig(OUT / "gap_ringing_slide.png", dpi=130, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
# A 0.5 s clock error, recovered from the cross-correlation peak.
offset_true = 0.5
z_late = np.roll(z, int(offset_true * fs))
zf_late = signal.sosfiltfilt(sos, z_late)
w0, w1 = int(740 * fs), int(800 * fs)
a_, b_ = zf[w0:w1], zf_late[w0:w1]
cc = signal.correlate(b_, a_, mode="full")
lags = signal.correlation_lags(len(b_), len(a_), mode="full") / fs
lag_best = lags[np.argmax(cc)]

fig, ax = plt.subplots(2, 1, figsize=(15, 8.2))
ax[0].plot(t, zf, color=BLUE, lw=1.6, label="reference clock")
ax[0].plot(t, zf_late, color=ORANGE, lw=1.6, alpha=0.85, label="faulty clock (+0.5 s)")
ax[0].set_xlim(754, 760)
ax[0].set_title("The same P wave under two clocks", loc="left")
ax[0].legend(loc="upper right", framealpha=0.95)
ax[0].set_ylabel("counts"); ax[0].set_xlabel("time (s)")
ax[1].plot(lags, cc / cc.max(), color=BLUE, lw=1.6)
ax[1].axvline(lag_best, color=RED, ls="--", lw=2,
              label=f"peak at {lag_best:+.2f} s — the clock error, measured")
ax[1].set_xlim(-2, 2)
ax[1].set_title("Cross-correlation of the two records", loc="left")
ax[1].legend(loc="upper left", framealpha=0.95)
ax[1].set_xlabel("lag (s)"); ax[1].set_ylabel("normalized correlation")
for a in ax:
    a.grid(True, alpha=0.4)
stamp(fig)
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2)
fig.savefig(OUT / "clock_shift_slide.png", dpi=130, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print(f"recovered lag {lag_best:+.2f} s; wrote 3 figures -> {OUT}")
