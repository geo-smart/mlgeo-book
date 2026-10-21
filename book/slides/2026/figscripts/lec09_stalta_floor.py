"""Slide figure (lecture 9): the STA/LTA detection floor measured on synthetics with known truth.

Reproduces notebook 2.10, section 4.4, at slide size. The noise is the UW.RATT
HHZ course copy (04:15 to 06:15 UTC on 29 July 2021, before the M8.2 Chignik
earthquake; snapshots/ beside the notebook), merged, linearly detrended and
5 % tapered as in the notebook, with its phases randomized by
mlgeo_synth.spectrum_matched_noise (same amplitude spectrum). The signal is a
Ricker wavelet (width parameter 0.5 s, unit peak) in the middle of a 50 s
window at 100 samples per second.
  Threshold: peak STA/LTA (short-term over long-term average, 1 s / 10 s) of 200
  noise-only windows (seeds 50,000 to 50,199); the 99th percentile fixes the
  false-alarm rate at 1 %.
  Detection: 20 SNR levels from 0.1 to 100 (SNR = peak signal amplitude over
  noise standard deviation), 60 noise windows each (seed 1000 i + k); an event
  is detected when the peak STA/LTA reaches the threshold. Error bars: Wilson
  score interval, z = 1.
Same seeds and steps as the notebook, so the printed numbers match its stored
outputs.

Regenerate: pixi run python book/slides/2026/figscripts/lec09_stalta_floor.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlgeo_synth  # noqa: E402
import numpy as np  # noqa: E402
import obspy  # noqa: E402
from obspy.signal.trigger import classic_sta_lta  # noqa: E402

plt.rcParams.update({"font.size": 18, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED, C1, C2 = "#26241f", "#6e675c", "#2a78d6", "#eb6834"
SNAP = Path(__file__).resolve().parents[3] / "Chapter2-DataManipulation" / "snapshots"
NOISE_FILE = "ratt_chignik_2021-07-29_UW.RATT.HHZ_0415-0615.mseed"

N = obspy.read(str(SNAP / NOISE_FILE))
N.merge()
N.detrend(type="linear")
N[0].taper(max_percentage=0.05)
real_noise = N[0].data.astype(float)
real_noise -= real_noise.mean()

fs, twin, a = 100.0, 50.0, 0.5
t = np.linspace(0, twin, int(twin * fs))
tw = np.arange(-2, 2, 1 / fs)
sa = 2.0 / (np.sqrt(3.0 * a) * np.pi ** 0.25) * (1.0 - (tw / a) ** 2) * np.exp(-(tw ** 2) / (2.0 * a ** 2))
pad = len(t) // 2 - len(sa) // 2
s = np.concatenate((np.zeros(pad), sa, np.zeros(len(t) - pad - len(sa))))
s /= np.max(np.abs(s))

nsta, nlta = int(1 * fs), int(10 * fs)


def peak_stalta(trace):
    return classic_sta_lta(trace, nsta, nlta)[nlta:].max()


noise_peaks = np.array([peak_stalta(mlgeo_synth.spectrum_matched_noise(real_noise, n=len(s), seed=50_000 + k))
                        for k in range(200)])
threshold = np.quantile(noise_peaks, 0.99)

snrs = np.logspace(-1, 2, 20)
n_trials = 60
prob = np.zeros(len(snrs))
for i, snr in enumerate(snrs):
    hits = 0
    for k in range(n_trials):
        nk = mlgeo_synth.spectrum_matched_noise(real_noise, n=len(s), seed=1000 * i + k)
        nk /= nk.std()
        hits += peak_stalta(s + nk / snr) >= threshold
    prob[i] = hits / n_trials
z = 1.0
centre = (prob + z ** 2 / (2 * n_trials)) / (1 + z ** 2 / n_trials)
half = z * np.sqrt(prob * (1 - prob) / n_trials + z ** 2 / (4 * n_trials ** 2)) / (1 + z ** 2 / n_trials)
snr50 = np.interp(0.5, prob, snrs)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8.2, 6.4))
ax.axhline(0.01, color=MUTED, lw=1.2, ls="--")
ax.text(0.11, 0.16, "dashed: 1 % false alarms,\nthreshold set on noise alone", fontsize=15, color=MUTED)
ax.errorbar(snrs, prob, yerr=[prob - (centre - half), centre + half - prob], marker="o", ms=8, color=C1,
            ecolor=C1, capsize=3, lw=2.2)
ax.axvline(snr50, color=C2, lw=2, ls=":")
ax.text(snr50 * 1.12, 0.45, f"50 % at\nSNR {snr50:.2f}", fontsize=17, color=INK, va="top")
ax.set_xscale("log")
ax.set_xlim(0.09, 110)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("SNR (peak signal / noise std)")
ax.set_ylabel("detection probability")
ax.set_title(f"STA/LTA 1 s / 10 s, threshold {threshold:.2f}", loc="left", fontsize=18)
ax.grid(alpha=0.3, which="both")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "figs" / "lec09_noise" / "stalta_floor.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"noise-only peak STA/LTA: median {np.median(noise_peaks):.2f}, 99th percentile {threshold:.2f}")
for snr, p in zip(snrs, prob):
    print(f"SNR {snr:6.2f}: detection probability {p:.2f}")
print(f"detection probability crosses 50% near SNR {snr50:.2f}")
