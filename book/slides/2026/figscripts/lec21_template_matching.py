"""Slide figures for lecture 21, Part 1: what template matching does, drawn step by step.

Synthetic record (mlgeo_synth.synthetic_seismogram, no data file): 60 s of
noise at one station, with the same source (magnitude 2 at 20 km: P then S)
recorded twice. The first earthquake is large (peak 8 times the standard
deviation of the noise): it is the known earthquake, and its 6 s of record are
cut out as the template. The second is small (peak 2.5 times the noise) and
is the one to find.

The score at each position is the correlation coefficient between the template
and the 6 s of record under it: both standardized, multiplied sample by sample,
and averaged. Computing it at every position is the cross-correlation.

Figures, written to figs/lec21_neuron/ (numbers to numbers_template.json):
  tm_take_template.png      the record, the known earthquake, and the template cut from it
  tm_slide_score.png        the template at three positions, and the score at every position
  tm_aligned_products.png   record under the template, sample-by-sample products and their
                            mean: aligned, shifted by 0.08 s, and on noise only

Regenerate: pixi run python book/slides/2026/figscripts/lec21_template_matching.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # book/, for mlgeo_synth
from mlgeo_synth.seismic import synthetic_seismogram  # noqa: E402

plt.rcParams.update({
    "font.size": 22, "axes.titlesize": 24, "axes.titleweight": "bold",
    "axes.labelsize": 23, "xtick.labelsize": 19, "ytick.labelsize": 19,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, INK, MUTED = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#26241f", "#6e675c"
SYN = "Synthetic (mlgeo_synth): one station, the same source at 20 km recorded twice, in noise"
FIGS = HERE.parent / "figs" / "lec21_neuron"
FS, N = 100.0, 600                       # 100 samples per second; template of 6 s
T_KNOWN, T_HIDDEN = 10.0, 38.0           # start of the 6 s that hold each earthquake (s)
A_KNOWN, A_HIDDEN = 8.0, 2.5             # peak amplitude over the standard deviation of the noise
numbers = {}


def save(fig, name):
    fig.text(0.5, -0.03, SYN, ha="center", fontsize=17, color=MUTED)
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def standardize(v):
    return (v - v.mean()) / v.std()


# --- the record ---------------------------------------------------------------------
_, noise, _ = synthetic_seismogram(duration_s=60.0, magnitude=-10, snr=1e-12, seed=5)
noise = noise / noise.std()
_, clean, _ = synthetic_seismogram(magnitude=2.0, distance_km=20.0, snr=1e9, seed=0)
wave = clean[750:750 + N]                # 6 s: P arrives 0.83 s after the start, S 3.21 s
wave = wave / np.abs(wave).max()
record = noise.copy()
for t0, amp in [(T_KNOWN, A_KNOWN), (T_HIDDEN, A_HIDDEN)]:
    record[int(t0 * FS):int(t0 * FS) + N] += amp * wave
t = np.arange(len(record)) / FS
i_known, i_hidden = int(T_KNOWN * FS), int(T_HIDDEN * FS)

# The template: the 6 s of record that hold the known earthquake, noise included
template = standardize(record[i_known:i_known + N])
t_tpl = np.arange(N) / FS


def score_at(i):
    """Correlation coefficient between the template and the record from sample i."""
    return float(np.mean(standardize(record[i:i + N]) * template))


starts = np.arange(len(record) - N)
score = np.array([score_at(i) for i in starts])

# Detection threshold: the largest score the template reaches on 10 minutes of noise with no earthquake
_, quiet, _ = synthetic_seismogram(duration_s=600.0, magnitude=-10, snr=1e-12, seed=6)
frames = np.lib.stride_tricks.sliding_window_view(quiet, N)[::5]
frames = (frames - frames.mean(axis=1, keepdims=True)) / frames.std(axis=1, keepdims=True)
threshold = float((frames @ template / N).max())
numbers["record"] = {"duration_s": 60.0, "template_s": N / FS, "n_samples_template": N,
                     "known_peak_over_noise": A_KNOWN, "hidden_peak_over_noise": A_HIDDEN,
                     "score_at_known": score[i_known], "score_at_hidden": score[i_hidden],
                     "threshold_largest_score_on_600_s_of_noise": threshold,
                     "largest_score_elsewhere": float(np.max(np.delete(score, np.r_[i_known - 100:i_known + 100,
                                                                                    i_hidden - 100:i_hidden + 100])))}

# --- 1. take the template -------------------------------------------------------------
fig = plt.figure(figsize=(21, 7.6))
grid = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1], hspace=0.5)
top, bottom = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 1])
top.plot(t, record, color=MUTED, lw=1.2)
top.plot(t[i_known:i_known + N], record[i_known:i_known + N], color=INK, lw=1.6)
lo, hi = record.min() * 1.15, record.max() * 1.15
top.add_patch(Rectangle((T_KNOWN, lo), N / FS, hi - lo, fc="none", ec=RED, lw=3.5))
top.text(T_KNOWN + N / FS + 0.6, hi * 0.72, "the known earthquake:\ncut these 6 s out", color=RED, fontsize=22, va="center")
top.text(47, hi * 0.72, "is there another one\nin this record?", color=MUTED, fontsize=22, va="center", ha="center")
top.set(xlim=(0, 60), ylim=(lo, hi), title="1. The record of one station")
top.set_xlabel("Time (s)", loc="right")
top.set_yticks([])
bottom.plot(t_tpl, template, color=INK, lw=2.5)
bottom.set(xlabel="Time in the template (s)", xlim=(0, 6), ylim=(template.min() * 1.1, template.max() * 1.1))
bottom.set_yticks([])
for side in ("top", "right"):
    bottom.spines[side].set_visible(True)
for spine in bottom.spines.values():
    spine.set(color=RED, lw=3)
bottom.text(1.0, template.max() * 0.55, "P wave", fontsize=22, color=INK, ha="center")
bottom.text(4.6, template.max() * 0.8, "S wave", fontsize=22, color=INK, ha="center")
fig.text(0.31, 0.27, "2. The template:\n6 s of record,\n600 numbers", fontsize=25, fontweight="bold", ha="right", va="center")
fig.text(0.71, 0.27, "used as\n600 weights", fontsize=25, fontweight="bold", color=RED, ha="left", va="center")
for x_top, x_bot in [(T_KNOWN, 0), (T_KNOWN + N / FS, N / FS)]:
    fig.add_artist(ConnectionPatch(xyA=(x_top, lo), coordsA=top.transData, xyB=(x_bot, template.max() * 1.1),
                                   coordsB=bottom.transData, color=RED, lw=2, ls="--"))
save(fig, "tm_take_template")

# --- 2. slide it along the record: the score at every position ---------------------------
fig, (top, bottom) = plt.subplots(2, 1, figsize=(21, 7.8), sharex=True, gridspec_kw={"hspace": 0.3})
top.plot(t, record, color=MUTED, lw=1.2)
scale = 0.45 * A_KNOWN / np.abs(template).max()
for t0, color, lab in [(22.0, BLUE, "A"), (T_HIDDEN, GREEN, "B"), (50.0, BLUE, "C")]:
    top.plot(t0 + t_tpl, template * scale + hi * 0.55, color=color, lw=1.8)
    top.add_patch(Rectangle((t0, lo), N / FS, hi * 0.98 - lo, fc=color, ec=color, lw=2, alpha=0.10))
    top.text(t0 + 0.2, lo * 0.93, lab, color=color, fontsize=26, fontweight="bold", va="bottom")
    bottom.scatter([t0], [score[int(t0 * FS)]], s=260, color=color, zorder=4)
    bottom.text(t0 + 0.7, score[int(t0 * FS)] + 0.06, f"{lab}: {score[int(t0 * FS)]:.2f}", color=color, fontsize=23, fontweight="bold")
top.annotate("", xy=(34, hi * 0.9), xytext=(29, hi * 0.9), arrowprops=dict(arrowstyle="->", color=INK, lw=3))
top.text(31.5, hi * 0.98, "slide", ha="center", fontsize=22, color=INK)
top.set(ylim=(lo, hi * 1.15), title="The template placed at three positions on the record")
top.set_yticks([])
bottom.plot(starts / FS, score, color=INK, lw=1.8)
bottom.axhline(threshold, color=RED, lw=2.5, ls="--")
bottom.text(59.8, threshold + 0.04, "threshold", color=RED, fontsize=21, ha="right")
bottom.text(T_KNOWN + 0.7, 0.9, "the known earthquake: 1.00", color=INK, fontsize=21)
bottom.set(xlabel="Position of the start of the template on the record (s)", ylabel="Score", xlim=(0, 60), ylim=(-0.45, 1.12),
           title="The score at every position: the cross-correlation")
save(fig, "tm_slide_score")

# --- 3. why the score is high when the waveforms are aligned ------------------------------
SHIFT = 8                                 # samples: 0.08 s, a quarter of the period of the S wave (3 Hz)
cases = [("Aligned (position B)", i_hidden, GREEN), (f"Shifted by {SHIFT / FS:.2f} s", i_hidden + SHIFT, ORANGE),
         ("Noise only (position A)", 2200, BLUE)]
zoom = slice(300, 520)                    # 2.2 s around the S wave, for legibility
fig, axes = plt.subplots(2, 3, figsize=(21, 7.6), sharex=True, gridspec_kw={"hspace": 0.12, "wspace": 0.08})
numbers["three_positions"] = {}
for col, (title, i, color) in enumerate(cases):
    under = standardize(record[i:i + N])
    product = under * template
    value = float(product.mean())
    numbers["three_positions"][title] = value
    a, b = axes[0, col], axes[1, col]
    a.plot(t_tpl[zoom], under[zoom], color=color, lw=2.6, label="record under the template")
    a.plot(t_tpl[zoom], template[zoom], color=INK, lw=2.6, label="template")
    a.set_title(title, color=color)
    a.set_ylim(-4.6, 7.5)
    b.fill_between(t_tpl[zoom], 0, product[zoom], where=product[zoom] >= 0, color=GREEN, alpha=0.85, lw=0)
    b.fill_between(t_tpl[zoom], 0, product[zoom], where=product[zoom] < 0, color="#C0392B", alpha=0.85, lw=0)
    b.axhline(0, color=INK, lw=1)
    b.set_ylim(-9, 17)
    b.text(0.5, 0.99, f"mean of the products:\nscore {value:.2f}", transform=b.transAxes, ha="center", va="top",
           fontsize=24, fontweight="bold", color=color)
    b.set_xlabel("Time in the template (s)")
    for ax in (a, b):
        ax.set_yticks([])
axes[0, 0].legend(frameon=False, fontsize=18, loc="upper right", ncol=1)
axes[0, 0].set_ylabel("Two waveforms")
axes[1, 0].set_ylabel("Product,\nsample by sample")
axes[1, 0].text(0.02, 0.05, "green: same sign\nred: opposite signs", transform=axes[1, 0].transAxes, fontsize=18, color=MUTED, va="bottom")
save(fig, "tm_aligned_products")

(FIGS / "numbers_template.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
