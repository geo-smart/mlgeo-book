"""Slide figures: the 1-D CNN detector's two reality checks (lecture 22).

Two figures, re-plotted at lecture scale from the executed outputs of book
notebook 4.3 (mlgeo_4.3_CNN.ipynb, sections 5.4 and 6). The arrays below are
copied verbatim from the notebook's printed sweep table and miniPNW summary;
retraining the detector here would produce a figure inconsistent with the
book, so the executed numbers are the source of truth. If the notebook is
re-run with different seeds, update these arrays from its printed output.

Figure 1  detection_floor_slide.png   — CNN vs STA/LTA on identical traces
Figure 2  synth_to_real_slide.png     — the synthetic-to-real collapse on miniPNW

Regenerate: pixi run python book/slides/2026/figscripts/lec22_detector_panels.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "legend.fontsize": 17,
})

FIGS = Path(__file__).resolve().parent.parent / "figs" / "mlgeo_4.3_CNN"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- executed sweep table, notebook 4.3 section 5.4 (60 trials per point) ----
snrs = np.array([0.32, 0.50, 0.79, 1.26, 2.00, 3.16, 5.01, 7.94, 12.59, 19.95])
cnn_detect = np.array([0.02, 0.02, 0.07, 0.23, 0.92, 1.00, 1.00, 1.00, 1.00, 1.00])
cnn_fa = np.array([0.00, 0.05, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00])
sta_detect = np.array([0.02, 0.03, 0.00, 0.03, 0.07, 0.05, 0.57, 1.00, 1.00, 1.00])
N_PER = 60


def wilson(p, n, z=1.0):
    """Wilson binomial error bars (asymmetric), as in the notebook."""
    k = np.round(p * n)
    denom = 1 + z**2 / n
    center = (k / n + z**2 / (2 * n)) / denom
    half = z * np.sqrt(k / n * (1 - k / n) / n + z**2 / (4 * n**2)) / denom
    return (p - (center - half), (center + half) - p)


# ---------------------------- Figure 1: detection floor ----------------------------
fig, ax = plt.subplots(figsize=(15, 7.6))
ax.errorbar(snrs, cnn_detect, yerr=wilson(cnn_detect, N_PER), marker="o", ms=9,
            capsize=4, lw=2.5, color="#1f77b4", label="learned detector (1-D CNN)")
ax.errorbar(snrs, sta_detect, yerr=wilson(sta_detect, N_PER), marker="s", ms=9,
            capsize=4, lw=2.5, color="#ff7f0e", label="classical trigger (STA/LTA)")
ax.plot(snrs, cnn_fa, ls="--", lw=1.5, marker="^", ms=6, color="#b3402a",
        label="CNN false alarms (0–5%)")
ax.axhline(0.01, ls=":", lw=1.5, color="#6e675c",
           label="STA/LTA false alarms (pinned at 1%)")
ax.axhline(0.5, color="#6e675c", lw=1, alpha=0.5)
ax.axvspan(1.5, 4.8, color="#116b66", alpha=0.10)
ax.text(3.1, 0.70, "the band the learned\ndetector wins", ha="center", va="center",
        fontsize=18, fontweight="bold", color="#116b66")
ax.annotate("50% crossing\nSNR ≈ 1.5", xy=(1.5, 0.5), xytext=(0.5, 0.62),
            fontsize=18, color="#1f77b4", ha="center",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))
ax.annotate("50% crossing\nSNR ≈ 4.8", xy=(4.8, 0.5), xytext=(11.0, 0.55),
            fontsize=18, color="#ff7f0e", ha="center",
            arrowprops=dict(arrowstyle="->", color="#ff7f0e"))
ax.set_xscale("log")
ax.set_ylim(-0.05, 1.06)
ax.set_xlabel("SNR (peak signal amplitude / noise standard deviation)")
ax.set_ylabel("fraction of event windows detected")
ax.set_title("Two detectors, identical traces, matched false-alarm budgets", loc="left")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), framealpha=0.95)
ax.grid(alpha=0.3, which="both")
fig.suptitle("Synthetic seismograms (mlgeo_synth) — 60 trials per point, Wilson error bars",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out1 = FIGS / "detection_floor_slide.png"
fig.savefig(out1, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out1}")

# ---- executed miniPNW summary, notebook 4.3 section 6 (300 windows each) ----
synthetic_acc = 87.9     # % balanced accuracy, synthetic test set (section 5.3)
real_acc = 46.8          # % balanced accuracy, real miniPNW windows
real_det = 86.3          # % of real earthquake windows flagged
real_fa = 92.7           # % of real pre-event noise windows flagged
snr_bin_edges = ["-12.1\nto 1.7", "1.7\nto 4.8", "4.8\nto 11.2", "11.2\nto 59.1"]
snr_bin_det = np.array([0.99, 0.85, 0.80, 0.81])

fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 7.2))

bars = axl.bar(["synthetic test\n(same generator)", "real miniPNW\n(new noise)"],
               [synthetic_acc, real_acc], width=0.55,
               color=["#1f77b4", "#b3402a"])
axl.axhline(50, ls="--", lw=2, color="#6e675c")
axl.text(-0.42, 52, "chance", fontsize=18, color="#6e675c", ha="left")
for b, v in zip(bars, [synthetic_acc, real_acc]):
    axl.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center",
             fontsize=21, fontweight="bold")
axl.set_ylim(0, 100)
axl.set_ylabel("balanced accuracy (%)")
axl.set_title("Same detector, new noise: the collapse", loc="left")

axr.bar(snr_bin_edges, 100 * snr_bin_det, width=0.6, color="#1f77b4",
        label="real earthquake windows detected")
axr.axhline(real_fa, ls="--", lw=2.5, color="#b3402a",
            label=f"real noise windows flagged ({real_fa:.0f}%)")
axr.set_ylim(0, 105)
axr.set_xlabel("catalog SNR bin (dB, quartiles)")
axr.set_ylabel("flagged as earthquake (%)")
axr.set_title("Detections track the noise", loc="left")
axr.legend(loc="lower left", framealpha=0.95)

for ax in (axl, axr):
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Synthetic-trained CNN scored unchanged on real miniPNW waveforms "
             "(Pacific Northwest, Ni et al. 2023)",
             x=0.01, ha="left", fontsize=20, fontweight="normal", color="#6e675c")
fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5)
out2 = FIGS / "synth_to_real_slide.png"
fig.savefig(out2, dpi=130, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out2}")
