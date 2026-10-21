"""Slide figure (lecture 9): one deformation bowl under three kinds of 2D noise.

Synthetic, fixed seed. A 256 x 256 grid at 100 m spacing (25.6 km on a side)
carries the vertical surface displacement of a Mogi point source (pressure
change in a small sphere) 4 km deep, scaled to a 30 mm peak uplift, the size
of the signals InSAR resolves at a restless volcano. Three noise fields are
added or applied:
  white          independent Gaussian pixels, standard deviation 6 mm
  correlated     power-law field with power spectral density ~ k^(-8/3), the
                 slope Hanssen (2001) reports for InSAR tropospheric delay
                 between about 0.5 and 6 km (3D turbulence; about -5/3 at
                 larger scales), standard deviation 6 mm
  decorrelation  additive phase noise whose standard deviation follows a
                 coherence map: coherence 0.9 on bare ground, falling to 0.2 in
                 a vegetated patch 4 km east and 2 km north of the source
                 (Gaussian, 3.5 km wide). Phase std from the Cramer-Rao bound
                 sqrt(1 - g^2) / (g sqrt(2L)) (Rodriguez and Martin 1992;
                 Hanssen 2001) with L = 4 looks, capped at pi/sqrt(3) (uniform
                 phase), converted to line-of-sight
                 displacement with lambda / (4 pi), lambda = 55.5 mm (C band,
                 Sentinel-1). The noise level varies in space and does not
                 depend on the deformation. (Speckle multiplies SAR intensity,
                 not displacement; it is on the slide about transforms.)
Top: the four fields on one color scale; the label in each is std(signal)/std(noise)
and, in brackets, peak(signal)/std(noise). Bottom left: east-west profile through
the source. Bottom right: radially averaged power spectral density of the signal
and of each noise field (observed minus signal), with fitted log-log slopes.
Signal-to-noise ratios are printed as std(signal)/std(noise) over the grid and
as peak(signal)/std(noise).

Regenerate: pixi run python book/slides/2026/figscripts/lec09_noise_2d.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED = "#26241f", "#6e675c"
COLORS = {"white": "#2a78d6", "correlated": "#eb6834", "decorrelation": "#1baf7a"}   # dataviz reference slots 1-3
SEED, NPIX, DX_KM = 2026, 256, 0.1
DEPTH_KM, PEAK_MM = 4.0, 30.0
SIGMA_MM, BETA, LOOKS, WAVELENGTH_MM = 6.0, 8 / 3, 4, 55.5

rng = np.random.default_rng(SEED)
x = (np.arange(NPIX) - NPIX / 2) * DX_KM
X, Y = np.meshgrid(x, x)
R2 = X ** 2 + Y ** 2
mogi = DEPTH_KM / (R2 + DEPTH_KM ** 2) ** 1.5
signal = PEAK_MM * mogi / mogi.max()

white = SIGMA_MM * rng.standard_normal((NPIX, NPIX))

kx = np.fft.fftfreq(NPIX, DX_KM)
KX, KY = np.meshgrid(kx, kx)
K = np.hypot(KX, KY)
filt = np.zeros_like(K)
filt[K > 0] = K[K > 0] ** (-BETA / 2)
corr = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((NPIX, NPIX))) * filt))
corr = SIGMA_MM * (corr - corr.mean()) / corr.std()

PATCH = (4.0, 2.0, 3.5)                            # east, north, width (km) of the low-coherence patch
coh = 0.9 - 0.7 * np.exp(-((X - PATCH[0]) ** 2 + (Y - PATCH[1]) ** 2) / (2 * PATCH[2] ** 2))
sigma_phase = np.minimum(np.sqrt(1 - coh ** 2) / (coh * np.sqrt(2 * LOOKS)), np.pi / np.sqrt(3))  # rad; capped at uniform phase
sigma_decor = WAVELENGTH_MM / (4 * np.pi) * sigma_phase              # mm along the line of sight
decor = sigma_decor * rng.standard_normal((NPIX, NPIX))

fields = {"white": signal + white, "correlated": signal + corr, "decorrelation": signal + decor}
noises = {k: v - signal for k, v in fields.items()}


DK = 1 / (NPIX * DX_KM)
RING = np.rint(K / DK).astype(int)                 # ring index: integer multiples of the fundamental wavenumber


def radial_psd(z):
    P = np.abs(np.fft.fft2(z - z.mean())) ** 2 * DX_KM ** 2 / NPIX ** 2
    n = np.arange(1, NPIX // 2 + 1)
    pr = np.bincount(RING.ravel(), P.ravel())[n] / np.bincount(RING.ravel())[n]
    return n * DK, pr


spec = {"signal": radial_psd(signal)}
spec.update({k: radial_psd(v) for k, v in noises.items()})
slopes = {}
for k, (kc, pr) in spec.items():
    ok = np.isfinite(pr) & (pr > 0) & (kc > 2 / (NPIX * DX_KM))
    slopes[k] = np.polyfit(np.log10(kc[ok]), np.log10(pr[ok]), 1)[0]

snr = {k: dict(std_ratio=signal.std() / n.std(), peak_ratio=signal.max() / n.std(), noise_std=n.std(),
               r_center=float(np.corrcoef(fields[k].ravel(), signal.ravel())[0, 1]))
       for k, n in noises.items()}

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16, 9.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.38, wspace=0.12)
gsb = gs[1, :].subgridspec(1, 2, wspace=0.28)
ext = (x[0], x[-1], x[0], x[-1])
vmin, vmax = -20, 40
panels = [("signal", signal, "signal")] + [
    (k, fields[k], {"white": "+ white noise", "correlated": "+ power-law noise",
                    "decorrelation": "+ decorrelation"}[k]) for k in fields]
for i, (k, z, sub) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(z, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(("(a) " if i == 0 else f"({'bcd'[i - 1]}) ") + sub, loc="left", fontsize=16)
    if k != "signal":
        ax.text(0.03, 0.04, f"SNR {snr[k]['std_ratio']:.1f} (peak {snr[k]['peak_ratio']:.1f})",
                transform=ax.transAxes, fontsize=15, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    if k == "decorrelation":
        cs = ax.contour(x, x, coh, levels=[0.5], colors=INK, linewidths=1.6, linestyles="--")
        ax.text(PATCH[0], PATCH[1] - 4.6, "coherence < 0.5", fontsize=13, color=INK, ha="center", va="top")
    ax.set_xticks([-10, 0, 10])
    if i == 0:
        ax.set_yticks([-10, 0, 10])
        ax.set_ylabel("north (km)")
    else:
        ax.set_yticks([])
    ax.set_xlabel("east (km)")
cax = fig.add_axes([0.915, 0.555, 0.009, 0.32])
cb = fig.colorbar(im, cax=cax)
cb.ax.set_title("mm", fontsize=14)

axp = fig.add_subplot(gsb[0])
row = NPIX // 2
for k, z in fields.items():
    axp.plot(x, z[row], color=COLORS[k], lw=1.2, label=k)
axp.plot(x, signal[row], color=INK, lw=2.4, label="signal")
axp.set_xlim(x[0], x[-1])
axp.set_xlabel("east (km), profile through the source")
axp.set_ylabel("uplift (mm)")
axp.legend(frameon=False, fontsize=13, ncol=4, loc="upper left", columnspacing=0.8)
axp.set_ylim(-25, 75)
axp.set_title("(e) east-west profile", loc="left", fontsize=16)

axs = fig.add_subplot(gsb[1])
for k in ["signal", *fields]:
    kc, pr = spec[k]
    axs.loglog(kc, pr, color=INK if k == "signal" else COLORS[k], lw=2.6 if k == "signal" else 2)
    j = -1 if k != "signal" else int(np.argmin(np.abs(kc - 0.6)))
    va = {"white": "bottom", "decorrelation": "bottom", "correlated": "top"}.get(k, "center")
    sl = f"{slopes[k]:.1f}".replace("-0.0", "0.0").replace("-", "−")
    axs.text(kc[j] * 1.12, pr[j], f"{k} ({sl})", fontsize=14, va=va, color=INK)
axs.set_xlabel("wavenumber (cycles/km)")
axs.set_ylabel("PSD (mm² km²)")
axs.set_xlim(spec["signal"][0][0] * 0.8, 25)
axs.set_ylim(1e-6, 1e6)
axs.set_title("(f) radially averaged spectrum (fitted slope)", loc="left", fontsize=16)
for ax in (axp, axs):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

out = Path(__file__).resolve().parent.parent / "figs" / "lec09_noise" / "noise_2d_bowl.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"seed {SEED}; grid {NPIX}x{NPIX} at {DX_KM * 1000:.0f} m; Mogi depth {DEPTH_KM} km, peak {signal.max():.1f} mm, "
      f"signal std over grid {signal.std():.2f} mm, half-peak radius {np.sqrt(2 ** (2 / 3) - 1) * DEPTH_KM:.2f} km")
print(f"noise targets: white and correlated std {SIGMA_MM} mm, power-law exponent {BETA:.3f}; decorrelation: "
      f"coherence {coh.min():.2f} to {coh.max():.2f}, {LOOKS} looks, wavelength {WAVELENGTH_MM} mm, noise std "
      f"{sigma_decor.min():.2f} to {sigma_decor.max():.2f} mm")
print("field | noise std (mm) | SNR std/std | SNR peak/std | SNR std/std in dB | corr(observed, signal) | PSD slope")
for k, v in snr.items():
    print(f"{k:<10} | {v['noise_std']:.2f} | {v['std_ratio']:.2f} | {v['peak_ratio']:.2f} | "
          f"{20 * np.log10(v['std_ratio']):.1f} | {v['r_center']:.3f} | {slopes[k]:.2f}")
print(f"signal PSD slope {slopes['signal']:.2f} (not a power law; fitted for reference)")
low, high = coh < 0.5, coh > 0.85
print(f"decorrelation noise std where coherence < 0.5: {noises['decorrelation'][low].std():.2f} mm "
      f"({100 * low.mean():.1f} % of the grid); where coherence > 0.85: {noises['decorrelation'][high].std():.2f} mm; "
      f"inside the half-peak radius {noises['decorrelation'][signal > PEAK_MM / 2].std():.2f} mm")
for k in fields:
    e = fields[k] - signal
    print(f"{k}: RMS error inside the half-peak radius {np.sqrt(np.mean(e[signal > PEAK_MM / 2] ** 2)):.2f} mm, "
          f"beyond 5 km {np.sqrt(np.mean(e[R2 > 25] ** 2)):.2f} mm")
