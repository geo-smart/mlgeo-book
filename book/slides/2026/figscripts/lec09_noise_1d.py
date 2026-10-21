"""Slide figure (lecture 9): the same two transients buried in four colors of noise of equal variance.

Synthetic, fixed seed. A daily series of N = 4096 samples (about 11 years, the
length of a GNSS position record) carries two transients of the same peak
amplitude, 2 noise standard deviations:
  - a short pulse (Gaussian, 4-day standard deviation), the shape of a
    swarm or a short burst of slip,
  - a slow transient (smooth ramp over about 2 years), the shape of a long
    slow-slip event or postseismic relaxation.
Each is added to four noise models, each scaled by one constant so that its
expected sample standard deviation over the record is 1 (a random walk's
sample standard deviation still varies a lot between realizations and grows
with record length). Panels are ordered by fitted spectral slope:
  white                 independent Gaussian samples, PSD slope 0
  flicker (pink), 1/f   spectral synthesis over 2N samples, amplitude ~ f^(-1/2),
                        first N kept, PSD slope -1
  AR(1), phi = 0.9      first-order autoregressive, x[t] = 0.9 x[t-1] + e[t]; flat
                        below a corner near -ln(phi)/(2 pi) = 0.017 cycles/day and
                        about f^-2 above it, so its fitted slope depends on the
                        band; climate papers call it "red"
  random walk           cumulative sum of white noise, PSD slope -2; Brownian,
                        called "red" in physics
The noise models are generated in a fixed order (white, AR(1), flicker, random
walk) so that the seeded draws do not depend on the panel order.
Left: one realization (mean removed; the first of the ensemble whose sample
standard deviation is within 10 % of 1, so the panels compare like with like;
a random walk's sample variance varies a lot between realizations) with the
true signal in black.
Right: Hann-tapered periodogram of that realization (light) and the mean of 200
realizations (bold), with the power spectra of the two transients (black); the
fitted log-log slope is printed. Detectability is the matched-filter
signal-to-noise ratio SNR = sqrt(s^T C^-1 s), with s the transient and C the
exact N x N covariance matrix of each noise model: the best any detector can do
when it knows the noise statistics.

Regenerate: pixi run python book/slides/2026/figscripts/lec09_noise_1d.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import cho_factor, cho_solve, toeplitz  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED = "#26241f", "#6e675c"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # dataviz reference slots 1-4
SEED, N, NREAL = 2026, 4096, 200
PHI = 0.9                                                # AR(1) coefficient
AMP = 2.0                                                # transient peak, in noise standard deviations
t = np.arange(N)
f = np.fft.rfftfreq(N, 1.0)                              # cycles per day


def white(rng):
    return rng.standard_normal(N)


def red(rng):
    e = rng.standard_normal(N + 500)
    x = np.zeros_like(e)
    for i in range(1, e.size):
        x[i] = PHI * x[i - 1] + e[i]
    return x[500:]                                        # drop the spin-up


f2 = np.fft.rfftfreq(2 * N, 1.0)
AMP_FLICKER = np.r_[0.0, f2[1:] ** -0.5]


def flicker(rng):
    spec = np.fft.rfft(rng.standard_normal(2 * N))
    return np.fft.irfft(spec * AMP_FLICKER, n=2 * N)[:N]


def random_walk(rng):
    return np.cumsum(rng.standard_normal(N))


LABELS = {"white": "white", "red, AR(1)": "AR(1), φ = 0.9 (climate 'red')",
          "flicker, 1/f": "flicker (pink), 1/f", "random walk": "random walk (physics 'red')"}
MODELS = [("white", white, 0), ("red, AR(1)", red, None), ("flicker, 1/f", flicker, -1), ("random walk", random_walk, -2)]


def covariance(name):
    """Exact covariance of the unscaled generator over the N kept samples."""
    lag = np.arange(N)
    if name == "white":
        return np.eye(N)
    if name.startswith("red"):
        return toeplitz(PHI ** lag / (1 - PHI ** 2))
    if name.startswith("flicker"):
        acov = np.fft.irfft(AMP_FLICKER ** 2, n=2 * N)       # circular autocovariance of the 2N generator
        return toeplitz(acov[:N])
    return np.minimum.outer(lag + 1, lag + 1).astype(float)  # random walk from x[-1] = 0


pulse = AMP * np.exp(-0.5 * ((t - 1100) / 4.0) ** 2)
slow = AMP * 0.5 * (1 + np.tanh((t - 2800) / 300.0))
SIGNALS = [("short pulse", pulse), ("slow transient", slow)]
signal = pulse + slow

rng = np.random.default_rng(SEED)
results = []
for name, gen, theo in MODELS:
    raw = np.array([gen(rng) for _ in range(NREAL + 1)])
    scale = 1 / np.sqrt(np.mean(raw.var(axis=1)))          # one constant per model: expected sample var = 1
    ens = (raw[1:] - raw[1:].mean(axis=1, keepdims=True)) * scale
    stds = raw.std(axis=1) * scale
    i_ex = int(np.argmax(np.abs(stds - 1) < 0.1))
    example = (raw[i_ex] - raw[i_ex].mean()) * scale
    taper = np.hanning(N)
    norm = np.sum(taper ** 2)
    P_ens = np.mean(np.abs(np.fft.rfft(ens * taper, axis=1)) ** 2, axis=0) / norm
    P_one = np.abs(np.fft.rfft(example * taper)) ** 2 / norm
    fit = (f > 2 / N) & (f <= 0.5)
    slope = np.polyfit(np.log10(f[fit]), np.log10(P_ens[fit]), 1)[0]
    lo = (f > 2 / N) & (f < 0.01)
    hi = (f >= 0.05) & (f <= 0.5)
    slope_lo = np.polyfit(np.log10(f[lo]), np.log10(P_ens[lo]), 1)[0]
    slope_hi = np.polyfit(np.log10(f[hi]), np.log10(P_ens[hi]), 1)[0]
    cf = cho_factor(covariance(name) * scale ** 2)
    snr = {sname: float(np.sqrt(sig @ cho_solve(cf, sig))) for sname, sig in SIGNALS}
    results.append(dict(name=name, theo=theo, example=example, scale=scale, sample_std=float(example.std()),
                        std_range=np.percentile(stds, [5, 95]), P_ens=P_ens, P_one=P_one, slope=slope,
                        slope_lo=slope_lo, slope_hi=slope_hi, snr=snr,
                        lag1=float(np.mean([np.corrcoef(e[:-1], e[1:])[0, 1] for e in ens]))))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16, 9.4))
gs = fig.add_gridspec(4, 2, width_ratios=[3.0, 1.15], hspace=0.55, wspace=0.16)
yrs = t / 365.25
_tap = np.hanning(N)
S_pulse = np.abs(np.fft.rfft((pulse - pulse.mean()) * _tap)) ** 2 / np.sum(_tap ** 2)
S_slow = np.abs(np.fft.rfft((slow - slow.mean()) * _tap)) ** 2 / np.sum(_tap ** 2)
for i, r in enumerate(sorted(results, key=lambda r: -r["slope"])):
    c = COLORS[[m[0] for m in MODELS].index(r["name"])]
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(yrs, signal, color=INK, lw=1.8, zorder=3)
    ax.plot(yrs, r["example"] + signal, color=c, lw=0.8, zorder=2)
    ax.set_xlim(0, yrs[-1])
    ax.set_ylim(-4.6, 4.6)
    ax.set_yticks([-4, 0, 4])
    ax.set_title(LABELS[r["name"]], loc="left", fontsize=18)
    ax.set_title(f"best-case SNR: pulse {r['snr']['short pulse']:.1f}, slow "
                 f"{r['snr']['slow transient']:.1f}", loc="right", fontsize=15, fontweight="normal")
    if i < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("time (years of daily samples)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    axp = fig.add_subplot(gs[i, 1])
    axp.loglog(f[1:], r["P_one"][1:], color=c, lw=0.5, alpha=0.45)
    axp.loglog(f[1:], r["P_ens"][1:], color=c, lw=2.4)
    axp.loglog(f[1:], S_pulse[1:], color=INK, lw=1.4, ls="--")
    axp.loglog(f[1:], S_slow[1:], color=INK, lw=1.4)
    axp.set_xlim(f[1], 0.5)
    axp.set_ylim(1e-4, 1e4)
    axp.set_yticks([1e-4, 1, 1e4])
    axp.text(0.97, 0.93, f"slope {r['slope']:.2f}", transform=axp.transAxes, ha="right", va="top", fontsize=15,
             color=INK)
    if i < 3:
        axp.set_xticklabels([])
    else:
        axp.set_xlabel("frequency (cycles/day)")
    if i == 0:
        axp.set_title("power spectrum", loc="left", fontsize=17)
        axp.text(1.5e-3, 3e-3, "slow", fontsize=14, color=INK, ha="right")
        axp.text(1.2e-2, 1e-3, "pulse", fontsize=14, color=INK)
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
fig.text(0.07, 0.5, "noise + signal (noise std = 1)", rotation=90, va="center", ha="center", fontsize=17)

out = Path(__file__).resolve().parent.parent / "figs" / "lec09_noise" / "noise_1d_colors.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.1)
print(f"wrote {out}")
print(f"seed {SEED}, N = {N} daily samples ({N / 365.25:.1f} yr), {NREAL} realizations per model, AR(1) phi = {PHI}, "
      f"AR(1) corner frequency -ln(phi)/(2 pi) = {-np.log(PHI) / (2 * np.pi):.4f} cycles/day")
print(f"signals: pulse peak {pulse.max():.2f}, std 4 d at day 1100; slow transient {slow.min():.2f} to {slow.max():.2f}, "
      f"tanh width 300 d centred on day 2800; both in units of noise std")
print("model | slope all f (2/N to 0.5) | slope f < 0.01 | slope 0.05-0.5 | expected | lag-1 autocorr | "
      "SNR pulse | SNR slow")
for r in results:
    print(f"{r['name']:<13} | {r['slope']:6.2f} | {r['slope_lo']:6.2f} | {r['slope_hi']:6.2f} | "
          f"{'n/a (corner)' if r['theo'] is None else r['theo']} | {r['lag1']:.3f} | "
          f"{r['snr']['short pulse']:7.1f} | {r['snr']['slow transient']:7.1f} | scale {r['scale']:.4g}, "
          f"example std {r['sample_std']:.2f}, ensemble sample std 5-95 % "
          f"{r['std_range'][0]:.2f}-{r['std_range'][1]:.2f}")
