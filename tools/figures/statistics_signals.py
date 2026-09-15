"""Draw the statistics and signal-processing figures for Chapter 2 (2.7-2.9).

Replaces images copied from Gregory Gundersen's blog (mean, variance,
skewness, kurtosis), Ahmet Taspinar's blog (Wavelet-Out1, wavelet_families)
and an untraceable source (filters). Every figure is a plot of a textbook
function computed here with numpy / scipy / pywt.

    pixi run python tools/figures/statistics_signals.py

Outputs (book/img/):
    mean.png              gamma pdf, area shaded, mean marked
    variance.png          three normal pdfs, same mean, sigma = 0.5, 1, 2
    skewness.png          left-skewed / symmetric / right-skewed pdfs
    kurtosis.png          Laplace / normal / uniform at unit variance
    Wavelet-Out1.jpeg     sine (Fourier basis) vs Morlet wavelet
    wavelet_families.png  2 x 4 grid of continuous and discrete wavelets
    filters.png           5th-order low-pass: Butterworth, Chebyshev I/II, elliptic
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "book" / "img"
DPI = 200

TEAL = "#116b66"
ORANGE = "#b3402a"
GREY = "#6e675c"
FILL_BLUE = "#cfe3ee"
FILL_GREEN = "#d9ead3"
FILL_ORANGE = "#f9cb9c"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
    }
)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {OUT / name}")


# ---------------------------------------------------------------- 2.7 moments


def fig_mean():
    """Gamma(k=3, theta=1): mean k*theta = 3, mode (k-1)*theta = 2."""
    k, theta = 3.0, 1.0
    dist = stats.gamma(k, scale=theta)
    z = np.linspace(0, 12, 600)
    p = dist.pdf(z)
    mu = dist.mean()

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.fill_between(z, p, color=FILL_BLUE, label=r"$P(z)$")
    ax.plot(z, p, color=TEAL, lw=2)
    ax.axvline(mu, color=ORANGE, lw=1.8, ls="--")
    ax.annotate(
        r"$\mu = \int z\,P(z)\,dz$",
        xy=(mu, dist.pdf(mu)),
        xytext=(mu + 2.2, 0.22),
        color=ORANGE,
        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1),
        fontsize=11,
    )
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$P(z)$")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 0.3)
    ax.set_xticks([0, mu, 6, 9, 12])
    ax.set_xticklabels(["0", r"$\mu$", "6", "9", "12"])
    ax.get_xticklabels()[1].set_color(ORANGE)
    ax.set_title("Mean: the first raw moment")
    save(fig, "mean.png")


def fig_variance():
    """Three normal pdfs, mean 0, sigma = 0.5, 1, 2."""
    z = np.linspace(-6, 6, 800)
    sigmas = [0.5, 1.0, 2.0]
    colors = [TEAL, ORANGE, GREY]
    fills = [FILL_BLUE, FILL_ORANGE, FILL_GREEN]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for s, c, f in zip(sigmas, colors, fills):
        p = stats.norm(0, s).pdf(z)
        ax.fill_between(z, p, color=f, alpha=0.6)
        ax.plot(z, p, color=c, lw=2, label=rf"$\sigma^2 = {s**2:g}$")
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.text(0.08, 0.83, r"$\mu = 0$", fontsize=11, ha="left")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$P(z)$")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.88)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(r"Variance $\sigma^2 = \int (z-\mu)^2 P(z)\,dz$: same mean, different spread")
    save(fig, "variance.png")


def fig_skewness():
    """Left-skewed (mirrored gamma), normal, right-skewed (gamma), each
    standardized to mean 0 and variance 1 so only the tails differ."""
    k = 4.0
    g = stats.gamma(k)  # mean k, var k, skewness 2/sqrt(k) = 1.0
    zs = np.linspace(-4.5, 4.5, 900)
    sd = np.sqrt(k)

    def right(z):
        return g.pdf(z * sd + k) * sd

    def left(z):
        return right(-z)

    panels = [
        ("Negative skew", left, -float(g.stats(moments="s"))),
        ("Symmetric", stats.norm(0, 1).pdf, 0.0),
        ("Positive skew", right, float(g.stats(moments="s"))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, (title, pdf, m3) in zip(axes, panels):
        p = pdf(zs)
        ax.fill_between(zs, p, color=FILL_BLUE)
        ax.plot(zs, p, color=TEAL, lw=2)
        ax.axvline(0, color=ORANGE, lw=1.5, ls="--")
        m3_txt = "0" if m3 == 0 else f"{m3:+.1f}"
        ax.set_title(f"{title}: $m_3 = {m3_txt}$")
        ax.set_xlabel(r"$(z-\mu)/\sigma$")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(0, 0.62)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.text(0.15, 0.55, r"$\mu$", color=ORANGE, fontsize=11)
    axes[0].set_ylabel(r"$P(z)$")
    axes[0].text(-4.0, 0.42, "long tail\nto the left", color=GREY, fontsize=9, ha="left")
    axes[2].text(4.0, 0.42, "long tail\nto the right", color=GREY, fontsize=9, ha="right")
    fig.tight_layout(w_pad=1.0)
    save(fig, "skewness.png")


def fig_kurtosis():
    """Laplace, normal, uniform; all mean 0, variance 1.
    Excess kurtosis: 3, 0, -1.2."""
    dists = [
        ("Laplace", stats.laplace(0, 1 / np.sqrt(2)), ORANGE),
        ("Normal", stats.norm(0, 1), TEAL),
        ("Uniform", stats.uniform(-np.sqrt(3), 2 * np.sqrt(3)), GREY),
    ]
    z = np.linspace(-5, 5, 2001)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for name, d, c in dists:
        kurt = float(d.stats(moments="k"))
        var = float(d.var())
        assert abs(var - 1) < 1e-9, (name, var)
        p = d.pdf(z)
        label = f"{name} (excess kurtosis {kurt:g})"
        ax1.plot(z, p, color=c, lw=2, label=label)
        ax2.plot(z, p, color=c, lw=2, label=label)
    ax1.set_xlabel(r"$z$")
    ax1.set_ylabel(r"$P(z)$")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 1.0)
    ax1.set_title(r"Same $\mu = 0$ and $\sigma^2 = 1$, different tails")
    ax1.legend(frameon=False, loc="upper right")

    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1)
    ax2.set_xlim(-5, 5)
    ax2.set_xlabel(r"$z$")
    ax2.set_ylabel(r"$P(z)$ (log scale)")
    ax2.set_title("Tails: what the fourth moment weighs")
    ax2.axvspan(-5, -3, color=FILL_ORANGE, alpha=0.4, lw=0)
    ax2.axvspan(3, 5, color=FILL_ORANGE, alpha=0.4, lw=0)
    ax2.text(-4.9, 0.4, r"$|z|>3\sigma$", color=GREY, fontsize=9)
    fig.tight_layout(w_pad=1.5)
    save(fig, "kurtosis.png")


# --------------------------------------------------------------- 2.8 wavelets


def morlet(t, w0=5.0):
    """Real Morlet wavelet, cos(w0 t) exp(-t^2/2)."""
    return np.cos(w0 * t) * np.exp(-(t**2) / 2)


def fig_sine_vs_wavelet():
    t = np.linspace(-10, 10, 2000)
    sine = np.sin(2 * np.pi * 0.8 * t)
    if pywt is not None:
        psi, x = pywt.ContinuousWavelet("morl").wavefun(length=2000)
        wav = np.interp(t, x, psi, left=0, right=0)
        wav_label = "Wavelet basis: Morlet (pywt 'morl'), same frequency, localized in time"
    else:
        wav = morlet(t)
        wav_label = "Wavelet basis: Morlet, same frequency, localized in time"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.0), sharex=True)
    ax1.plot(t, sine, color=TEAL, lw=1.8)
    ax1.set_title("Fourier basis: sine, one frequency, extends over all time", loc="left", fontsize=10)
    ax2.plot(t, wav, color=ORANGE, lw=1.8)
    ax2.set_title(wav_label, loc="left", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim(-1.25, 1.25)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_xlim(-10, 10)
        ax.set_xticks(np.arange(-10, 11, 2.5))
        ax.set_xticklabels([f"{v:g}".replace("-", "\u2212") for v in np.arange(-10, 11, 2.5)])
    ax2.set_xlabel("time")
    # arrows showing the infinite extent of the sine
    ax1.annotate("", xy=(-10, -1.1), xytext=(-8, -1.1), arrowprops=dict(arrowstyle="->", color=GREY))
    ax1.annotate("", xy=(10, -1.1), xytext=(8, -1.1), arrowprops=dict(arrowstyle="->", color=GREY))
    fig.tight_layout(h_pad=0.8)
    fig.savefig(OUT / "Wavelet-Out1.jpeg", format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)
    print(f"wrote {OUT / 'Wavelet-Out1.jpeg'}")


def fig_wavelet_families():
    fig = plt.figure(figsize=(10.0, 5.2))
    top, bottom = fig.subfigures(2, 1, hspace=0.08)
    axes_top = top.subplots(1, 4)
    axes_bot = bottom.subplots(1, 4)
    top.suptitle("Continuous wavelets (closed-form $\\psi(t)$; used by the CWT)", color=TEAL, fontsize=10.5)
    bottom.suptitle("Discrete wavelets (compact support, built from a filter bank; used by the DWT)", color=ORANGE, fontsize=10.5)

    # Row 1: continuous wavelets (closed-form psi, sampled with pywt)
    if pywt is not None:
        cont = [
            ("Mexican hat  'mexh'", "mexh", 5),
            ("Morlet  'morl'", "morl", 5),
            ("Gaussian derivative  'gaus2'", "gaus2", 5),
            ("Complex Morlet  'cmor1.5-1.0'", "cmor1.5-1.0", 3),
        ]
        for ax, (title, name, half) in zip(axes_top, cont):
            psi, x = pywt.ContinuousWavelet(name).wavefun(length=2048)
            if np.iscomplexobj(psi):
                ax.plot(x, psi.real, color=TEAL, lw=1.6, label="real")
                ax.plot(x, psi.imag, color=ORANGE, lw=1.3, ls="--", label="imag")
                ax.legend(frameon=False, loc="upper right", fontsize=8)
            else:
                ax.plot(x, psi, color=TEAL, lw=1.6)
            ax.set_xlim(-half, half)
            ax.set_title(title, fontsize=9.5)
    else:  # numpy fallback
        t = np.linspace(-5, 5, 2048)
        mexh = (2 / (np.sqrt(3) * np.pi**0.25)) * (1 - t**2) * np.exp(-(t**2) / 2)
        gaus2 = -(1 - t**2) * np.exp(-(t**2) / 2)
        cmor = np.exp(2j * np.pi * t) * np.exp(-(t**2) / 1.5)
        cont = [
            ("Mexican hat", mexh, 5),
            ("Morlet", morlet(t), 5),
            ("Gaussian derivative (2nd)", gaus2, 5),
            ("Complex Morlet", cmor, 3),
        ]
        for ax, (title, psi, half) in zip(axes_top, cont):
            if np.iscomplexobj(psi):
                ax.plot(t, psi.real, color=TEAL, lw=1.6, label="real")
                ax.plot(t, psi.imag, color=ORANGE, lw=1.3, ls="--", label="imag")
                ax.legend(frameon=False, loc="upper right", fontsize=8)
            else:
                ax.plot(t, psi, color=TEAL, lw=1.6)
            ax.set_xlim(-half, half)
            ax.set_title(title, fontsize=9.5)

    # Row 2: discrete wavelets (psi from the filter bank by cascade)
    if pywt is not None:
        disc = [
            ("Haar  'haar'", "haar"),
            ("Daubechies  'db4'", "db4"),
            ("Symlet  'sym4'", "sym4"),
            ("Coiflet  'coif2'", "coif2"),
        ]
        for ax, (title, name) in zip(axes_bot, disc):
            phi, psi, x = pywt.Wavelet(name).wavefun(level=8)
            ax.plot(x, psi, color=ORANGE, lw=1.4)
            ax.set_title(title, fontsize=9.5)
    else:
        t = np.linspace(0, 1, 512)
        haar = np.where(t < 0.5, 1.0, -1.0)
        axes_bot[0].plot(t, haar, color=ORANGE, lw=1.4)
        axes_bot[0].set_title("Haar", fontsize=9.5)
        for ax in axes_bot[1:]:
            ax.axis("off")

    for ax in list(axes_top) + list(axes_bot):
        ax.axhline(0, color=GREY, lw=0.5)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("$t$", fontsize=9, labelpad=1)
    axes_top[0].set_ylabel(r"$\psi(t)$", fontsize=10)
    axes_bot[0].set_ylabel(r"$\psi(t)$", fontsize=10)
    for sf in (top, bottom):
        sf.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.2, wspace=0.42)
    save(fig, "wavelet_families.png")


# ---------------------------------------------------------------- 2.9 filters


def gain_db(sos, f):
    """Magnitude response in dB at normalized frequency f (1 = Nyquist)."""
    _, h = signal.sosfreqz(sos, worN=[f * np.pi])
    return 20 * np.log10(abs(h[0]))


def fig_filters():
    """5th-order low-pass filters with the same pass-band edge f_c = 0.3 x
    Nyquist, 1 dB pass-band ripple and 40 dB stop-band attenuation.

    scipy's ``Wn`` means different things per family: the -3 dB point for
    Butterworth, the pass-band edge (-rp) for Chebyshev I and elliptic, and
    the stop-band edge (-rs) for Chebyshev II. So the Chebyshev II ``Wn`` is
    solved for numerically so that its gain at f_c is -rp, which puts all
    four pass-band edges at the same frequency and lets the roll-off differ.
    """
    from scipy.optimize import brentq

    N, fc, rp, rs = 5, 0.3, 1.0, 40.0
    wn2 = brentq(lambda w: gain_db(signal.cheby2(N, rs, w, output="sos"), fc) + rp, fc + 1e-3, 0.95)
    designs = [
        ("Butterworth", signal.butter(N, fc, output="sos"), TEAL, "-"),
        (f"Chebyshev I ({rp:g} dB ripple)", signal.cheby1(N, rp, fc, output="sos"), ORANGE, "-"),
        (f"Chebyshev II ({rs:g} dB stop band)", signal.cheby2(N, rs, wn2, output="sos"), GREY, "-"),
        ("Elliptic", signal.ellip(N, rp, rs, fc, output="sos"), "#7a3e8c", "--"),
    ]
    print(f"  Chebyshev II stop-band edge solved: Wn = {wn2:.3f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for name, sos, c, ls in designs:
        w, h = signal.sosfreqz(sos, worN=4096)
        f = w / np.pi  # normalized to Nyquist
        db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
        ax1.plot(f, db, color=c, lw=1.8, ls=ls, label=name)
        ax2.plot(f, db, color=c, lw=1.8, ls=ls, label=name)

    for ax in (ax1, ax2):
        ax.axvline(fc, color="k", lw=0.8, ls=":")
        ax.set_xlabel(r"frequency / $f_{\rm Nyquist}$")
        ax.set_ylabel("gain (dB)")
    ax1.axhline(-rs, color=GREY, lw=0.8, ls=":")
    ax1.text(0.02, -rs + 1.5, f"$-{rs:g}$ dB stop band", color=GREY, fontsize=8)
    ax1.set_ylim(-90, 5)
    ax1.set_xlim(0, 1)
    ax1.set_title(f"Low-pass, order {N}, pass-band edge $f_c = {fc:g}\\,f_{{\\rm Nyquist}}$", fontsize=10)
    ax1.legend(frameon=False, loc="upper right", fontsize=8)

    ax2.axhline(-rp, color=GREY, lw=0.8, ls=":")
    ax2.axhline(-3, color=GREY, lw=0.8, ls=":")
    ax2.text(0.44, -rp + 0.15, f"$-{rp:g}$ dB ripple", color=GREY, fontsize=8, ha="right")
    ax2.text(0.44, -3 + 0.15, "$-3$ dB", color=GREY, fontsize=8, ha="right")
    ax2.set_xlim(0, 0.45)
    ax2.set_ylim(-6, 0.5)
    ax2.set_title("Pass band zoom: ripple versus roll-off", fontsize=10)
    ax2.text(fc + 0.006, 0.18, "$f_c$", fontsize=9, color="k")
    ax1.text(fc + 0.01, -87, "$f_c$", fontsize=9, color="k")
    fig.tight_layout(w_pad=1.5)
    save(fig, "filters.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("pywt:", getattr(pywt, "__version__", "not available"))
    fig_mean()
    fig_variance()
    fig_skewness()
    fig_kurtosis()
    fig_sine_vs_wavelet()
    fig_wavelet_families()
    fig_filters()
