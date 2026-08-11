"""Synthetic GNSS daily position time series.

Stands in for: daily station positions such as the Nevada Geodetic Laboratory
tenv3 products (secular plate motion, seasonal hydrological loading, earthquake
offsets, postseismic relaxation, and temporally correlated noise).

Breaks down: real GNSS noise includes equipment changes, snow on antennas,
reference-frame realization jumps, and common-mode regional signals that this
generator does not model. Use it to learn methods, not to publish velocities.
"""

import numpy as np
import pandas as pd


def _colored_noise(n, exponent, rng):
    """Noise with power spectrum ~ 1/f**exponent, unit variance, zero mean."""
    freqs = np.fft.rfftfreq(n, d=1.0)
    amp = np.ones_like(freqs)
    amp[1:] = freqs[1:] ** (-exponent / 2.0)
    amp[0] = 0.0
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = amp * np.exp(1j * phases)
    x = np.fft.irfft(spectrum, n=n)
    return (x - x.mean()) / x.std()


def gnss_series(
    n_years=10.0,
    velocity_mm_yr=12.0,
    annual_mm=3.0,
    semiannual_mm=1.0,
    eq_day=None,
    coseismic_mm=25.0,
    postseismic_tau_days=90.0,
    postseismic_mm=10.0,
    white_mm=1.0,
    flicker_mm=2.0,
    random_walk_mm=0.5,
    seed=0,
):
    """Generate a daily one-component GNSS displacement series in millimeters.

    Parameters mirror the physical decomposition taught in Chapter 2/3:
    secular velocity + annual/semi-annual loading + optional coseismic step at
    ``eq_day`` + postseismic logarithmic decay + colored noise (white +
    flicker 1/f + random walk 1/f^2).

    Returns a DataFrame with columns ``date``, ``disp_mm`` (the observed
    series) and the noise-free components (``trend_mm``, ``seasonal_mm``,
    ``eq_mm``) so lessons can compare estimates to ground truth.
    """
    rng = np.random.default_rng(seed)
    n = int(round(n_years * 365.25))
    t_days = np.arange(n)
    t_yr = t_days / 365.25

    trend = velocity_mm_yr * t_yr
    seasonal = annual_mm * np.sin(2 * np.pi * t_yr + rng.uniform(0, 2 * np.pi)) + (
        semiannual_mm * np.sin(4 * np.pi * t_yr + rng.uniform(0, 2 * np.pi))
    )

    eq = np.zeros(n)
    if eq_day is not None:
        after = t_days >= eq_day
        dt = np.where(after, t_days - eq_day, 0.0)
        eq = after * coseismic_mm + after * postseismic_mm * np.log1p(
            dt / postseismic_tau_days
        )

    noise = (
        white_mm * rng.standard_normal(n)
        + flicker_mm * _colored_noise(n, 1.0, rng)
        + random_walk_mm * _colored_noise(n, 2.0, rng)
    )

    disp = trend + seasonal + eq + noise
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "disp_mm": disp,
            "trend_mm": trend,
            "seasonal_mm": seasonal,
            "eq_mm": eq,
        }
    )
