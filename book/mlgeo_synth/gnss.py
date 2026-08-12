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


def degrade_series(
    df,
    gap_windows=None,
    n_random_gaps=0,
    gap_days=(5, 60),
    drift_mm=0.0,
    drift_shape="exponential",
    drift_tau_years=3.0,
    timing_window=None,
    timing_offset_days=0.0,
    detection_limit_mm=None,
    hetero_sigma_mm=0.0,
    hetero_period_days=365.25,
    seed=0,
):
    """Inject instrument pathologies into a ``gnss_series`` DataFrame.

    Stands in for what real sensor streams do to the textbook signal: outages
    (gaps), slow baseline wander from a degrading antenna or monument
    (drift — distinct from the tectonic trend because it is not linear),
    a clock error over a window (samples report the field at the wrong time),
    a reporting floor (censored values, as on discharge or chemical sensors),
    and noise whose variance changes with conditions (heteroscedastic —
    e.g. winter snow on the antenna). Every pathology is returned as ground
    truth so a repair can be graded.

    Breaks down: pathologies are injected independently; real ones co-occur
    and correlate (the storm that causes the gap also raises the noise), and
    real metadata rarely tells you the truth columns this function hands over.

    Parameters
    ----------
    df : DataFrame from ``gnss_series`` (any DataFrame with ``date`` and
        ``disp_mm`` works). Not modified in place.
    gap_windows : list of (start_day, end_day) index pairs to blank, or None.
    n_random_gaps : additional random gaps, lengths uniform in ``gap_days``.
    drift_mm : total drift amplitude over the record (mm); 0 disables.
    drift_shape : 'exponential' (1 - exp(-t/tau), tau = ``drift_tau_years``)
        or 'quadratic' (t**2, reaching ``drift_mm`` at the end).
    timing_window : (start_day, end_day) over which the clock is wrong, or None.
    timing_offset_days : clock error (days, may be fractional); within the
        window the series reports the displacement at ``t + offset``.
    detection_limit_mm : values below this are reported AT the limit and
        flagged ``censored``; None disables.
    hetero_sigma_mm : amplitude of an added sinusoidally modulated noise
        (mm, 1-sigma at the seasonal peak); 0 disables.
    hetero_period_days : period of the sigma modulation.
    seed : controls gap placement and the added heteroscedastic noise.

    Returns
    -------
    (out, truth) : ``out`` is a copy of ``df`` where ``disp_mm`` is degraded
    (NaN in gaps) plus truth columns ``gap`` (bool), ``drift_mm``,
    ``timing_offset_days`` (per-sample), ``censored`` (bool), ``sigma_mm``
    (per-sample 1-sigma of the added noise; 0 where none was added).
    ``truth`` dict: ``gap_windows`` (all gaps, specified and random),
    ``timing_window``, ``timing_offset_days``, ``detection_limit_mm``, the
    drift parameters, and ``clean`` (the original observed series).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    t_days = np.arange(n)
    t_yr = t_days / 365.25
    disp = out["disp_mm"].to_numpy(dtype=float).copy()
    clean = disp.copy()

    # Drift: slow baseline wander, monotone but curved, added to the signal.
    if drift_mm != 0.0:
        if drift_shape == "quadratic":
            drift = drift_mm * (t_yr / t_yr[-1]) ** 2
        else:
            tau = drift_tau_years
            drift = drift_mm * (1.0 - np.exp(-t_yr / tau)) / (1.0 - np.exp(-t_yr[-1] / tau))
    else:
        drift = np.zeros(n)
    disp = disp + drift

    # Timing offset: within the window, report the field at t + offset.
    offset_col = np.zeros(n)
    if timing_window is not None and timing_offset_days != 0.0:
        s, e = timing_window
        window = (t_days >= s) & (t_days < e)
        shifted = np.interp(t_days + timing_offset_days, t_days, disp)
        disp = np.where(window, shifted, disp)
        offset_col[window] = timing_offset_days

    # Heteroscedastic noise: sinusoidally modulated per-sample sigma.
    sigma = np.zeros(n)
    if hetero_sigma_mm > 0.0:
        sigma = hetero_sigma_mm * 0.5 * (
            1.0 + np.sin(2 * np.pi * t_days / hetero_period_days + rng.uniform(0, 2 * np.pi))
        )
        disp = disp + sigma * rng.standard_normal(n)

    # Censoring: values below the limit are reported at the limit.
    censored = np.zeros(n, dtype=bool)
    if detection_limit_mm is not None:
        censored = disp < detection_limit_mm
        disp = np.where(censored, detection_limit_mm, disp)

    # Gaps last, so a gap wins over every other pathology.
    windows = [tuple(w) for w in (gap_windows or [])]
    for _ in range(n_random_gaps):
        length = int(rng.integers(gap_days[0], gap_days[1] + 1))
        s = int(rng.integers(0, max(1, n - length)))
        windows.append((s, s + length))
    gap = np.zeros(n, dtype=bool)
    for s, e in windows:
        gap[(t_days >= s) & (t_days < e)] = True
    disp[gap] = np.nan

    out["disp_mm"] = disp
    out["gap"] = gap
    out["drift_mm"] = drift
    out["timing_offset_days"] = offset_col
    out["censored"] = censored
    out["sigma_mm"] = sigma
    truth = {
        "gap_windows": windows,
        "timing_window": timing_window,
        "timing_offset_days": timing_offset_days,
        "detection_limit_mm": detection_limit_mm,
        "drift_amplitude_mm": drift_mm,
        "drift_shape": drift_shape,
        "drift_tau_years": drift_tau_years,
        "clean": pd.Series(clean, index=out.index, name="disp_mm_clean"),
    }
    return out, truth
