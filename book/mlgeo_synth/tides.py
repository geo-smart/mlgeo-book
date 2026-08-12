"""Hourly tide-gauge sea-level series.

Stands in for: a coastal tide-gauge record (NOAA CO-OPS / PSMSL style) —
deterministic astronomical tide from a few constituents, a slow relative
sea-level trend, an annual steric cycle, weather-driven correlated residuals,
and optional storm surges. The tide is the rare geophysical signal that is
genuinely predictable, which makes it the clean testbed for harmonic
regression, spectral analysis, and forecasting: skill on the tide is nearly
free, skill on the surge is the hard part — score them separately.

Breaks down: four constituents where real harmonic analysis uses 37+ (no
spring-neap beyond M2/S2 beating, no nodal 18.6-yr modulation, no shallow-
water overtides), the trend is linear (real records have vertical-land-motion
breaks), and surges arrive at random rather than riding winter storms.
"""

import numpy as np
import pandas as pd

from .events import inject_rare_events
from .gnss import _colored_noise

# Principal constituents: name -> (period in hours, default amplitude in m).
_CONSTITUENTS = {
    "M2": (12.4206012, 0.80),  # principal lunar semidiurnal
    "S2": (12.0000000, 0.30),  # principal solar semidiurnal
    "K1": (23.9344696, 0.15),  # lunisolar diurnal
    "O1": (25.8193417, 0.10),  # principal lunar diurnal
}


def tide_gauge_series(
    n_days=365.0,
    amplitudes_m=None,
    trend_mm_yr=3.0,
    seasonal_m=0.08,
    noise_m=0.05,
    surge_rate_per_year=0.0,
    surge_amplitude_m=(0.3, 1.2),
    surge_duration_days=(1, 2),
    surge_tail="uniform",
    seed=0,
):
    """Hourly sea level (m above local datum) with known tidal ground truth.

    Signal = sum of the M2, S2, K1, O1 constituents (periods fixed at their
    astronomical values, phases seeded random, amplitudes from
    ``amplitudes_m`` or the defaults) + linear trend (``trend_mm_yr``) +
    annual cycle (``seasonal_m``) + flicker-spectrum residual (``noise_m``,
    1-sigma — the weather) + optional storm surges injected with
    ``inject_rare_events`` when ``surge_rate_per_year > 0`` (``surge_tail``
    passes through, so surges can be heavy-tailed).

    Returns ``(df, truth)``. ``df`` columns: ``time`` (hourly datetimes),
    ``sea_level_m`` (the observation), and the components ``tide_m``,
    ``trend_m``, ``seasonal_m``, ``surge_m``, plus ``surge`` /``surge_id``
    labels (0/-1 when surges are off). ``truth``: ``constituents``
    (DataFrame: ``period_h``, ``amplitude_m``, ``phase_rad`` per
    constituent), ``trend_mm_yr``, ``seasonal_m``, ``noise_m``.
    """
    rng = np.random.default_rng(seed)
    n = int(round(n_days * 24))
    t_h = np.arange(n, dtype=float)
    t_yr = t_h / (24 * 365.25)

    amps = dict({k: v[1] for k, v in _CONSTITUENTS.items()}, **(amplitudes_m or {}))
    rows = {}
    tide = np.zeros(n)
    for name, (period_h, _) in _CONSTITUENTS.items():
        phase = rng.uniform(0, 2 * np.pi)
        tide += amps[name] * np.cos(2 * np.pi * t_h / period_h + phase)
        rows[name] = (period_h, amps[name], phase)

    trend = (trend_mm_yr / 1000.0) * t_yr
    seasonal = seasonal_m * np.sin(2 * np.pi * t_yr + rng.uniform(0, 2 * np.pi))
    noise = noise_m * _colored_noise(n, 1.0, rng)

    surge = np.zeros(n)
    surge_mask = np.zeros(n, dtype=int)
    surge_id = np.full(n, -1, dtype=int)
    if surge_rate_per_year > 0:
        ev = inject_rare_events(
            np.zeros(n),
            rate_per_year=surge_rate_per_year,
            duration_days=surge_duration_days,
            amplitude=surge_amplitude_m,
            shape="spike",
            samples_per_day=24,
            tail=surge_tail,
            seed=int(rng.integers(0, 2**31)),
        )
        surge = ev["value"].to_numpy()
        surge_mask = ev["event"].to_numpy()
        surge_id = ev["event_id"].to_numpy()

    sea_level = tide + trend + seasonal + surge + noise
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "time": times,
            "sea_level_m": sea_level,
            "tide_m": tide,
            "trend_m": trend,
            "seasonal_m": seasonal,
            "surge_m": surge,
            "surge": surge_mask,
            "surge_id": surge_id,
        }
    )
    truth = {
        "constituents": pd.DataFrame(
            rows, index=["period_h", "amplitude_m", "phase_rad"]
        ).T.rename_axis("constituent"),
        "trend_mm_yr": trend_mm_yr,
        "seasonal_m": seasonal_m,
        "noise_m": noise_m,
    }
    return df, truth
