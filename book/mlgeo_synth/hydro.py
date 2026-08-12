"""Irregular multi-well groundwater-head records.

Stands in for: decades of manual and transducer water-level measurements
across a monitoring network — the archetypal irregular sparse point stream.
Wells share a regional signal (seasonal recharge plus a multi-year decline)
but each has its own datum offset, its own measurement quality, its own
active period, multi-year gaps, and uneven sampling; a few wells were visited
only a handful of times. Nothing about the sampling is regular, and that is
the lesson: resampling, gap policy, and per-well weighting must be chosen,
not defaulted.

Breaks down: every well feels the same regional signal with the same
amplitude and no lag; real aquifers attenuate and delay recharge with depth
and transmissivity, and pumping produces local drawdown cones this generator
does not model. Sampling times are independent of the water level — real
records are visited more often when something is wrong (informative
missingness is absent).
"""

import numpy as np
import pandas as pd


def _regional_head(t_yr, trend_m_per_decade, seasonal_m, seasonal_phase, interannual_m):
    """Shared regional head anomaly (m) at decimal-year times ``t_yr``."""
    return (
        trend_m_per_decade * t_yr / 10.0
        + seasonal_m * np.sin(2 * np.pi * t_yr + seasonal_phase)
        + interannual_m * np.sin(2 * np.pi * t_yr / 4.7)
    )


def well_table(
    n_wells=25,
    n_years=40.0,
    start="1980-01-01",
    mean_obs_per_year=6.0,
    trend_m_per_decade=-0.5,
    seasonal_m=1.0,
    interannual_m=0.4,
    offset_sigma_m=8.0,
    sigma_range_m=(0.02, 0.3),
    n_sparse_wells=3,
    domain_km=50.0,
    seed=0,
):
    """Long-format groundwater-head table, irregular in space and time.

    Each well's head (m above an arbitrary regional datum) is

        ``head = offset_well + regional(t) + noise(sigma_well)``

    where ``regional(t)`` = linear trend (``trend_m_per_decade``) + annual
    recharge cycle (``seasonal_m``) + a slow ~4.7-yr wet/dry mode
    (``interannual_m``). Per-well irregularity, all seeded:

    - active period: each well starts and ends at random times inside the
      record, so wells overlap only partially;
    - sampling: measurement dates are drawn uniformly within the active
      period at an average of ``mean_obs_per_year`` visits/yr — intervals are
      uneven by construction;
    - gaps: each well loses one multi-year window (1-8 yr) of visits;
    - quality: per-well 1-sigma noise drawn log-uniformly in
      ``sigma_range_m`` (transducer vs steel-tape territory), reported in the
      ``sigma_m`` column;
    - ``n_sparse_wells`` wells keep only 3-6 observations total.

    Returns ``(df, truth)``. ``df``: ``well_id``, ``date``, ``t_yr`` (decimal
    years since ``start``), ``head_m``, ``sigma_m``, sorted by well then time.
    ``truth``: ``wells`` (DataFrame per well: ``x_km``, ``y_km``,
    ``offset_m``, ``sigma_m``, ``n_obs``), ``regional`` (callable
    regional(t_yr)), and the regional parameters.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)

    def regional(t_yr):
        return _regional_head(
            np.asarray(t_yr, dtype=float),
            trend_m_per_decade,
            seasonal_m,
            phase,
            interannual_m,
        )

    xy = rng.uniform(0, domain_km, size=(n_wells, 2))
    offsets = rng.normal(0, offset_sigma_m, n_wells)
    lo, hi = sigma_range_m
    sigmas = np.exp(rng.uniform(np.log(lo), np.log(hi), n_wells))
    sparse = rng.choice(n_wells, size=min(n_sparse_wells, n_wells), replace=False)

    frames = []
    for w in range(n_wells):
        t0 = rng.uniform(0, 0.4 * n_years)
        t1 = rng.uniform(0.6 * n_years, n_years)
        if w in sparse:
            n_obs = int(rng.integers(3, 7))
        else:
            n_obs = max(2, rng.poisson(mean_obs_per_year * (t1 - t0)))
        t = np.sort(rng.uniform(t0, t1, n_obs))
        # One multi-year gap per (non-sparse) well.
        if w not in sparse:
            g0 = rng.uniform(t0, t1 - 1.0)
            g1 = g0 + rng.uniform(1.0, 8.0)
            t = t[(t < g0) | (t > g1)]
        head = offsets[w] + regional(t) + sigmas[w] * rng.standard_normal(len(t))
        frames.append(
            pd.DataFrame(
                {
                    "well_id": w,
                    "t_yr": t,
                    "head_m": head,
                    "sigma_m": sigmas[w],
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    origin = pd.Timestamp(start)
    df.insert(1, "date", origin + pd.to_timedelta(df["t_yr"] * 365.25, unit="D"))

    wells = pd.DataFrame(
        {
            "x_km": xy[:, 0],
            "y_km": xy[:, 1],
            "offset_m": offsets,
            "sigma_m": sigmas,
            "n_obs": df.groupby("well_id").size().reindex(range(n_wells), fill_value=0),
        },
        index=pd.RangeIndex(n_wells, name="well_id"),
    )
    truth = {
        "wells": wells,
        "regional": regional,
        "trend_m_per_decade": trend_m_per_decade,
        "seasonal_m": seasonal_m,
        "seasonal_phase": phase,
        "interannual_m": interannual_m,
    }
    return df, truth
