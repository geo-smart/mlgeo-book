"""Rare-event injection onto continuous background series.

Stands in for: the defining data problem of the geosciences — fields that vary
continuously over seasons and years, punctuated by short, rare, unpredictably
triggered extremes (floods on a discharge record, eruptions on a tremor
record, offsets on a strain record). Used to teach class imbalance, leakage,
and event-aware train/test splitting: a random split scatters samples from a
single event across train and test and inflates every metric.

Breaks down: event occurrence is a Poisson process — no clustering,
seasonality of triggers, or aftershock-like cascades unless you add them.
"""

import numpy as np
import pandas as pd


def inject_rare_events(
    background,
    rate_per_year=3.0,
    duration_days=(1, 5),
    amplitude=(5.0, 20.0),
    shape="spike",
    samples_per_day=1,
    tail="uniform",
    tail_params=None,
    seed=0,
):
    """Add rare transient events to a 1-D background series.

    ``background`` is any daily-sampled array (e.g. the seasonal component of
    a discharge or GNSS series). Events arrive as a Poisson process with
    ``rate_per_year``; each has uniform random duration (days) and an
    amplitude drawn according to ``tail``, with ``shape`` = 'spike' (linear
    rise, exponential decay) or 'step'.

    ``tail`` selects the amplitude distribution:
    - 'uniform' (default): uniform on ``amplitude`` — bounded, no tail.
      This is the safe classroom case and the trap: a model scored on it
      never meets an event larger than 20.
    - 'lognormal': median and log-sigma from ``tail_params`` (defaults
      ``{"median": 10.0, "sigma": 1.0}``) — most events are ordinary, a few
      are 5-10x the median, as flood and surge records are.
    - 'gpd': generalized Pareto above a threshold, ``tail_params`` defaults
      ``{"loc": 5.0, "scale": 5.0, "xi": 0.5}``; ``xi > 0`` gives a
      power-law tail with infinite variance at ``xi >= 0.5`` — the
      extreme-value regime where bulk metrics (MAE on all samples) stay
      excellent while the largest events are badly missed.

    Returns a DataFrame with ``value``, ``event`` (0/1 mask, the label),
    ``event_id`` (-1 outside events) — event_id exists so lessons can split by
    event rather than by sample — and ``amplitude`` (the true drawn amplitude
    on event samples, 0.0 outside; the ground truth for tail-stratified
    scoring).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(background, dtype=float).copy()
    n = len(x)
    mask = np.zeros(n, dtype=int)
    event_id = np.full(n, -1, dtype=int)
    amplitudes = np.zeros(n)

    if tail == "lognormal":
        p = {"median": 10.0, "sigma": 1.0, **(tail_params or {})}
    elif tail == "gpd":
        p = {"loc": 5.0, "scale": 5.0, "xi": 0.5, **(tail_params or {})}
    elif tail != "uniform":
        raise ValueError(f"tail must be 'uniform', 'lognormal', or 'gpd', got {tail!r}")

    n_events = rng.poisson(rate_per_year * n / (365.25 * samples_per_day))
    starts = np.sort(rng.integers(0, n, n_events))
    for k, s in enumerate(starts):
        dur = int(rng.integers(duration_days[0], duration_days[1] + 1) * samples_per_day)
        if tail == "lognormal":
            amp = p["median"] * np.exp(p["sigma"] * rng.standard_normal())
        elif tail == "gpd":
            # Inverse CDF of the generalized Pareto distribution.
            u = rng.random()
            amp = p["loc"] + p["scale"] * ((1.0 - u) ** (-p["xi"]) - 1.0) / p["xi"]
        else:
            amp = rng.uniform(*amplitude)
        e = min(s + dur, n)
        idx = np.arange(s, e)
        if shape == "step":
            x[s:] += amp
        else:
            rise = np.minimum((idx - s + 1) / max(1, dur // 4), 1.0)
            decay = np.exp(-(idx - s) / max(1.0, dur / 2.0))
            x[idx] += amp * rise * decay
        mask[idx] = 1
        event_id[idx] = k
        amplitudes[idx] = amp

    return pd.DataFrame(
        {"value": x, "event": mask, "event_id": event_id, "amplitude": amplitudes}
    )
