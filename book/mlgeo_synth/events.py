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
    seed=0,
):
    """Add rare transient events to a 1-D background series.

    ``background`` is any daily-sampled array (e.g. the seasonal component of
    a discharge or GNSS series). Events arrive as a Poisson process with
    ``rate_per_year``; each has uniform random duration (days) and amplitude,
    with ``shape`` = 'spike' (linear rise, exponential decay) or 'step'.

    Returns a DataFrame with ``value``, ``event`` (0/1 mask, the label), and
    ``event_id`` (-1 outside events) — event_id exists so lessons can split by
    event rather than by sample.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(background, dtype=float).copy()
    n = len(x)
    mask = np.zeros(n, dtype=int)
    event_id = np.full(n, -1, dtype=int)

    n_events = rng.poisson(rate_per_year * n / (365.25 * samples_per_day))
    starts = np.sort(rng.integers(0, n, n_events))
    for k, s in enumerate(starts):
        dur = int(rng.integers(duration_days[0], duration_days[1] + 1) * samples_per_day)
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

    return pd.DataFrame({"value": x, "event": mask, "event_id": event_id})
