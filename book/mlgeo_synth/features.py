"""Detector feature tables for binary classification lessons.

Stands in for: waveform-derived features an analyst or trigger algorithm
computes when discriminating earthquakes from noise (STA/LTA ratio, kurtosis,
spectral centroid, dominant frequency). Feature distributions overlap the way
real detector populations do, so precision/recall trade-offs are meaningful.

Breaks down: features are drawn from parametric distributions rather than
computed from waveforms; cross-feature correlations are simplified.
"""

import numpy as np
import pandas as pd


def detector_features(n=2000, event_fraction=0.5, seed=0):
    """Feature table for event-vs-noise classification, physical units in names.

    ``event_fraction=0.5`` gives the balanced teaching set; use 0.02 for the
    1:50 imbalanced variant that makes precision/recall trade-offs concrete
    (a detector tuned for recall floods the analyst with false triggers).

    Columns: sta_lta (dimensionless), kurtosis (dimensionless),
    spectral_centroid_hz, dominant_freq_hz, label (1 = event).
    """
    rng = np.random.default_rng(seed)
    n_event = int(round(n * event_fraction))
    n_noise = n - n_event

    # Events: impulsive (high STA/LTA, heavy-tailed kurtosis), band-limited energy.
    ev = pd.DataFrame(
        {
            "sta_lta": rng.lognormal(np.log(6.0), 0.6, n_event),
            "kurtosis": rng.lognormal(np.log(8.0), 0.7, n_event),
            "spectral_centroid_hz": rng.normal(8.0, 2.5, n_event).clip(0.5),
            "dominant_freq_hz": rng.normal(5.0, 2.0, n_event).clip(0.2),
            "label": 1,
        }
    )
    # Noise: near-Gaussian windows (kurtosis ~ 3), microseism-dominated spectrum.
    nz = pd.DataFrame(
        {
            "sta_lta": rng.lognormal(np.log(1.5), 0.4, n_noise),
            "kurtosis": rng.normal(3.0, 0.8, n_noise).clip(1.0),
            "spectral_centroid_hz": rng.lognormal(np.log(2.0), 0.8, n_noise),
            "dominant_freq_hz": rng.lognormal(np.log(0.8), 0.9, n_noise),
            "label": 0,
        }
    )
    df = pd.concat([ev, nz], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
