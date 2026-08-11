"""Sanity tests: units, physical bounds, and planted structure of generators."""

import numpy as np

from mlgeo_synth import (
    climate_field,
    detector_features,
    geochem_table,
    gnss_series,
    gutenberg_richter_magnitudes,
    inject_rare_events,
    seismogram_dataset,
    spectrum_matched_noise,
    synthetic_seismogram,
)


def test_gnss_reproducible_and_decomposed():
    a = gnss_series(n_years=3, eq_day=400, seed=1)
    b = gnss_series(n_years=3, eq_day=400, seed=1)
    assert (a["disp_mm"] == b["disp_mm"]).all()
    # Components sum to the noise-free series; residual is the noise.
    resid = a["disp_mm"] - (a["trend_mm"] + a["seasonal_mm"] + a["eq_mm"])
    assert 0.5 < resid.std() < 10.0
    # Coseismic step visible at eq_day.
    assert a["eq_mm"].iloc[399] == 0.0 and a["eq_mm"].iloc[401] > 20.0


def test_seismogram_snr_and_arrivals():
    t, tr, meta = synthetic_seismogram(snr=10.0, seed=2)
    assert meta["t_s"] > meta["t_p"] > 5.0
    pre = tr[t < meta["t_p"] - 1]
    assert np.abs(tr).max() / pre.std() > 3.0  # signal stands above noise


def test_seismogram_dataset_labels():
    X, y, _ = seismogram_dataset(n_events=20, n_noise=20, seed=3)
    assert X.shape == (40, 3000) and y.sum() == 20
    assert np.isfinite(X).all()


def test_spectrum_matched_noise_is_real_and_matched():
    rng = np.random.default_rng(4)
    ref = rng.standard_normal(4096)
    out = spectrum_matched_noise(ref, seed=5)
    assert out.dtype.kind == "f" and len(out) == 4096
    a1 = np.abs(np.fft.rfft(ref))
    a2 = np.abs(np.fft.rfft(out))
    assert np.allclose(a1, a2, rtol=1e-8)


def test_detector_features_imbalance():
    df = detector_features(n=5000, event_fraction=0.02, seed=6)
    assert 0.01 < df["label"].mean() < 0.03
    assert (df["sta_lta"] > 0).all() and (df["dominant_freq_hz"] > 0).all()


def test_geochem_bounds_and_classes():
    df = geochem_table(n=3000, seed=7)
    assert set(df["label"]) == {"granite", "basalt", "andesite"}
    assert (df["SIO2"] > 30).all() and (df["SIO2"] < 90).all()
    assert (df["density_g_cm3"].between(2.3, 3.3)).all()
    # Granite should be the majority class (planted imbalance).
    assert (df["label"] == "granite").mean() > 0.4


def test_climate_field_physical_structure():
    field, truth = climate_field(n_lat=20, n_lon=40, n_months=120, seed=8)
    assert field.shape == (120, 20, 40)
    # Hemisphere antisymmetry of the seasonal cycle: NH and SH January anomalies
    # have opposite sign on average.
    nh = field[0, truth["lat"] > 30].mean()
    sh = field[0, truth["lat"] < -30].mean()
    assert nh * sh < 0


def test_gutenberg_richter_b_value():
    m = gutenberg_richter_magnitudes(n=200_000, b=1.0, seed=9)
    assert m.min() >= 1.0 and m.max() <= 8.0
    # Maximum-likelihood b estimate (Aki 1965) should be ~1.
    b_est = np.log10(np.e) / (m.mean() - 1.0)
    assert 0.9 < b_est < 1.1


def test_rare_events_mask_and_ids():
    base = 10 * np.sin(2 * np.pi * np.arange(3650) / 365.25)
    df = inject_rare_events(base, rate_per_year=5, seed=10)
    assert df["event"].sum() > 0
    assert df["event"].mean() < 0.2  # rare
    assert df.loc[df["event"] == 1, "event_id"].min() >= 0
