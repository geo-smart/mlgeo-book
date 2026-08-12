"""Sanity tests: units, physical bounds, and planted structure of generators."""

import numpy as np

from mlgeo_synth import (
    climate_field,
    degrade_series,
    detector_features,
    event_station_table,
    geochem_table,
    gnss_series,
    gutenberg_richter_magnitudes,
    inject_rare_events,
    multisite_table,
    seismogram_dataset,
    spectrum_matched_noise,
    synthetic_seismogram,
    tide_gauge_series,
    well_table,
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


def test_rare_events_default_tail_unchanged():
    base = np.zeros(3650)
    a = inject_rare_events(base, rate_per_year=5, seed=10)
    b = inject_rare_events(base, rate_per_year=5, tail="uniform", seed=10)
    assert (a["value"] == b["value"]).all() and (a["event_id"] == b["event_id"]).all()
    # Uniform amplitudes stay in the stated bounds; truth column matches events.
    on = a["event"] == 1
    assert a.loc[on, "amplitude"].between(5.0, 20.0).all()
    assert (a.loc[~on, "amplitude"] == 0.0).all()


def test_rare_events_heavy_tails():
    base = np.zeros(200 * 365)

    def event_amps(tail):
        df = inject_rare_events(base, rate_per_year=10, tail=tail, seed=11)
        return df.loc[df["event"] == 1].groupby("event_id")["amplitude"].first()

    uni, logn, gpd = event_amps("uniform"), event_amps("lognormal"), event_amps("gpd")
    assert uni.max() <= 20.0
    # Heavy tails exceed the uniform bound and are right-skewed:
    # the mean sits well above the median.
    for amps in (logn, gpd):
        assert amps.max() > 20.0
        assert amps.mean() > 1.2 * amps.median()


def test_multisite_spatial_and_group_structure():
    df, truth = multisite_table(seed=12)
    df2, _ = multisite_table(seed=12)
    assert (df["target"] == df2["target"]).all()
    assert len(df) == 6 * 5 * 10
    assert {"site_id", "cluster_id", "x_km", "y_km", "feat_0", "target"} <= set(df.columns)
    # Ground-truth field is deterministic and drives the site effect:
    # target minus the feature part averages to the site's field value.
    beta = truth["beta"]
    feats = df[[f"feat_{j}" for j in range(len(beta))]].to_numpy()
    resid = df["target"] - feats @ beta
    site_mean = resid.groupby(df["site_id"]).mean()
    assert np.corrcoef(site_mean, truth["site_field"])[0, 1] > 0.9
    # Spatial autocorrelation: field values at nearby points agree,
    # points far apart (relative to field_scale_km = 25) do not. Sample a
    # transect long enough (200 correlation lengths) for stable estimates.
    field = truth["field"]
    x = np.arange(0, 5000.0, 1.0)
    f0 = field(x, np.zeros_like(x))
    near = np.corrcoef(f0[:-2], f0[2:])[0, 1]  # 2 km apart
    far = np.corrcoef(f0[:-150], f0[150:])[0, 1]  # 150 km apart
    assert near > 0.9
    assert abs(far) < 0.3
    assert near > far


def test_multisite_binary_small_n():
    df, truth = multisite_table(binary=True, positive_fraction=0.12, seed=13)
    assert len(df) == 300
    assert 0.10 <= df["label"].mean() <= 0.15
    assert set(df["label"]) == {0, 1}
    assert "threshold" in truth


def test_event_station_grouping():
    df, truth = event_station_table(seed=14)
    df2, _ = event_station_table(seed=14)
    assert (df["log10_pga"] == df2["log10_pga"]).all()
    assert len(df) == 60 * 15
    assert {"event_id", "station_id", "cluster_id", "time_days", "dist_km"} <= set(df.columns)
    # Recordings of one event share its event term: per-event mean residual
    # from the truth attenuation relation recovers the planted terms.
    c = truth["coeffs"]
    median = (
        c["a_mag"] * df["magnitude"]
        + c["b_dist"] * np.log10(df["dist_km"] + c["c_km"])
        + c["intercept"]
    )
    ev_resid = (df["log10_pga"] - median).groupby(df["event_id"]).mean()
    assert np.corrcoef(ev_resid, truth["event_terms"])[0, 1] > 0.9
    # Events cluster in time: within-cluster time spread << record span.
    spread = df.groupby("cluster_id")["time_days"].std().max()
    assert spread < 100.0


def test_gnss_series_backwards_compatible():
    df = gnss_series(n_years=2, seed=15)
    assert list(df.columns) == ["date", "disp_mm", "trend_mm", "seasonal_mm", "eq_mm"]
    assert len(df) == int(round(2 * 365.25))


def test_degrade_series_truth_matches_injection():
    clean = gnss_series(n_years=4, seed=16)
    out, truth = degrade_series(
        clean,
        gap_windows=[(100, 130)],
        n_random_gaps=2,
        drift_mm=15.0,
        timing_window=(600, 700),
        timing_offset_days=10.0,
        detection_limit_mm=-5.0,
        hetero_sigma_mm=3.0,
        seed=17,
    )
    out2, _ = degrade_series(
        clean,
        gap_windows=[(100, 130)],
        n_random_gaps=2,
        drift_mm=15.0,
        timing_window=(600, 700),
        timing_offset_days=10.0,
        detection_limit_mm=-5.0,
        hetero_sigma_mm=3.0,
        seed=17,
    )
    assert out["disp_mm"].equals(out2["disp_mm"])
    # Gaps: NaN exactly where the truth says, including the specified window.
    assert out["disp_mm"].isna().equals(out["gap"].astype(bool))
    assert out.loc[100:129, "gap"].all()
    assert (100, 130) in truth["gap_windows"] and len(truth["gap_windows"]) == 3
    # Drift is returned and monotone, reaching its stated amplitude.
    assert out["drift_mm"].iloc[0] == 0.0
    assert np.isclose(out["drift_mm"].iloc[-1], 15.0)
    assert (np.diff(out["drift_mm"]) >= 0).all()
    # Timing offset flagged only inside its window.
    flagged = out.index[out["timing_offset_days"] != 0.0]
    assert flagged.min() >= 600 and flagged.max() < 700
    # Censored values sit at the limit.
    assert (out.loc[out["censored"] & ~out["gap"], "disp_mm"] == -5.0).all()
    # Heteroscedastic sigma present, non-negative, and time-varying.
    assert (out["sigma_mm"] >= 0).all() and out["sigma_mm"].std() > 0
    # The clean observed series is preserved in the truth dict.
    assert np.allclose(truth["clean"], clean["disp_mm"])
    # The input frame is untouched.
    assert not clean["disp_mm"].isna().any()


def test_degrade_series_noop_returns_input():
    clean = gnss_series(n_years=2, seed=18)
    out, truth = degrade_series(clean, seed=19)
    assert np.allclose(out["disp_mm"], clean["disp_mm"])
    assert not out["gap"].any() and not out["censored"].any()
    assert (out["drift_mm"] == 0).all() and (out["sigma_mm"] == 0).all()
    assert truth["gap_windows"] == []


def test_well_table_irregular_and_recoverable():
    df, truth = well_table(seed=20)
    df2, _ = well_table(seed=20)
    assert (df["head_m"] == df2["head_m"]).all()
    assert list(df.columns) == ["well_id", "date", "t_yr", "head_m", "sigma_m"]
    assert df["well_id"].nunique() == 25
    # Sampling is uneven: intervals within a well are not all equal.
    dts = df.groupby("well_id")["t_yr"].diff().dropna()
    assert dts.std() > 0 and dts.max() > 1.0  # at least one multi-year gap
    # A few wells have only a handful of observations.
    counts = truth["wells"]["n_obs"]
    assert (counts <= 6).sum() >= 3 and counts.max() > 50
    # Per-well sigma is constant within a well and spans the quality range.
    assert (df.groupby("well_id")["sigma_m"].nunique() == 1).all()
    assert truth["wells"]["sigma_m"].max() / truth["wells"]["sigma_m"].min() > 3
    # Removing the shared regional signal recovers the well offsets.
    resid = df["head_m"] - truth["regional"](df["t_yr"].to_numpy())
    est = resid.groupby(df["well_id"]).mean()
    dense = counts[counts > 10].index
    assert np.corrcoef(est[dense], truth["wells"].loc[dense, "offset_m"])[0, 1] > 0.99


def test_tide_gauge_constituents_and_surges():
    df, truth = tide_gauge_series(n_days=90, seed=21)
    df2, _ = tide_gauge_series(n_days=90, seed=21)
    assert (df["sea_level_m"] == df2["sea_level_m"]).all()
    assert len(df) == 90 * 24
    assert (df["time"].diff().dropna() == np.timedelta64(1, "h")).all()
    # Components sum to the observation (no surges here).
    recon = df["tide_m"] + df["trend_m"] + df["seasonal_m"] + df["surge_m"]
    resid = df["sea_level_m"] - recon
    assert 0.0 < resid.std() < 0.2 and (df["surge_m"] == 0).all()
    # The tide spectrum peaks in the semidiurnal band (M2 dominates).
    amp = np.abs(np.fft.rfft(df["tide_m"].to_numpy()))
    freqs_cpd = np.fft.rfftfreq(len(df), d=1 / 24.0)  # cycles per day
    peak = freqs_cpd[amp.argmax()]
    assert abs(peak - 24.0 / 12.4206012) < 0.05
    # Truth table carries the four constituents with their periods.
    assert list(truth["constituents"].index) == ["M2", "S2", "K1", "O1"]
    assert np.isclose(truth["constituents"].loc["M2", "period_h"], 12.4206012)
    # Surges appear when requested, with labels.
    dfs, _ = tide_gauge_series(n_days=365, surge_rate_per_year=12, seed=22)
    assert dfs["surge"].sum() > 0
    assert dfs.loc[dfs["surge"] == 1, "surge_id"].min() >= 0
    assert dfs.loc[dfs["surge"] == 1, "surge_m"].max() > 0.1
