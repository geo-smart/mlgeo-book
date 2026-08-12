"""Multi-site tables with planted spatial and group structure.

Stands in for: regional field campaigns and sensor networks — geotechnical
soundings, groundwater wells, seismic stations — where sites come in clusters
(deployments follow roads, basins, and budgets) and each site is measured many
times. The target at every site rides on a smooth regional field, so two
observations from the same site (or from neighboring sites) are not
independent. That dependence is the whole lesson: random K-fold CV scatters a
site's observations across folds and reports skill the model does not have on
new sites; GroupKFold-by-site and buffered spatial splits report the truth.

Breaks down: the regional field is stationary and isotropic (a sum of random
cosines approximating a squared-exponential Gaussian random field); real
fields have anisotropy, faults, and nonstationary variance. Site features are
site-level constants plus jitter; real covariates vary within a site.
"""

import numpy as np
import pandas as pd


def _random_field(length_scale_km, amplitude, n_modes, rng):
    """Smooth stationary Gaussian random field via random Fourier features.

    Wavevectors drawn N(0, 1/length_scale**2) per component give a field whose
    covariance approximates amplitude**2 * exp(-r**2 / (2 length_scale**2)).
    Returns a deterministic callable field(x_km, y_km).
    """
    k = rng.normal(0.0, 1.0 / length_scale_km, size=(n_modes, 2))
    phase = rng.uniform(0, 2 * np.pi, n_modes)
    scale = amplitude * np.sqrt(2.0 / n_modes)

    def field(x_km, y_km):
        x = np.asarray(x_km, dtype=float)
        y = np.asarray(y_km, dtype=float)
        arg = np.multiply.outer(x, k[:, 0]) + np.multiply.outer(y, k[:, 1]) + phase
        return scale * np.cos(arg).sum(axis=-1)

    return field


def multisite_table(
    n_clusters=6,
    sites_per_cluster=5,
    obs_per_site=10,
    domain_km=100.0,
    cluster_spread_km=4.0,
    field_scale_km=25.0,
    field_amplitude=2.0,
    n_features=3,
    feature_site_frac=0.85,
    signal_scale=2.0,
    noise=0.3,
    binary=False,
    positive_fraction=0.12,
    seed=0,
):
    """Site-grouped regression (or small-n classification) table on a regional field.

    Geometry: ``n_clusters`` cluster centers uniform on a ``domain_km`` square;
    ``sites_per_cluster`` sites Gaussian-scattered (``cluster_spread_km``)
    around each center; ``obs_per_site`` repeat observations per site (repeat
    soundings, repeat surveys). Total rows = n_clusters * sites_per_cluster *
    obs_per_site.

    Target: ``y = X @ beta + field(x, y) + noise``, where ``field`` is a smooth
    regional random field (correlation length ``field_scale_km``, standard
    deviation ``field_amplitude``) and ``beta`` is scaled so the feature signal
    has standard deviation ``signal_scale`` — the honest, transportable part of
    the skill. Features are a site-level component (fraction
    ``feature_site_frac`` of unit variance) plus per-observation jitter, so a
    flexible model can recognize a site from its features and memorize that
    site's field value — the leakage that random CV rewards and grouped CV
    exposes. Adding ``x_km``/``y_km`` as features sharpens the ladder: random
    CV interpolates the field almost perfectly, GroupKFold-by-site still leaks
    through cluster neighbors, and only leave-cluster-out (or a buffered
    spatial split) scores the model as a new region would.

    With ``binary=True`` the continuous latent is thresholded at its
    ``1 - positive_fraction`` quantile, giving an imbalanced 0/1 ``label``
    (e.g. liquefaction observed / not observed) with exactly that positive
    fraction. Defaults give 300 rows — the small-n geotech case. Because the
    field drives the latent, positives cluster in space: plain GroupKFold can
    deal folds with no positives at all (undefined AUC). That failure is part
    of the small-sample lesson; StratifiedGroupKFold is the repair.

    Returns ``(df, truth)``. ``df`` columns: ``site_id``, ``cluster_id``,
    ``x_km``, ``y_km``, ``feat_0..``, and ``target`` (or ``label`` and
    ``latent`` when binary). ``truth``: ``field`` (callable field(x_km, y_km)),
    ``beta`` (true feature coefficients), ``site_field`` (Series of the field
    value at each site, indexed by site_id), ``noise`` and geometry parameters.
    """
    rng = np.random.default_rng(seed)
    n_sites = n_clusters * sites_per_cluster
    n = n_sites * obs_per_site

    centers = rng.uniform(0, domain_km, size=(n_clusters, 2))
    cluster_of_site = np.repeat(np.arange(n_clusters), sites_per_cluster)
    site_xy = centers[cluster_of_site] + rng.normal(
        0, cluster_spread_km, size=(n_sites, 2)
    )

    field = _random_field(field_scale_km, field_amplitude, n_modes=256, rng=rng)
    site_field = field(site_xy[:, 0], site_xy[:, 1])

    beta = rng.normal(0, 1.0, n_features)
    beta *= signal_scale / np.linalg.norm(beta)
    site_feat = rng.normal(0, 1.0, size=(n_sites, n_features))

    site_of_obs = np.repeat(np.arange(n_sites), obs_per_site)
    jitter = rng.normal(0, 1.0, size=(n, n_features))
    X = (
        np.sqrt(feature_site_frac) * site_feat[site_of_obs]
        + np.sqrt(1.0 - feature_site_frac) * jitter
    )
    latent = X @ beta + site_field[site_of_obs] + noise * rng.standard_normal(n)

    df = pd.DataFrame(
        {
            "site_id": site_of_obs,
            "cluster_id": cluster_of_site[site_of_obs],
            "x_km": site_xy[site_of_obs, 0],
            "y_km": site_xy[site_of_obs, 1],
        }
    )
    for j in range(n_features):
        df[f"feat_{j}"] = X[:, j]

    truth = {
        "field": field,
        "beta": beta,
        "site_field": pd.Series(site_field, index=pd.RangeIndex(n_sites, name="site_id")),
        "noise": noise,
        "domain_km": domain_km,
        "field_scale_km": field_scale_km,
    }

    if binary:
        thresh = np.quantile(latent, 1.0 - positive_fraction)
        df["latent"] = latent
        df["label"] = (latent > thresh).astype(int)
        truth["threshold"] = thresh
    else:
        df["target"] = latent
    return df, truth


def event_station_table(
    n_events=60,
    n_stations=15,
    n_clusters=4,
    domain_km=200.0,
    cluster_spread_km=8.0,
    cluster_spread_days=20.0,
    span_days=365.0,
    mag_range=(3.0, 6.0),
    event_term_sigma=0.25,
    station_term_sigma=0.15,
    noise=0.1,
    seed=0,
):
    """Event-station ground-motion table with events clustered in space AND time.

    Stands in for a regional ground-motion dataset: earthquake sequences
    (mainshock-aftershock clusters) recorded by a fixed station network. Each
    row is one recording. The amplitude follows a toy attenuation relation

        ``log10_pga = 0.5 * magnitude - 1.5 * log10(dist_km + 10) - 1.0
                      + event_term + station_term + noise``

    (PGA in g, dimensionless log10). The per-event and per-station random
    effects are the leakage mechanism: all recordings of one event share its
    event term, so a random split puts the same event on both sides and the
    model looks better than it is on unseen events. Group by ``event_id``.

    Breaks down: attenuation coefficients are illustrative, not a published
    GMPE; no site amplification vs Vs30, no magnitude-dependent decay, and
    aftershocks do not follow Omori rates — clusters are Gaussian blobs in
    space and time.

    Returns ``(df, truth)``. ``df`` columns: ``event_id``, ``station_id``,
    ``cluster_id``, ``time_days``, ``magnitude``, ``event_x_km``,
    ``event_y_km``, ``station_x_km``, ``station_y_km``, ``dist_km``,
    ``log10_pga``. ``truth``: ``coeffs`` dict of the attenuation relation,
    ``event_terms`` and ``station_terms`` (Series indexed by id).
    """
    rng = np.random.default_rng(seed)

    centers_xy = rng.uniform(0, domain_km, size=(n_clusters, 2))
    centers_t = rng.uniform(0, span_days, n_clusters)
    cluster_of_event = rng.integers(0, n_clusters, n_events)
    ev_xy = centers_xy[cluster_of_event] + rng.normal(
        0, cluster_spread_km, size=(n_events, 2)
    )
    ev_t = centers_t[cluster_of_event] + rng.normal(0, cluster_spread_days, n_events)
    mags = rng.uniform(*mag_range, n_events)
    event_terms = rng.normal(0, event_term_sigma, n_events)

    st_xy = rng.uniform(0, domain_km, size=(n_stations, 2))
    station_terms = rng.normal(0, station_term_sigma, n_stations)

    ev_idx = np.repeat(np.arange(n_events), n_stations)
    st_idx = np.tile(np.arange(n_stations), n_events)
    dist = np.hypot(
        ev_xy[ev_idx, 0] - st_xy[st_idx, 0], ev_xy[ev_idx, 1] - st_xy[st_idx, 1]
    )

    coeffs = {"a_mag": 0.5, "b_dist": -1.5, "c_km": 10.0, "intercept": -1.0}
    median = (
        coeffs["a_mag"] * mags[ev_idx]
        + coeffs["b_dist"] * np.log10(dist + coeffs["c_km"])
        + coeffs["intercept"]
    )
    log10_pga = (
        median
        + event_terms[ev_idx]
        + station_terms[st_idx]
        + noise * rng.standard_normal(len(ev_idx))
    )

    df = pd.DataFrame(
        {
            "event_id": ev_idx,
            "station_id": st_idx,
            "cluster_id": cluster_of_event[ev_idx],
            "time_days": ev_t[ev_idx],
            "magnitude": mags[ev_idx],
            "event_x_km": ev_xy[ev_idx, 0],
            "event_y_km": ev_xy[ev_idx, 1],
            "station_x_km": st_xy[st_idx, 0],
            "station_y_km": st_xy[st_idx, 1],
            "dist_km": dist,
            "log10_pga": log10_pga,
        }
    )
    truth = {
        "coeffs": coeffs,
        "event_terms": pd.Series(
            event_terms, index=pd.RangeIndex(n_events, name="event_id")
        ),
        "station_terms": pd.Series(
            station_terms, index=pd.RangeIndex(n_stations, name="station_id")
        ),
        "noise": noise,
    }
    return df, truth
