"""Physically consistent climate fields and earthquake-magnitude samples.

The climate generator stands in for: gridded monthly surface temperature with
latitudinal structure, hemisphere-antisymmetric seasonal cycle, zonal (land/
ocean-like) variation, a warming trend, and spatially correlated weather noise
— sensible input for EOF/PCA lessons. Values respect physical bounds.

Breaks down: no dynamics, no teleconnections beyond the imposed modes; the
"weather" is smoothed white noise in space and time.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def climate_field(n_lat=40, n_lon=80, n_months=360, trend_c_per_decade=0.25, seed=0):
    """Monthly temperature anomaly field, shape (n_months, n_lat, n_lon), in deg C.

    Structure (all returned in the ``truth`` dict for checking EOF results):
    - seasonal cycle proportional to sin(lat), antisymmetric between hemispheres
      (poles do NOT warm and cool together — the 2024 edition's generator did);
    - a zonal mode mimicking land/ocean contrast (cos of longitude);
    - a linear warming trend amplified at high northern latitudes;
    - spatially correlated noise (gaussian_filter of white noise).

    For EOF analysis remember area weighting: weight each grid cell by
    sqrt(cos(lat)) before the SVD.
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(-88, 88, n_lat)
    lon = np.linspace(0, 360, n_lon, endpoint=False)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    t = np.arange(n_months)

    seasonal_pattern = np.sin(np.deg2rad(LAT))  # antisymmetric
    seasonal_cycle = 10.0 * np.cos(2 * np.pi * (t % 12) / 12.0 - np.pi)  # peak NH summer
    zonal_pattern = 2.0 * np.cos(np.deg2rad(2 * LON)) * np.cos(np.deg2rad(LAT))
    zonal_cycle = np.sin(2 * np.pi * t / 60.0)  # slow 5-yr mode
    arctic_amp = 1.0 + 1.5 * np.clip(np.deg2rad(LAT), 0, None)
    trend = (trend_c_per_decade / 120.0) * t

    field = (
        seasonal_pattern[None] * seasonal_cycle[:, None, None]
        + zonal_pattern[None] * zonal_cycle[:, None, None]
        + arctic_amp[None] * trend[:, None, None]
    )
    noise = rng.standard_normal((n_months, n_lat, n_lon))
    noise = gaussian_filter(noise, sigma=(1.0, 2.0, 2.0)) * 2.0
    field = field + noise

    truth = {
        "lat": lat,
        "lon": lon,
        "seasonal_pattern": seasonal_pattern,
        "zonal_pattern": zonal_pattern,
        "trend_c_per_decade": trend_c_per_decade,
    }
    return field, truth


def gutenberg_richter_magnitudes(n=10_000, b=1.0, m_min=1.0, m_max=8.0, seed=0):
    """Earthquake magnitudes following the Gutenberg-Richter law.

    P(M >= m) ~ 10**(-b (m - m_min)), truncated at ``m_max``. This replaces the
    2024 edition's log-normal "magnitudes", which followed no seismological
    distribution. A histogram of these samples on a log count axis is a
    straight line of slope -b — that check is the sanity test.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    beta = b * np.log(10.0)
    # Inverse-CDF sampling of a truncated exponential in (m - m_min).
    c = 1.0 - np.exp(-beta * (m_max - m_min))
    return m_min - np.log(1.0 - u * c) / beta
