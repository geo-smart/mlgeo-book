"""mlgeo_synth — physically motivated synthetic data generators for the MLGeo book.

Every generator in this package produces data with named physical quantities,
units, and a documented statistical structure, so that lessons on training,
evaluation, and data curation run against signals whose ground truth is known.
Each docstring states what the synthetic data stands in for and where the
approximation breaks down; that framing is part of the curriculum (see
Chapter 2.10, "when is synthetic data admissible").

All generators accept a ``seed`` argument and are deterministic given it.
Instructors regenerate hidden test sets each year from a private seed.
"""

from .gnss import gnss_series, degrade_series
from .seismic import synthetic_seismogram, seismogram_dataset, spectrum_matched_noise
from .features import detector_features
from .geochem import geochem_table
from .climate import climate_field, gutenberg_richter_magnitudes
from .events import inject_rare_events
from .spatial import multisite_table, event_station_table
from .hydro import well_table
from .tides import tide_gauge_series

__all__ = [
    "gnss_series",
    "degrade_series",
    "synthetic_seismogram",
    "seismogram_dataset",
    "spectrum_matched_noise",
    "detector_features",
    "geochem_table",
    "climate_field",
    "gutenberg_richter_magnitudes",
    "inject_rare_events",
    "multisite_table",
    "event_station_table",
    "well_table",
    "tide_gauge_series",
]
