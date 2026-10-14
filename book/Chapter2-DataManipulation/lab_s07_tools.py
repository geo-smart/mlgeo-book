"""Session 7 lab tools: exploration cards, one checker for people and agents, snapshot hashes.

The lab notebook in this folder imports this module. Everything a pair needs is here:

    import lab_s07_tools as lab
    card = lab.empty_card("A")             # the card to fill by hand: a dict, one entry per question
    lab.check_card("A", card)              # table of checks; never prints the true values
    before = lab.hash_snapshots()          # sha256 of every file in snapshots/, before the agent runs
    print(lab.make_prompt("A", "W-bare"))  # the fixed agent prompt, with the card attached
    agent = lab.load_agent_card("agent_A.json")
    lab.compare("A", card, agent)          # the same checks on both cards, side by side
    lab.compare_hashes(before, lab.hash_snapshots())
    lab.audit_agent_script("agent_A.py", "A")

Stations: "0" Nooksack River (worked example), "A" GHCN-Daily SeaTac, "B" ComCat Puget
Sound, "D" UW.RATT seismogram of the 2021 Chignik earthquake, "E" NCEP/NCAR Reanalysis 1
air temperature.

What this module holds and what it does not. The true values (shapes, counts, minima and
maxima, b-values, spectral peaks) are recomputed from the snapshot files every time the
checker runs; none is typed in here. Which surprising values are real and which are
errors is a judgment, not a computation: that list is stored only as salted SHA-256
hashes of dates, numbers and words (`_KEY`), so this file holds no readable answer. The
instructor's key text lives in a separate file outside the book repository and is shown
in class with `reveal_key`.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import importlib.util

import json
import math
import os
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SNAP = HERE / "snapshots"
RATT = "ratt_chignik_2021-07-29_"
FILES = {
    "0": "usgs_12213100_2021-11.csv",
    "A": "ghcnd_USW00024233_1948-2024.csv",
    "B": "comcat_puget_2000-2025.csv",
    "D": RATT + "UW.RATT.HHZ_0615-0815.mseed",
    "E": "ncep_air_2013-2014.nc",
}
STATIONS = {
    "0": "Nooksack River at Ferndale, WA (USGS 12213100), discharge every 15 min, November 2021",
    "A": "Seattle-Tacoma airport (GHCN-Daily USW00024233), daily weather, 1948 to 2024",
    "B": "Puget Sound earthquakes (USGS ComCat), 2000 to 2025",
    "D": "UW.RATT vertical seismogram, 2021-07-29 Mww 8.2 Chignik, Alaska earthquake",
    "E": "NCEP/NCAR Reanalysis 1 air temperature, North America, 6-hourly, 2013 to 2014",
}
CARD_KEYS = ("data_model", "reader", "shape", "units", "time_base", "missing_and_flags",
             "physical_bounds", "min_max_si", "real_or_error", "question10")
CFS_TO_M3S = 0.028316846592     # 1 cubic foot = 0.028316846592 m^3 (exact, international foot)
FULL_SCALE_24BIT = 2 ** 23      # a 24-bit digitizer stores counts from -2^23 to 2^23 - 1
DECLINE = "unanswerable"        # an answer that starts with this word is counted as declined, not as FAIL

_U = None  # placeholder for "not filled yet"


def _unit(*names):
    return {n: {"unit": _U, "source": _U} for n in names}


def _bounds(*names):
    return {n: {"low": _U, "high": _U, "reason": _U} for n in names}


def _minmax(*names):
    return {n: {"min": _U, "min_where": _U, "max": _U, "max_where": _U} for n in names}


_TIME = {"zone": _U, "start": _U, "end": _U, "step": _U, "regular": _U}
_ROE = {"real": {"where": _U, "value": _U, "reason": _U}, "error": {"where": _U, "value": _U, "reason": _U}}

TEMPLATES = {
    "0": {"units": _unit("discharge_cfs"),
          "time_base": dict(_TIME),
          "missing_and_flags": {"missing_values": _U, "rows_not_approved": _U,
                                "repeated_local_stamps": _U, "off_grid_stamps": _U},
          "physical_bounds": _bounds("discharge_m3_per_s"),
          "min_max_si": _minmax("discharge_m3_per_s"),
          "question10": {"skew_discharge": _U, "skew_log10_discharge": _U, "closer_to": _U}},
    "A": {"units": _unit("TMAX", "PRCP"),
          "time_base": dict(_TIME),
          "missing_and_flags": {"TMIN_missing": _U, "PRCP_missing": _U, "AWND_missing": _U,
                                "PRCP_trace_days": _U, "PRCP_trace_stored_as": _U, "quality_flag_O_days": _U},
          "physical_bounds": _bounds("TMAX_C", "TMIN_C", "PRCP_mm"),
          "min_max_si": _minmax("TMAX_C", "TMIN_C", "PRCP_mm"),
          "question10": {"hottest_day": _U, "jja_mean_C": _U, "jja_sd_C": _U, "z": _U}},
    "B": {"units": _unit("depth", "mag"),
          "time_base": dict(_TIME),
          "missing_and_flags": {"missing_values": _U, "negative_depths": _U, "mag_exactly_zero": _U,
                                "magType_labels": _U},
          "physical_bounds": _bounds("depth_km", "mag"),
          "min_max_si": _minmax("depth_km", "mag"),
          "question10": {"b_Mc1.0": _U, "b_Mc1.5": _U, "b_Mc2.0": _U, "bin_width": _U, "why_b_moves": _U}},
    "D": {"units": _unit("samples", "sensitivity"),
          "time_base": dict(_TIME),
          "missing_and_flags": {"n_traces": _U, "n_gaps": _U, "samples_at_full_scale": _U},
          "physical_bounds": _bounds("counts"),
          "min_max_si": _minmax("velocity_m_per_s"),
          "question10": {"peak_velocity_m_per_s": _U, "response_used": _U, "noise_peak_hz": _U,
                         "noise_peak_cause": _U}},
    "E": {"units": _unit("air"),
          "time_base": dict(_TIME, lat_order=_U, lon_range=_U),
          "missing_and_flags": {"dtype_on_disk": _U, "scale_factor": _U, "fill_value_attribute": _U,
                                "nan_after_decoding": _U},
          "physical_bounds": _bounds("air_K"),
          "min_max_si": _minmax("air_K"),
          "question10": {"cell_lat": _U, "cell_lon": _U, "mean_C": _U,
                         "spectral_peaks_cycles_per_day": _U, "shortest_period_h": _U}},
}


def _station(station):
    s = str(station).strip().upper()
    if s not in FILES:
        raise ValueError(f"unknown station {station!r}; use one of {sorted(FILES)}")
    return s


def empty_card(station):
    """Return the card to fill for one station: every answer is None until you replace it."""
    s = _station(station)
    t = TEMPLATES[s]
    card = {"station": s, "data_model": _U, "reader": _U, "shape": _U}
    for k in ("units", "time_base", "missing_and_flags", "physical_bounds", "min_max_si"):
        card[k] = copy.deepcopy(t[k])
    card["real_or_error"] = copy.deepcopy(_ROE)
    card["question10"] = copy.deepcopy(t["question10"])
    return card


# ---------------------------------------------------------------------------
# The card as Python source: the notebook shows it to the pair, the prompt gives it to the agent
# ---------------------------------------------------------------------------
BOUNDS_SOURCE = {
    "A": '''bounds_A = {
    "TMAX_C": {"low": None, "high": None, "reason": None},   # daily maximum temperature at SeaTac, °C
    "TMIN_C": {"low": None, "high": None, "reason": None},   # daily minimum temperature, °C
    "PRCP_mm": {"low": None, "high": None, "reason": None},  # precipitation in one day, mm
}''',
    "B": '''bounds_B = {
    "depth_km": {"low": None, "high": None, "reason": None},  # hypocentre depth, km, positive down
    "mag": {"low": None, "high": None, "reason": None},       # magnitude, any type in the file
}''',
    "D": '''bounds_D = {
    "counts": {"low": None, "high": None, "reason": None},    # raw samples, counts (a 24-bit digitizer)
}''',
    "E": '''bounds_E = {
    "air_K": {"low": None, "high": None, "reason": None},     # near-surface air temperature, K
}''',
}
_HEAD = '''    "station": "{s}",
    "data_model": None,   # what one {unit} is, in a few words: table, catalog, series, waveform or grid
    "reader": None,       # the line of code that read the file, as a string
'''
_ROE_SRC = '''    "real_or_error": {
        "real": {"where": None, "value": None, "reason": None},    # a surprising value that is real, with a physical reason
        "error": {"where": None, "value": None, "reason": None},   # an error in the values or in the metadata, with a reason
    },
'''
_BOUNDS_LINE = '''    "physical_bounds": bounds_{s},   # from the predict cell; revise bounds_{s} only with a reason that says what you saw
'''
CARD_SOURCE = {
    "A": 'card_A = {\n' + _HEAD.format(s="A", unit="row") + '''    "shape": None,        # [rows, columns]
    "units": {            # the unit of the stored number, and the file, attribute or document that states it
        "TMAX": {"unit": None, "source": None},
        "PRCP": {"unit": None, "source": None},
    },
    "time_base": {
        "zone": None,     # which clock or which day the dates refer to, and where that is stated
        "start": None,    # first date, "YYYY-MM-DD"
        "end": None,      # last date, "YYYY-MM-DD"
        "step": None,     # interval between rows, with its unit, e.g. "1 day"
        "regular": None,  # True if every interval is the same and no date is skipped, else False
    },
    "missing_and_flags": {
        "TMIN_missing": None,           # count of empty TMIN cells
        "PRCP_missing": None,           # count of empty PRCP cells
        "AWND_missing": None,           # count of empty AWND cells
        "PRCP_trace_days": None,        # count of days with measurement flag PRCP_M equal to "T"
        "PRCP_trace_stored_as": None,   # the number stored in PRCP on those days
        "quality_flag_O_days": None,    # count of days on which any quality flag (a _Q column) is "O"
    },
''' + _BOUNDS_LINE.format(s="A") + '''    "min_max_si": {                     # in °C and mm; where = the date, "YYYY-MM-DD"; flagged values in or out, your choice
        "TMAX_C": {"min": None, "min_where": None, "max": None, "max_where": None},
        "TMIN_C": {"min": None, "min_where": None, "max": None, "max_where": None},
        "PRCP_mm": {"min": None, "min_where": None, "max": None, "max_where": None},
    },
''' + _ROE_SRC + '''    "question10": {
        "hottest_day": None,   # "YYYY-MM-DD"
        "jja_mean_C": None,    # mean TMAX over all June, July and August days, °C
        "jja_sd_C": None,      # standard deviation of the same days, °C
        "z": None,             # (hottest TMAX - jja_mean_C) / jja_sd_C
    },
}''',
    "B": 'card_B = {\n' + _HEAD.format(s="B", unit="row") + '''    "shape": None,        # [rows, columns]
    "units": {            # the unit of the stored number (for depth, also which way is positive), and the document that states it
        "depth": {"unit": None, "source": None},
        "mag": {"unit": None, "source": None},
    },
    "time_base": {
        "zone": None,     # the time zone of the origin times
        "start": None,    # first origin time
        "end": None,      # last origin time
        "step": None,     # the interval between rows, or why there is none
        "regular": None,  # True if every interval is the same, else False
    },
    "missing_and_flags": {
        "missing_values": None,    # count of empty cells in the whole table
        "negative_depths": None,   # count of events with depth < 0
        "mag_exactly_zero": None,  # count of events with mag == 0.0
        "magType_labels": None,    # list of the magnitude types in the file
    },
''' + _BOUNDS_LINE.format(s="B") + '''    "min_max_si": {                # km and magnitude units; where = the date, "YYYY-MM-DD"
        "depth_km": {"min": None, "min_where": None, "max": None, "max_where": None},
        "mag": {"min": None, "min_where": None, "max": None, "max_where": None},
    },
''' + _ROE_SRC + '''    "question10": {
        "b_Mc1.0": None,       # Aki maximum-likelihood b for magnitudes >= 1.0
        "b_Mc1.5": None,       # same, >= 1.5
        "b_Mc2.0": None,       # same, >= 2.0
        "bin_width": None,     # the magnitude bin width you used: 0.1 or 0.01
        "why_b_moves": None,   # one sentence: why does b change with Mc?
    },
}''',
    "D": 'card_D = {\n' + _HEAD.format(s="D", unit="sample") + '''    "shape": None,        # [number of traces, samples per trace]
    "units": {            # the unit, and the file or attribute that states it
        "samples": {"unit": None, "source": None},       # what the stored numbers are
        "sensitivity": {"unit": None, "source": None},   # what the overall sensitivity converts to counts: the file's label, and what the sensor measures
    },
    "time_base": {
        "zone": None,     # the time zone of the start time
        "start": None,    # first sample, "YYYY-MM-DDTHH:MM:SS.ssssssZ"
        "end": None,      # last sample
        "step": None,     # interval between samples, with its unit
        "regular": None,  # True if every interval is the same, else False
    },
    "missing_and_flags": {
        "n_traces": None,                # traces in the file
        "n_gaps": None,                  # gaps between traces
        "samples_at_full_scale": None,   # samples with |counts| >= 2**23 - 1 (clipped)
    },
''' + _BOUNDS_LINE.format(s="D") + '''    "min_max_si": {                      # ground velocity in m/s; where = the UTC time of each
        "velocity_m_per_s": {"min": None, "min_where": None, "max": None, "max_where": None},
    },
''' + _ROE_SRC + '''    "question10": {
        "peak_velocity_m_per_s": None,   # peak |velocity|, m/s
        "response_used": None,           # the sensitivity alone, or which response stages
        "noise_peak_hz": None,           # largest spectral peak, 0.03 to 1 Hz, in the file ending 0415-0615
        "noise_peak_cause": None,        # what produces that peak
    },
}''',
    "E": 'card_E = {\n' + _HEAD.format(s="E", unit="cell") + '''    "shape": None,        # dims with sizes, e.g. {"time": ..., "lat": ..., "lon": ...}
    "units": {            # the unit, and the attribute or document that states it
        "air": {"unit": None, "source": None},
    },
    "time_base": {
        "zone": None,       # the time zone of the time coordinate
        "start": None,      # first time step
        "end": None,        # last time step
        "step": None,       # interval between time steps, with its unit
        "regular": None,    # True if every interval is the same, else False
        "lat_order": None,  # the order in which latitudes are stored
        "lon_range": None,  # [first, last] longitude as stored, and the convention
    },
    "missing_and_flags": {
        "dtype_on_disk": None,          # the integer type stored in the file
        "scale_factor": None,           # the scale_factor attribute
        "fill_value_attribute": None,   # the _FillValue attribute, or "none"
        "nan_after_decoding": None,     # count of NaN after xr.open_dataset decodes the file
    },
''' + _BOUNDS_LINE.format(s="E") + '''    "min_max_si": {                     # K; where = the date, "YYYY-MM-DD", and the place
        "air_K": {"min": None, "min_where": None, "max": None, "max_where": None},
    },
''' + _ROE_SRC + '''    "question10": {
        "cell_lat": None,                        # latitude of the grid cell nearest SeaTac
        "cell_lon": None,                        # its longitude, as stored
        "mean_C": None,                          # the 2013-2014 mean at that cell, °C
        "spectral_peaks_cycles_per_day": None,   # the two largest peaks, cycles per day
        "shortest_period_h": None,               # shortest period the 6-hourly series can resolve, h
    },
}''',
}


def agent_card_text(station):
    """The card the agent receives: the bounds block written out in place of the variable."""
    s = _station(station)
    body = BOUNDS_SOURCE[s].split("=", 1)[1].strip()
    body = "\n".join(("    " + ln if i else ln) for i, ln in enumerate(body.splitlines()))
    txt = re.sub(rf'"physical_bounds": bounds_{s},[^\n]*',
                 lambda m: f'"physical_bounds": {body},   # set from physics before reading the data, each with a reason',
                 CARD_SOURCE[s])
    assert f"bounds_{s}" not in txt.split("=", 1)[1], s
    return "card = " + txt.split("=", 1)[1].strip()


# ---------------------------------------------------------------------------
# Raw-data integrity
# ---------------------------------------------------------------------------

def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of(path):
    """SHA-256 of one file, as hex."""
    return _sha256(path)


def hash_snapshots(folder=SNAP):
    """Return {path relative to the folder: sha256} for every file in the snapshots folder and its subfolders."""
    folder = Path(folder)
    out = {}
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if p.is_file() and not any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            out[rel.as_posix()] = _sha256(p)
    return out


def compare_hashes(before, after):
    """Compare two hash_snapshots() results; print one summary line and return a table per file."""
    rows = []
    for name in sorted(set(before) | set(after)):
        b, a = before.get(name), after.get(name)
        status = ("unchanged" if a == b else "added" if b is None else "removed" if a is None else "CHANGED")
        rows.append({"file": name, "before": (b or "")[:12], "after": (a or "")[:12], "status": status})
    table = pd.DataFrame(rows, columns=["file", "before", "after", "status"])
    bad = table[table["status"] != "unchanged"]
    if bad.empty:
        print(f"all {len(table)} snapshot files unchanged")
    else:
        print(f"{len(bad)} of {len(table)} snapshot files differ: " + ", ".join(f"{r.file} ({r.status})" for r in bad.itertuples()))
    return table


# ---------------------------------------------------------------------------
# Truth, recomputed from the files at run time (only what the checks need)
# ---------------------------------------------------------------------------

def _fmt_td(td):
    sec = pd.Timedelta(td).total_seconds()
    for unit, n in (("day", 86400), ("h", 3600), ("min", 60)):
        if sec >= n and sec % n == 0:
            return f"{sec / n:g} {unit}"
    return f"{sec:g} s"


def _where_ts(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


@lru_cache(maxsize=None)
def _truth_0():
    u = pd.read_csv(SNAP / FILES["0"])
    iv = u[u["product"] == "instantaneous"]
    q = iv["discharge_cfs"] * CFS_TO_M3S
    t = pd.to_datetime(iv["time_local"])
    off = int(((t.dt.minute % 15) != 0).sum())
    i_min, i_max = q.idxmin(), q.idxmax()
    from scipy import stats
    mn_w = iv.loc[i_min, "time_local"] + " " + iv.loc[i_min, "tz"]
    mx_w = iv.loc[i_max, "time_local"] + " " + iv.loc[i_max, "tz"]
    return {
        "shape": [list(u.shape)],
        "missing_values": int(u.isna().sum().sum()),
        "rows_not_approved": int((u["approval"] != "A").sum()),
        "repeated_local_stamps": [int(iv["time_local"].duplicated().sum()),
                                  int(iv["time_local"].duplicated(keep=False).sum())],
        "repeated_day": sorted(set(iv.loc[iv["time_local"].duplicated(), "time_local"].str[:10])),
        "off_grid_stamps": off,
        "off_grid_where": iv.loc[(t.dt.minute % 15) != 0, "time_local"].tolist(),
        "n_iv": len(iv), "n_dv": int((u["product"] == "daily_mean").sum()),
        "start": pd.Timestamp(iv["time_local"].iloc[0]), "end": pd.Timestamp(iv["time_local"].iloc[-1]),
        "step": pd.Timedelta("15min"), "regular": False,
        "discharge_m3_per_s": {"min": float(q.min()), "max": float(q.max()), "min_where": mn_w, "max_where": mx_w,
                               "keep": (float(q.min()), float(q.max())), "keep_where": (mn_w, mx_w)},
        "peak_cfs": float(iv["discharge_cfs"].max()),
        "skew": [float(stats.skew(iv["discharge_cfs"])), float(stats.skew(u["discharge_cfs"]))],
        "skew_log": [float(stats.skew(np.log10(iv["discharge_cfs"]))), float(stats.skew(np.log10(u["discharge_cfs"])))],
    }


@lru_cache(maxsize=None)
def _truth_A():
    g = pd.read_csv(SNAP / FILES["A"], parse_dates=["date"])
    flagged = (g["TMAX_Q"] == "O") | (g["TMIN_Q"] == "O")
    out = {"shape": [list(g.shape)], "start": g["date"].iloc[0], "end": g["date"].iloc[-1],
           "step": pd.Timedelta("1D"),
           "regular": bool((g["date"].diff().dropna() == pd.Timedelta("1D")).all()),
           "TMIN_missing": int(g["TMIN"].isna().sum()), "PRCP_missing": int(g["PRCP"].isna().sum()),
           "AWND_missing": int(g["AWND"].isna().sum()),
           "PRCP_trace_days": int((g["PRCP_M"] == "T").sum()),
           "PRCP_trace_stored_as": sorted(g.loc[g["PRCP_M"] == "T", "PRCP"].unique().tolist()),
           "quality_flag_O_days": int(flagged.sum())}
    day = lambda i: g.loc[i, "date"].strftime("%Y-%m-%d")  # noqa: E731
    for col, var in (("TMAX", "TMAX_C"), ("TMIN", "TMIN_C"), ("PRCP", "PRCP_mm")):
        x = g[col] / 10
        keep = x[g[col + "_Q"] != "O"]          # the same column without its own O-flagged values
        out[var] = {"min": float(x.min()), "max": float(x.max()),
                    "min_where": day(x.idxmin()), "max_where": day(x.idxmax()),
                    "keep": (float(keep.min()), float(keep.max())),
                    "keep_where": (day(keep.idxmin()), day(keep.idxmax()))}
    out["PRCP_mm"]["min_where"] = None   # many dry days share the minimum
    out["PRCP_mm"]["keep_where"] = (None, out["PRCP_mm"]["keep_where"][1])
    jja = g.loc[g["date"].dt.month.isin([6, 7, 8]), "TMAX"] / 10
    hot = g.loc[g["TMAX"].idxmax()]
    out["q10"] = {"hottest_day": hot["date"].strftime("%Y-%m-%d"), "TMAX_C": hot["TMAX"] / 10,
                  "jja_mean_C": float(jja.mean()), "jja_sd_C": float(jja.std()), "n_jja": int(jja.size),
                  "z": float((hot["TMAX"] / 10 - jja.mean()) / jja.std())}
    return out


def _aki_variants(mag, mc):
    """Aki (1965) b from the mean magnitude, with and without Utsu's half-bin correction, two bin widths."""
    vals = []
    for dm in (0.1, 0.01):
        m = mag[mag >= mc - 1e-9]
        vals.append(math.log10(math.e) / (m.mean() - (mc - dm / 2)))
        vals.append(math.log10(math.e) / (m.mean() - mc))
        r = mag.round(int(round(-math.log10(dm))))
        r = r[r >= mc - 1e-9]
        vals.append(math.log10(math.e) / (r.mean() - (mc - dm / 2)))
        vals.append(math.log10(math.e) / (r.mean() - mc))
    return vals


@lru_cache(maxsize=None)
def _truth_B():
    c = pd.read_csv(SNAP / FILES["B"], parse_dates=["time"])
    t = c["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    out = {"shape": [list(c.shape)], "start": t.iloc[0], "end": t.iloc[-1], "regular": False,
           "missing_values": int(c.isna().sum().sum()),
           "negative_depths": int((c["depth"] < 0).sum()),
           "mag_exactly_zero": int((c["mag"] == 0).sum()),
           "magType_counts": c["magType"].value_counts().to_dict()}
    for col, var in (("depth", "depth_km"), ("mag", "mag")):
        x = c[col]
        out[var] = {"min": float(x.min()), "max": float(x.max()),
                    "min_where": t[x.idxmin()].strftime("%Y-%m-%d"), "max_where": t[x.idxmax()].strftime("%Y-%m-%d"),
                    "keep": (float(x.min()), float(x.max()))}
        out[var]["keep_where"] = (out[var]["min_where"], out[var]["max_where"])
    out["mag"]["min_where"] = None   # hundreds of events share M 0.0
    out["mag"]["keep_where"] = (None, out["mag"]["max_where"])
    out["b"] = {}
    for mc in (1.0, 1.5, 2.0):
        v = _aki_variants(c["mag"], mc)
        m = c["mag"][c["mag"] >= mc - 1e-9]
        out["b"][mc] = {"n": int(m.size), "b_0.1": math.log10(math.e) / (m.mean() - (mc - 0.05)),
                        "b_0.01": math.log10(math.e) / (m.mean() - (mc - 0.005)),
                        "window": (min(v) - 0.02, max(v) + 0.02)}
    return out


@lru_cache(maxsize=None)
def _truth_D():
    import obspy
    from scipy import signal
    st = obspy.read(str(SNAP / FILES["D"]))
    tr = st[0]
    inv = obspy.read_inventory(str(SNAP / (RATT + "UW.RATT.stationxml.xml")))
    sens = inv.select(channel="HHZ")[0][0][0].response.instrument_sensitivity
    x = tr.data
    t0 = tr.stats.starttime
    i_max, i_min = int(np.argmax(x)), int(np.argmin(x))
    noise = obspy.read(str(SNAP / (RATT + "UW.RATT.HHZ_0415-0615.mseed")))[0]
    y = noise.data.astype(float) - noise.data.mean()
    f, p = signal.welch(y, fs=noise.stats.sampling_rate, nperseg=2 ** 15)
    band = (f >= 0.03) & (f <= 1.0)
    f_peak = float(f[band][np.argmax(p[band])])
    peak_v = float(np.abs(x).max() / sens.value)
    return {
        "shape": [[tr.stats.npts], [len(st), tr.stats.npts]], "npts": tr.stats.npts,
        "dtype": str(x.dtype), "fs": tr.stats.sampling_rate,
        "start": pd.Timestamp(t0.datetime), "end": pd.Timestamp(tr.stats.endtime.datetime),
        "step": pd.Timedelta(seconds=tr.stats.delta), "regular": True,
        "n_traces": len(st), "n_gaps": len(st.get_gaps()),
        "samples_at_full_scale": int((np.abs(x) >= FULL_SCALE_24BIT - 1).sum()),
        "peak_counts": int(np.abs(x).max()), "peak_fraction_full_scale": float(np.abs(x).max() / FULL_SCALE_24BIT),
        "max_counts": int(x.max()), "min_counts": int(x.min()),
        "max_where": _where_ts((t0 + i_max * tr.stats.delta).datetime),
        "min_where": _where_ts((t0 + i_min * tr.stats.delta).datetime),
        "sensitivity": float(sens.value),
        "velocity_m_per_s": {"min": float(x.min() / sens.value), "max": float(x.max() / sens.value),
                             "min_window": (-1.06 * abs(x.min()) / sens.value, -0.84 * abs(x.min()) / sens.value),
                             "max_window": (0.96 * x.max() / sens.value, 1.12 * x.max() / sens.value),
                             "where": ("06:27", "06:28")},
        "peak_velocity": peak_v, "peak_velocity_window": (0.96 * peak_v, 1.12 * peak_v),
        "noise_peak_hz": f_peak, "noise_window": (f_peak - 0.025, f_peak + 0.025),
    }


@lru_cache(maxsize=None)
def _truth_E():
    import xarray as xr
    path = SNAP / FILES["E"]
    with xr.open_dataset(path) as ds, xr.open_dataset(path, mask_and_scale=False) as raw:
        a = ds["air"].load()
        r = raw["air"]
        lat, lon, time = ds["lat"].values, ds["lon"].values, ds["time"].values
        i_min = np.unravel_index(int(np.argmin(a.values)), a.shape)
        i_max = np.unravel_index(int(np.argmax(a.values)), a.shape)
        wrong = a.sel(lat=47.6, lon=-122.3, method="nearest")
        cell = a.sel(lat=47.4447, lon=360 - 122.3144, method="nearest")
        x = cell.values - cell.values.mean()
        dt_days = float(np.diff(time[:2])[0] / np.timedelta64(1, "D"))
        f = np.fft.rfftfreq(x.size, d=dt_days)
        power = np.abs(np.fft.rfft(x)) ** 2
        top = [float(f[k]) for k in np.argsort(power[1:])[::-1][:3] + 1]
        mn_w, mx_w = f"{pd.Timestamp(time[i_min[0]]):%Y-%m-%d}", f"{pd.Timestamp(time[i_max[0]]):%Y-%m-%d}"
        return {
            "dims": {k: int(v) for k, v in a.sizes.items()}, "shape": [[int(n) for n in a.shape]],
            "start": pd.Timestamp(time[0]), "end": pd.Timestamp(time[-1]),
            "step": pd.Timedelta(np.diff(time).min()),
            "regular": bool(np.unique(np.diff(time)).size == 1),
            "lat_first_last": (float(lat[0]), float(lat[-1])), "lon_range": (float(lon.min()), float(lon.max())),
            "dtype_on_disk": str(r.dtype), "scale_factor": float(r.attrs.get("scale_factor", 1.0)),
            "add_offset": r.attrs.get("add_offset"), "fill_value_attribute": r.attrs.get("_FillValue"),
            "nan_after_decoding": int(a.isnull().sum()),
            "air_K": {"min": float(a.min()), "max": float(a.max()), "min_where": mn_w, "max_where": mx_w,
                      "keep": (float(a.min()), float(a.max())), "keep_where": (mn_w, mx_w)},
            "wrong_cell": (float(wrong.lat), float(wrong.lon), float(wrong.mean()) - 273.15),
            "cell": (float(cell.lat), float(cell.lon)), "cell_mean_C": float(cell.mean()) - 273.15,
            "slice_ascending_len": int(a.sel(lat=slice(40, 50)).sizes["lat"]),
            "top_freqs_cpd": top, "nyquist_cpd": 1 / (2 * dt_days), "shortest_period_h": 2 * dt_days * 24,
        }


_TRUTH = {"0": _truth_0, "A": _truth_A, "B": _truth_B, "D": _truth_D, "E": _truth_E}

# Limits set by physics or by the instrument, not by this file. A bound outside them is not a bound.
_LIMITS = {
    "discharge_m3_per_s": (0.0, 10_000.0),
    "TMAX_C": (-60.0, 60.0), "TMIN_C": (-60.0, 50.0), "PRCP_mm": (0.0, 1_000.0),
    "depth_km": (-10.0, 700.0), "mag": (-3.0, 10.0),
    "counts": (-FULL_SCALE_24BIT, FULL_SCALE_24BIT),
    "air_K": (150.0, 340.0),
}

# ---------------------------------------------------------------------------
# The key of real extremes and errors, as salted hashes of tokens from the card text
# ---------------------------------------------------------------------------
_SALT = "mlgeo-s07-2026|"
_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                       "dec"), 1)}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"


def _canon(x):
    """A number as plain decimal text, no sign, no exponent, no trailing zeros: 1234567, 0.00456, 12.5."""
    try:
        d = Decimal(repr(abs(float(x))))
    except (InvalidOperation, ValueError, OverflowError):
        return None
    if not d.is_finite():
        return None
    s = format(d.normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def _tokens(text):
    """Dates, times, numbers and words of a card entry, normalized; see the instructor key file for the scheme."""
    raw = str(text).replace("−", "-").replace("µ", "u").replace("μ", "u")
    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)          # InputUnits -> Input Units
    s = raw.lower().replace("_", " ")
    s = re.sub(r"<\s*0(?![\d.])", " below zero ", s)
    out = set()

    def date(y, m, d):
        try:
            out.add("d:" + dt.date(int(y), int(m), int(d)).isoformat())
        except ValueError:
            pass

    for y, m, d in re.findall(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s):
        date(y, m, d)
    for d, mon, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\s*,?\s*(\d{{4}})", s):
        date(y, _MONTHS[mon[:3]], d)
    for mon, d, y in re.findall(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})", s):
        date(y, _MONTHS[mon[:3]], d)
    s = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}", " ", s)
    for hh, mm, ss, frac in re.findall(r"(?<![\d.])(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?", s):
        out.add(f"t:{int(hh):02d}:{mm}")
        if ss:
            out.add(f"t:{int(hh):02d}:{mm}:{ss}")
        if frac and frac.strip("0"):
            out.add("f:" + frac.rstrip("0"))
    s = re.sub(r"(?<![\d.])\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?", " ", s)
    seq = []
    for m in re.finditer(r"[a-z]+|\d[\d,]*\.?\d*(?:e[-+]?\d+)?|\.\d+(?:e[-+]?\d+)?", s):
        tok = m.group()
        if tok[0].isalpha():
            out.add("w:" + tok)
            seq.append(tok)
            continue
        try:
            x = float(tok.replace(",", "").rstrip("."))
        except ValueError:
            continue
        c = _canon(x)
        if c is None:
            continue
        seq.append(c)
        out.add("n:" + c)
        for k in range(4):
            out.add(f"r{k}:" + _canon(round(abs(x), k)))
        for k in (2, 3):
            out.add(f"s{k}:" + _canon(float(f"{abs(x):.{k}g}")))
    for i in range(len(seq) - 1):
        out.add(f"b:{seq[i]} {seq[i + 1]}")
        if i + 2 < len(seq):
            out.add(f"g:{seq[i]} {seq[i + 1]} {seq[i + 2]}")
    return out


def _hashed(tokens):
    return {hashlib.sha256((_SALT + t).encode()).hexdigest()[:16] for t in tokens}


# Generated by the instructor's key file (lab_s07_key.py, not in the book): do not edit by hand.
_KEY = {'0': {'error': [{'all': [['4cfc183924b73df2', '7098fe2fdc252b51', '9a91a6e33a9bde9a', 'b22e9578d0de38b6',
                           'f004a673d518c02f']],
                  'override': False,
                  'unless': []}],
       'real': [{'all': [['5bb507241452b658', '7e06cb103cc415ba', '8573c2de44ee2102', '85b8f168bbbcb4cb',
                          'a9c849558d98c50f', 'ab48179f299bdeef', 'b8517d77a01b318f', 'e9e3073f86428599']],
                 'override': False,
                 'unless': []},
                {'all': [['03bf5336dc7d6837', '4e46d4b4d3ac5b5c', '50635ba076e8ca18', 'af8c22abfa1920a3',
                          'c1b9ec5083812017', 'c590674f356bd1b7', 'c718f7730983002c', 'ca3a143c8d1b0fdf',
                          'd481f1bbf4af291a', 'df647cc636dd2dbf', 'e50a9ba22fbfdd75', 'feb01b760886982a']],
                 'override': False,
                 'unless': []}]},
 'A': {'error': [{'all': [['307f027c9e1e2ae2', '4d4c9f7ffc14a34d']], 'override': False, 'unless': []},
                 {'all': [['233c5a9b8824bc2b', 'f1b20d274cab4d0c']], 'override': False, 'unless': []}],
       'real': [{'all': [['089a05fb47cfb1e7', '28264099faf96f37', 'c0a1d37651ef7e7f', 'd25929ba7b1cd703',
                          'd811f36ed567a055', 'fe5a2b5980792d3b']],
                 'override': False,
                 'unless': []},
                {'all': [['960a657cd6d89005', 'bf24d199b6f46e0f']], 'override': False, 'unless': []},
                {'all': [['1d539f30284e8c41', 'b3b95cf829136bad']], 'override': False, 'unless': []}]},
 'B': {'error': [{'all': [['44db745da3210440', '5f6278fb0d8e7fb0', '663be43e4f17f029', '8d8d3f5e46f262df',
                           '95c153d2c4e3b299', '9b19bd102fbee8d1', '9e64b67aa27d2458', 'c640dccd5b37498b',
                           'd65dc566794fc115', 'e7979c3fb6416581', 'ea86f45020bfe7fe']],
                  'override': False,
                  'unless': []},
                 {'all': [['5e66b77d3633b474', '6485f8b28ffa5012', '679b4a83941eeb6a'],
                          ['0666d6d62c28baec', '17de4ad5d208b995', '2dc26315e444ae22', '4893224433ccfdab',
                           '65c77849bde8d171', '7810b864ba89970f', 'a83588076c8a3fcd', 'c04d481d000e11c4',
                           'c35574df5392e02f', 'c677d628f1f23d72', 'c9047c4644ad9ba1', 'ca0fa7b09e383dbe',
                           'ca795bc74ed21ced', 'e52dd64e2741ebb2']],
                  'override': True,
                  'unless': []}],
       'real': [{'all': [['5e66b77d3633b474', '6485f8b28ffa5012', '679b4a83941eeb6a']],
                 'override': False,
                 'unless': []},
                {'all': [['0788f3678050070b', '08ecae5d17f3aedd', '20d14f86ea6d10d0', '7ec664a9ef2b8fcc',
                          '89b60b56813f403c', '932fc620ae646c43', 'a5e798710040a85e', 'b370337b8c186dc4',
                          'c1bb7a32ee648dbd', 'cd8603f68d951e98', 'eb7a9cc848bad7ac', 'ecb93f3062a70ca7',
                          'fb9e5fb121baaa34']],
                 'override': False,
                 'unless': []},
                {'all': [['4e5686156c3f5d9d', '7345cc7e4d586651', '9afed3386eb4d627', 'a5e798710040a85e',
                          'e4569744bc939e00', 'e747036890fd9e80', 'f0d89c31ca66b342', 'f8169e7ad806979f']],
                 'override': False,
                 'unless': []}],
       'veto_real': [{'all': [['5e66b77d3633b474', '6485f8b28ffa5012', '679b4a83941eeb6a'],
                              ['0666d6d62c28baec', '2dc26315e444ae22', '4893224433ccfdab', '5381f90e40822a7c',
                               '5b8bdcc82870ef85', '65c77849bde8d171', '8732afe2d1a28c31', '93db46dc363ec231',
                               'c04d481d000e11c4', 'c35574df5392e02f', 'c677d628f1f23d72', 'c9047c4644ad9ba1',
                               'ca795bc74ed21ced', 'dbcd611656236bd4']],
                      'override': False,
                      'unless': []}]},
 'D': {'error': [{'all': [['9bf7542e25c62957', 'd4d9d8dc548a0ceb']], 'override': False, 'unless': []},
                 {'all': [['14d8d11c6dd584b5', '75b0ee90d907a041', '818b953e7dd247ab', '8f4da9ba00de5da0']],
                  'override': False,
                  'unless': []},
                 {'all': [['0666d6d62c28baec', '63ce8537b863c5ff', '94f54485766b3f13', 'a5915905e1e54ab8',
                           'c35574df5392e02f', 'c677d628f1f23d72'],
                          ['e8a8d455215c30c4']],
                  'override': False,
                  'unless': []}],
       'q10_cause': [{'all': [['37b7a8c356232934', '79e66c450fba9093', 'a112a2b9adaebd4c']],
                      'override': False,
                      'unless': []}],
       'real': [{'all': [['008d0a403af039cf', '1fd5797ba8f4c20b', '2520278696e457f4', '4a7174ab80103e52',
                          '5bb3c18cb9a17b6d', '5dd575bae6d77830', '7ec2cd1325633b73', 'd4bcf2e74dc60b2e',
                          'd6746412096a738f', 'd74d88e551ed543e', 'de6f348e4917ff94', 'e944ed8ebb32246d']],
                 'override': False,
                 'unless': []},
                {'all': [['51b01c71082e4825', 'c98af5097209aa4e', 'd22f33708241e692', 'eb70b9d028da3d9e',
                          'fdad42503dc83e36']],
                 'override': False,
                 'unless': []},
                {'all': [['37b7a8c356232934', '79e66c450fba9093', 'a112a2b9adaebd4c']],
                 'override': False,
                 'unless': []}],
       'veto_real': [{'all': [['329126490f8e8a13', '732a471fce69b69c', 'c6939482c735cb03']],
                      'override': False,
                      'unless': ['56b4287d1027d878', '7ec2cd1325633b73', '9949c789438fdb1c',
                                 'e168dbfe2eaf8e3c']}]},
 'E': {'error': [{'all': [['0054c44656465b89', '4e006844f743764e', '67c465dbb91ef607', '78e536e342a96c51']],
                  'override': False,
                  'unless': []},
                 {'all': [['8db0935d89695fbd', '97f8fb1df00319a5']], 'override': False, 'unless': []}],
       'real': [{'all': [['32c27fbc32fffb6a', '353129233870e4b0', 'd77a5c5ffe178ce4', 'eef8eae290ddafde']],
                 'override': False,
                 'unless': []},
                {'all': [['48518b957df0ec99', '59ffee9a6a3dc915', '683c289518236385', 'cb907bd56b3bbe00',
                          'f37cec1ffa270865']],
                 'override': False,
                 'unless': []}]}}  # KEY-TABLE


def _match(item, H):
    return all(any(h in H for h in grp) for grp in item["all"]) and not any(h in H for h in item.get("unless", []))


def _classify(s, item, side):
    """Return (ok, note) for one real_or_error entry, from its where and value text."""
    text = " ".join(str(item.get(k, "")) for k in ("where", "value")) if isinstance(item, dict) else str(item)
    H = _hashed(_tokens(text))
    key = _KEY.get(s, {})
    real = any(_match(it, H) for it in key.get("real", []))
    err_plain = any(_match(it, H) for it in key.get("error", []) if not it.get("override"))
    err_override = any(_match(it, H) for it in key.get("error", []) if it.get("override"))
    veto = any(_match(it, H) for it in key.get("veto_real", []))
    if side == "real":
        ok = real and not err_plain and not veto
        if ok:
            return True, ""
        if real and (err_plain or veto):
            return False, "this entry matches an error in the key, or a claim the file contradicts: be specific"
        return False, "not a real extreme in the key"
    ok = err_override or (err_plain and not real)
    if ok:
        return True, ""
    if err_plain and real:
        return False, "matches more than one entry in the key: be specific"
    return False, "not an error in the key"


# ---------------------------------------------------------------------------
# Small helpers for the checks
# ---------------------------------------------------------------------------

class _Missing:
    pass


_MISSING = _Missing()


def _get(card, *path):
    v = card
    for p in path:
        if not isinstance(v, dict) or p not in v:
            return _MISSING
        v = v[p]
    return v


def _unfilled(v):
    return v is _MISSING or v is None or (isinstance(v, str) and v.strip() in ("", "...", "?", "TODO"))


def _declined(v):
    return isinstance(v, str) and v.strip().lower().startswith(DECLINE)


def _show(v, width=60):
    if v is _MISSING:
        return "(key missing)"
    s = v if isinstance(v, str) else json.dumps(v, default=str).replace("null", "None")
    return s if len(s) <= width else s[: width - 3] + "..."


_NUM = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _num(v):
    """A number from a number or from text such as '12.5 °C' or '1,234,567'. None if there is none."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        m = _NUM.search(v.replace("−", "-"))
        if m:
            try:
                return float(m.group().replace(",", ""))
            except ValueError:
                return None
    return None


def _nums(v):
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    if isinstance(v, dict):
        return [_num(x) for x in v.values()]
    if isinstance(v, str):
        v = re.sub(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",            # '1/365' -> its value
                   lambda m: repr(float(m.group(1)) / float(m.group(2))), v.replace("−", "-"))
        return [float(m.replace(",", "")) for m in _NUM.findall(v)]
    n = _num(v)
    return [] if n is None else [n]


def _bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if re.match(r"^(no|false|irregular|not regular|n)\b", s):
            return False
        if re.match(r"^(yes|true|regular|y)\b", s):
            return True
    return None


def _ts(v):
    s = re.sub(r"\s*\(?\b(PST|PDT|local( time)?|UTC|GMT)\)?\s*$", "", str(v).strip(), flags=re.I)
    try:
        t = pd.Timestamp(s.replace("Z", "+00:00") if s.endswith("Z") else s)
    except (ValueError, TypeError):
        return None
    if t is pd.NaT:
        return None
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


_STEP_WORDS = {"daily": "1D", "once a day": "1D", "one day": "1D", "hourly": "1h", "one hour": "1h",
               "6-hourly": "6h", "six-hourly": "6h", "6 hourly": "6h", "six hours": "6h"}


def _td(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return pd.Timedelta(seconds=float(v))
    s = str(v).strip().lower()
    for w, d in _STEP_WORDS.items():
        if s.startswith(w):
            return pd.Timedelta(d)
    try:
        return pd.Timedelta(s.replace("hours", "h").replace("hour", "h"))
    except (ValueError, TypeError):
        pass
    m = re.match(r"^\s*([\d.]+)\s*([a-z]+)", s)
    if m:
        try:
            return pd.Timedelta(float(m.group(1)), unit={"day": "D", "days": "D", "d": "D", "sec": "s"}.get(m.group(2), m.group(2)))
        except (ValueError, TypeError):
            return None
    return None


def _close(x, y, tol):
    return x is not None and y is not None and abs(x - y) <= tol


class _Rows:
    """Collects check rows: check | expected tolerance or rule | your value | pass | note."""

    def __init__(self):
        self.rows = []

    def add(self, check, rule, value, ok, note=""):
        if _unfilled(value):
            self.rows.append((check, rule, _show(value), "unfilled",
                              "key missing from the card" if value is _MISSING else "not filled: replace None with your answer"))
        elif _declined(value):
            self.rows.append((check, rule, _show(value), "declined", "declined with a reason: counted apart from FAIL"))
        else:
            self.rows.append((check, rule, _show(value), "pass" if ok else "FAIL", "" if ok else note))

    def composite(self, check, rule, parts, ok, note=""):
        """One row for an answer with several parts (dict name -> value); unfilled if any part is."""
        empty = [k for k, v in parts.items() if _unfilled(v)]
        shown = {k: (None if v is _MISSING else v) for k, v in parts.items()}
        if any(_declined(v) for v in parts.values()):
            self.rows.append((check, rule, _show(shown), "declined", "declined with a reason: counted apart from FAIL"))
        elif empty:
            self.rows.append((check, rule, _show(shown), "unfilled", "not filled: " + ", ".join(empty)))
        else:
            self.rows.append((check, rule, _show(shown), "pass" if ok else "FAIL", "" if ok else note))

    def number(self, check, value, truth, tol, rule, hints=()):
        x = _num(value)
        ok = _close(x, truth, tol)
        note = "not a number" if x is None else "outside the tolerance"
        if x is not None and not ok:
            for factor, offset, msg in hints:
                if _close(x * factor + offset, truth, tol * max(1.0, abs(factor))):
                    note = msg
                    break
        self.add(check, rule, value, ok, note)

    def regex(self, check, value, pattern, rule, note="does not match the rule", reject=None):
        ok = isinstance(value, str) and re.search(pattern, value, re.I) is not None
        if ok and reject and re.search(reject, value, re.I):
            ok = False
        self.add(check, rule, value, ok, note)

    def text(self, check, value, rule="a reason in words (at least 10 characters)", n=10):
        self.add(check, rule, value, isinstance(value, str) and len(value.strip()) >= n, "too short to be a reason")


def _check_common(R, s, card, T):
    R.regex("data_model", _get(card, "data_model"),
            {"0": r"series|table", "A": r"table|series", "B": r"catalog|table|event",
             "D": r"waveform|series|trace|seismogram", "E": r"grid|array|cube|raster"}[s],
            "names the data model (table, catalog, series, waveform or grid)")
    R.regex("reader", _get(card, "reader"),
            {"0": r"read_csv", "A": r"read_csv", "B": r"read_csv", "D": r"obspy\.read\(|\bread\(",
             "E": r"open_dataset|open_dataarray|open_mfdataset|Dataset\("}[s],
            "one line of code that calls the right reader for this format")
    v = _get(card, "shape")
    ok = False
    if not _unfilled(v) and not _declined(v):
        if s == "E" and isinstance(v, dict):
            ok = {str(k).lower(): _num(x) for k, x in v.items()} == {k: float(n) for k, n in T["dims"].items()}
        else:
            ok = any([float(n) for n in shape] == _nums(v) for shape in T["shape"])
    R.add("shape", "rows and columns, or dims with sizes, exactly as the file has them", v, ok,
          "does not match the file")


_UNIT_RULES = {
    # var: (patterns that must all match, pattern that must not match, rule shown to the student)
    "0": {"discharge_cfs": ([r"cfs|ft\s*\^?\s*3|ft³|cubic\s*f(ee|oo)t"], None, None)},
    "A": {"TMAX": ([r"tenth|0\.1|1/10|/\s*10", r"celsius|°\s*c\b|\bdeg(ree)?s?\s*c\b|degc|\bc\b"],
                   r"fahrenheit|°\s*f\b|\bdeg(ree)?s?\s*f\b|\bf\b", None),
          "PRCP": ([r"tenth|0\.1|1/10|/\s*10", r"\bmm\b|millimet"], r"inch", None)},
    "B": {"depth": ([r"\bkm\b|kilomet", r"down|below|deep"], r"positive[\s,]*up|up\s*(is\s*)?positive|elevation|altitude",
                    "unit and which way is positive, and where that is stated"),
          "mag": ([r"dimensionless|unitless|no unit|none|magnitude|\bm[dlhw]\b|log|magtype|type"],
                  r"\bkm\b|metre|meter|\bm/s\b|joule", None)},
    "D": {"samples": ([r"count"], None, None),
          "sensitivity": ([r"count", r"m\s*/\s*s|m\s*s\s*\^?\s*-1|m\.s|met(er|re)s? per second|m/sec|\bv\b|volt"], None,
                          "counts per unit of input (the file's label, or what the sensor measures), and where stated")},
    "E": {"air": ([r"\bk\b|degk|kelvin"], r"celsius|°\s*c\b|degc", None)},
}


def _check_units(R, s, card):
    for var, (pats, reject, rule) in _UNIT_RULES[s].items():
        unit, src = _get(card, "units", var, "unit"), _get(card, "units", var, "source")
        both = f"{unit} {src}" if s == "B" and var == "depth" else unit   # the direction may sit in either part
        ok_unit = isinstance(unit, str) and all(re.search(p, both, re.I) for p in pats)
        rejected = isinstance(unit, str) and reject is not None and re.search(reject, unit, re.I)
        ok_src = isinstance(src, str) and len(src.strip()) >= 3
        if rejected:
            note = "names a unit or direction that the stored values and their documentation do not support"
        elif not ok_unit:
            note = "does not match the rule for this unit"
        else:
            note = "name the file, attribute or document"
        R.composite(f"units: {var}", rule or "unit as stored in the file, and where that is stated",
                    {"unit": unit, "source": src}, ok_unit and ok_src and not rejected, note)


def _check_time(R, s, card, T):
    tb = lambda k: _get(card, "time_base", k)  # noqa: E731
    zone, reject = {
        "0": (r"pst|pdt|pacific|local|los.?angeles", None),
        "A": (r"local|\blst\b|\bpst\b|pacific|not stated|unstated|not given|unknown|\bnone\b|\bn/?a\b|"
              r"no (time ?zone|zone|clock)", None),   # a bare "UTC day" fails: no local or not-stated wording
        "B": (r"utc|gmt|\bz\b|universal", None), "D": (r"utc|gmt|\bz\b|universal", None),
        "E": (r"utc|gmt|\bz\b|universal", None)}[s]
    R.regex("time_base: zone", tb("zone"), zone, "the time zone or clock the stamps are in", reject=reject)
    tol = {"0": pd.Timedelta("1min"), "A": pd.Timedelta("1D"), "B": pd.Timedelta("1D"),
           "D": pd.Timedelta("0.5s"), "E": pd.Timedelta("1min")}[s]
    day_only = s in ("A", "B")
    for k in ("start", "end"):
        v = tb(k)
        t = _ts(v)
        if day_only:
            ok = t is not None and t.normalize() == pd.Timestamp(T[k]).normalize()
            rule = f"same day as the file's {'first' if k == 'start' else 'last'} {'row' if s == 'A' else 'event'}"
        else:
            ok = t is not None and abs(t - pd.Timestamp(T[k])) <= tol
            rule = (f"same instant as the file's {'first' if k == 'start' else 'last'} sample, "
                    f"within {_fmt_td(tol)}")
        R.add(f"time_base: {k}", rule, v, ok, "not a date" if t is None else "does not match the file")
    v = tb("step")
    if s == "B":
        R.regex("time_base: step", v, r"irreg|\bnone\b|event|varies|variable|\bn/?a\b|no fixed",
                "an event catalog has no fixed step: say so")
    else:
        d = _td(v)
        R.add("time_base: step", "the sampling interval, with its unit", v,
              d is not None and abs(d - T["step"]) <= pd.Timedelta("1ms"),
              "could not read it as a time interval" if d is None else "does not match the file")
    v = tb("regular")
    b = _bool(v)
    R.add("time_base: regular", "true or false, as the stamps actually are", v, b is not None and b == T["regular"],
          "answer true or false" if b is None else "does not match the stamps")
    if s == "E":
        R.regex("time_base: lat_order", tb("lat_order"), r"desc|decreas|north.{0,10}south|n\s*to\s*s|75.{0,10}15",
                "the order in which latitudes are stored")
        lo = _nums(tb("lon_range"))
        ok = len(lo) == 2 and _close(lo[0], T["lon_range"][0], 0.01) and _close(lo[1], T["lon_range"][1], 0.01)
        R.add("time_base: lon_range", "first and last longitude as stored, with the convention", tb("lon_range"), ok,
              "check the longitude convention" if len(lo) == 2 and _close(lo[0] + 360, T["lon_range"][0], 0.01)
              else "does not match the file")


def _check_flags(R, s, card, T):
    mf = lambda k: _get(card, "missing_and_flags", k)  # noqa: E731
    exact = "an exact count from the file"
    if s == "0":
        R.number("missing: missing_values", mf("missing_values"), T["missing_values"], 0, exact)
        R.number("missing: rows_not_approved", mf("rows_not_approved"), T["rows_not_approved"], 0, exact)
        v = mf("repeated_local_stamps")
        R.add("missing: repeated_local_stamps", "count of local stamps that appear twice (stamps or rows)", v,
              _num(v) in [float(n) for n in T["repeated_local_stamps"]], "does not match the file")
        R.number("missing: off_grid_stamps", mf("off_grid_stamps"), T["off_grid_stamps"], 0,
                 "count of stamps not on the :00, :15, :30, :45 grid")
    elif s == "A":
        for k in ("TMIN_missing", "PRCP_missing", "AWND_missing", "PRCP_trace_days", "quality_flag_O_days"):
            R.number(f"missing: {k}", mf(k), T[k], 0, exact)
        R.number("missing: PRCP_trace_stored_as", mf("PRCP_trace_stored_as"), T["PRCP_trace_stored_as"][0], 0,
                 "the value stored in PRCP on trace days")
    elif s == "B":
        for k in ("missing_values", "negative_depths", "mag_exactly_zero"):
            R.number(f"missing: {k}", mf(k), T[k], 0, exact)
        v = mf("magType_labels")
        labels = v if isinstance(v, (list, tuple, set)) else re.findall(r"[A-Za-z]+", v) if isinstance(v, str) else []
        R.add("missing: magType_labels", "the set of magnitude types in the file", v,
              {str(x).strip().lower() for x in labels} == {k.lower() for k in T["magType_counts"]},
              "does not match the set in the file")
    elif s == "D":
        for k in ("n_traces", "n_gaps", "samples_at_full_scale"):
            R.number(f"missing: {k}", mf(k), T[k], 0, exact)
    elif s == "E":
        R.regex("missing: dtype_on_disk", mf("dtype_on_disk"), rf"\b{T['dtype_on_disk']}\b|\bshort\b",
                "the integer type stored on disk")
        R.number("missing: scale_factor", mf("scale_factor"), T["scale_factor"], 1e-9, "the scale_factor attribute")
        v = mf("fill_value_attribute")
        if T["fill_value_attribute"] is None:
            R.regex("missing: fill_value_attribute", v, r"none|absent|^no|not (set|present|defined)|missing|n/?a|null",
                    "the _FillValue attribute, or the text 'none' if there is none")
        else:
            R.number("missing: fill_value_attribute", v, float(T["fill_value_attribute"]), 0, "the _FillValue attribute")
        R.number("missing: nan_after_decoding", mf("nan_after_decoding"), T["nan_after_decoding"], 0, exact)


def _hints(var, T):
    """(factor, offset, message): if value * factor + offset matches the truth, the message names the slip."""
    return {
        "discharge_m3_per_s": [(CFS_TO_M3S, 0, "looks like cubic feet per second: convert to m3/s")],
        "TMAX_C": [(0.1, 0, "looks like tenths of a degree: divide by 10"), (1, -273.15, "looks like kelvin")],
        "TMIN_C": [(0.1, 0, "looks like tenths of a degree: divide by 10"), (1, -273.15, "looks like kelvin")],
        "PRCP_mm": [(0.1, 0, "looks like tenths of a millimetre: divide by 10")],
        "air_K": [(0.01, 0, "looks like the stored integer: apply scale_factor"),
                  (1, 273.15, "looks like °C: the key asks for K")],
        "velocity_m_per_s": [(1 / T["sensitivity"], 0, "looks like counts: divide by the sensitivity")]
        if "sensitivity" in T else [],
    }.get(var, [])


def _check_bounds_minmax(R, s, card, T):
    rule_b = "low < high, keeps every real value, within physical limits, with a reason"
    for var in TEMPLATES[s]["physical_bounds"]:
        parts = {k: _get(card, "physical_bounds", var, k) for k in ("low", "high", "reason")}
        lo, hi = _num(parts["low"]), _num(parts["high"])
        keep = (T["min_counts"], T["max_counts"]) if var == "counts" else T[var]["keep"]
        floor, ceil = _LIMITS[var]
        notes = []
        if lo is None or hi is None or lo >= hi:
            notes.append("low and high must be numbers with low < high")
        else:
            if lo > keep[0] + 1e-9 or hi < keep[1] - 1e-9:
                notes.append("these bounds would throw away real values in the file")
            if lo < floor - 1e-9 or hi > ceil + 1e-9:
                notes.append("wider than physics or the instrument allows")
        if not (isinstance(parts["reason"], str) and len(parts["reason"].strip()) >= 10):
            notes.append("give a physical reason")
        R.composite(f"bounds: {var}", rule_b, parts, not notes, "; ".join(notes))

    for var in TEMPLATES[s]["min_max_si"]:
        tv = T[var]
        for i, side in enumerate(("min", "max")):
            v = _get(card, "min_max_si", var, side)
            w = _get(card, "min_max_si", var, f"{side}_where")
            x = _num(v)
            if s == "D":
                lo_w, hi_w = tv[f"{side}_window"]
                ok_v = x is not None and lo_w <= x <= hi_w
                wants = tv["where"]
                rule = f"{side} in m/s, from the sensitivity or the full response, and when (UTC)"
            else:
                tol = {"discharge_m3_per_s": 0.01 * abs(tv[side]), "air_K": 0.05, "depth_km": 0.05}.get(var, 0.05)
                # either the extreme of every stored value, or of the values without their quality flag
                options = [(tv[side], tv[f"{side}_where"]), (tv["keep"][i], tv["keep_where"][i])]
                hit = next(((val, wh) for val, wh in options if _close(x, val, tol)), None)
                ok_v = hit is not None
                where = (hit or options[0])[1]
                wants = None if where is None else (where[:10],)
                rule = f"{side} in SI units within {'1%' if var == 'discharge_m3_per_s' else tol}" + \
                       (", and where (date YYYY-MM-DD)" if tv[f"{side}_where"] is not None else "")
            ok_w = wants is None or (isinstance(w, str) and any(x_ in w for x_ in wants))
            note = ""
            if not ok_v:
                note = "not a number" if x is None else "value outside the tolerance"
                for factor, offset, msg in _hints(var, T):
                    y = None if x is None else x * factor + offset
                    if y is not None and ((s == "D" and lo_w <= y <= hi_w) or
                                          (s != "D" and _close(y, tv[side], max(0.05, 0.01 * abs(tv[side]))))):
                        note = msg
            elif not ok_w:
                note = "where does not match (give the date YYYY-MM-DD, or the UTC time)"
            parts = {"value": v} if wants is None and (s == "D" or tv[f"{side}_where"] is None) else {"value": v, "where": w}
            R.composite(f"min_max: {var} {side}", rule, parts, ok_v and ok_w, note)


def _check_real_error(R, s, card):
    for side in ("real", "error"):
        parts = {k: _get(card, "real_or_error", side, k) for k in ("where", "value", "reason")}
        rule = ("a surprising value that is real: where, value and a physical reason" if side == "real"
                else "a value or record that is an error: where, value and a reason")
        if s == "0" and side == "error":
            rule = "a value or record that is an error, or the text 'none' with a reason"
            parts["value"] = parts["value"] if not _unfilled(parts["value"]) else "-"   # "none" needs no value
        hit, note = _classify(s, parts, side)
        if hit and not (isinstance(parts["reason"], str) and len(parts["reason"].strip()) >= 10):
            hit, note = False, "give a reason (at least 10 characters)"
        R.composite(f"real_or_error: {side}", rule, parts, hit, note)


def _check_q10(R, s, card, T):
    q = lambda k: _get(card, "question10", k)  # noqa: E731
    if s == "0":
        for k, vals in (("skew_discharge", T["skew"]), ("skew_log10_discharge", T["skew_log"])):
            x = _num(q(k))
            R.add(f"q10: {k}", "sample skewness, 15-min values (daily means in or out), within 0.1", q(k),
                  any(_close(x, v, 0.1) for v in vals), "outside the tolerance")
        R.regex("q10: closer_to", q("closer_to"), r"log", "normal or log-normal")
    elif s == "A":
        R.regex("q10: hottest_day", q("hottest_day"), re.escape(T["q10"]["hottest_day"]), "date, YYYY-MM-DD")
        R.number("q10: jja_mean_C", q("jja_mean_C"), T["q10"]["jja_mean_C"], 0.1,
                 "mean June-August TMAX in °C, within 0.1", _hints("TMAX_C", T))
        R.number("q10: jja_sd_C", q("jja_sd_C"), T["q10"]["jja_sd_C"], 0.05,
                 "standard deviation in °C, within 0.05", [(0.1, 0, "looks like tenths of a degree: divide by 10")])
        R.number("q10: z", q("z"), T["q10"]["z"], 0.05, "standard deviations above the mean, within 0.05")
    elif s == "B":
        bs = []
        for mc in (1.0, 1.5, 2.0):
            k = f"b_Mc{mc}"
            x = _num(q(k))
            bs.append(x)
            lo, hi = T["b"][mc]["window"]
            R.add(f"q10: {k}", "Aki maximum-likelihood b at this Mc, within the spread of standard bin choices",
                  q(k), x is not None and lo <= x <= hi, "outside the range of standard estimates")
        if all(x is not None for x in bs):
            R.add("q10: b grows with Mc", "b at Mc 1.0 < 1.5 < 2.0, as the file gives", bs, bs[0] < bs[1] < bs[2],
                  "does not follow the file")
        R.add("q10: bin_width", "the magnitude bin width you used (0.1 or 0.01)", q("bin_width"),
              any(_close(_num(q("bin_width")), w, 1e-9) for w in (0.1, 0.01)), "use 0.1 or 0.01, the precisions in the file")
        R.text("q10: why_b_moves", q("why_b_moves"), "a reason in words (at least 15 characters)", 15)
    elif s == "D":
        x = _num(q("peak_velocity_m_per_s"))
        lo, hi = T["peak_velocity_window"]
        R.add("q10: peak_velocity_m_per_s", "peak |velocity| in m/s, sensitivity or full response", q("peak_velocity_m_per_s"),
              x is not None and lo <= x <= hi,
              "looks like counts: divide by the sensitivity" if x is not None and lo <= x / T["sensitivity"] <= hi
              else "outside the tolerance")
        R.text("q10: response_used", q("response_used"), "which response stages or sensitivity you used")
        x = _num(q("noise_peak_hz"))
        lo, hi = T["noise_window"]
        R.add("q10: noise_peak_hz", "frequency of the largest peak, 0.03 to 1 Hz, file ending 0415-0615, within 0.025 Hz",
              q("noise_peak_hz"), x is not None and lo <= x <= hi,
              "outside the tolerance (check the segment length: it sets the frequency resolution)")
        v = q("noise_peak_cause")
        ok = isinstance(v, str) and any(_match(it, _hashed(_tokens(v))) for it in _KEY.get("D", {}).get("q10_cause", []))
        R.add("q10: noise_peak_cause", "the physical cause of that peak", v, ok, "does not name the cause in the key")
    elif s == "E":
        R.number("q10: cell_lat", q("cell_lat"), T["cell"][0], 0.01, "latitude of the grid cell nearest SeaTac")
        x = _num(q("cell_lon"))
        R.add("q10: cell_lon", "longitude of that cell (degrees east, or negative west)", q("cell_lon"),
              _close(x, T["cell"][1], 0.01) or _close(x, T["cell"][1] - 360, 0.01), "not the nearest cell")
        R.number("q10: mean_C", q("mean_C"), T["cell_mean_C"], 0.05, "two-year mean at that cell in °C, within 0.05",
                 [(1, -273.15, "looks like kelvin: the key asks for °C")])
        v = q("spectral_peaks_cycles_per_day")
        xs = [x for x in _nums(v) if x is not None]
        annual = any(1 / 450 <= x <= 1 / 300 for x in xs)
        daily = any(abs(x - 1.0) <= 0.02 for x in xs)
        note = ("these look like periods in days: give frequencies in cycles per day"
                if any(300 <= x <= 450 for x in xs) else "does not match the two largest peaks")
        R.add("q10: spectral_peaks_cycles_per_day", "the two largest peaks, in cycles per day", v,
              len(xs) >= 2 and annual and daily, note)
        R.number("q10: shortest_period_h", q("shortest_period_h"), T["shortest_period_h"], 0.01,
                 "shortest resolvable period (two samples), in hours")


_CHECK_LOG = {}   # station -> times the pair ran check_card on a card (quiet runs from compare are not counted)
STATUSES = ("pass", "FAIL", "declined", "unfilled")


def _counts(col):
    return {k: int((col == k).sum()) for k in STATUSES}


def check_card(station, card, quiet=False):
    """Grade a card against the snapshot files. Returns a table; never prints the true values."""
    s = _station(station)
    if not isinstance(card, dict):
        raise TypeError("a card is a dict: start from lab.empty_card(station)")
    T = _TRUTH[s]()
    R = _Rows()
    if "station" in card and str(card["station"]).strip().upper() != s:
        R.rows.append(("station", "card is for this station", _show(card["station"]), "FAIL",
                       f"this card says station {card['station']}, you asked to check {s}"))
    _check_common(R, s, card, T)
    _check_units(R, s, card)
    _check_time(R, s, card, T)
    _check_flags(R, s, card, T)
    _check_bounds_minmax(R, s, card, T)
    _check_real_error(R, s, card)
    _check_q10(R, s, card, T)
    extra = sorted(set(card) - set(CARD_KEYS) - {"station", "tool", "notes"})
    for k in extra:
        R.rows.append((k, "not a card key", _show(card[k]), "ignored", "this key is not graded"))
    table = pd.DataFrame(R.rows, columns=["check", "expected tolerance or rule", "your value", "pass", "note"])
    if not quiet:
        _CHECK_LOG.setdefault(s, []).append(dt.datetime.now())
        c = _counts(table["pass"])
        print(f"station {s}: {c['pass']} of {sum(c.values())} checks pass, {c['FAIL']} fail, "
              f"{c['declined']} declined, {c['unfilled']} unfilled")
    return table


def check_log(station):
    """How many times check_card ran on this station in this session, and when last."""
    times = _CHECK_LOG.get(_station(station), [])
    return {"n": len(times), "last": f"{times[-1]:%H:%M}" if times else None}


def _summary(col):
    c = _counts(col)
    filled = c["pass"] + c["FAIL"] + c["declined"]
    return c, filled


def compare(station, hand_card, agent_card):
    """Run the same checks on the pair's card and the agent's card; return them side by side."""
    s = _station(station)
    h = check_card(s, hand_card, quiet=True)
    a = check_card(s, agent_card, quiet=True)
    out = pd.DataFrame({"check": h["check"], "expected tolerance or rule": h["expected tolerance or rule"],
                        "hand value": h["your value"], "hand": h["pass"]})
    a = a.rename(columns={"your value": "agent value", "pass": "agent"})[["check", "agent value", "agent"]]
    order = list(out["check"]) + [c for c in a["check"] if c not in set(out["check"])]
    out = out.merge(a, on="check", how="outer").set_index("check").loc[order].reset_index()
    graded = out[~out["hand"].isin(["ignored"]) & ~out["agent"].isin(["ignored"]) & ~out["check"].isin(["station"])]
    n = len(graded)
    diff = int((graded["hand"] != graded["agent"]).sum())
    for who in ("hand", "agent"):
        c, filled = _summary(graded[who])
        print(f"station {s}, {who}: {c['pass']} pass of {filled} filled ({n} checks; {c['FAIL']} FAIL, "
              f"{c['declined']} declined, {c['unfilled']} unfilled)")
    print(f"{diff} checks where hand and agent differ")
    return out


def load_agent_card(path):
    """Read the card an agent saved as JSON. Accepts {"card": {...}} or the card itself."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the agent's script should save its card there as JSON")
    try:
        card = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON ({e}); open it and look") from None
    if isinstance(card, dict) and isinstance(card.get("card"), dict):
        card = card["card"]
    if not isinstance(card, dict):
        raise ValueError(f"{path} holds a {type(card).__name__}, not a card dict")
    missing = [k for k in CARD_KEYS if k not in card]
    if missing:
        print(f"note: the agent's card has no {', '.join(missing)}; those checks will show as unfilled")
    return card


# ---------------------------------------------------------------------------
# Reading the agent's script
# ---------------------------------------------------------------------------
_WRITE_METHODS = {"to_csv", "to_netcdf", "to_parquet", "to_json", "to_excel", "to_hdf", "to_zarr",
                  "write_text", "write_bytes", "unlink", "rename", "replace", "chmod", "rmdir", "touch", "mkdir",
                  "savez", "save", "savetxt", "tofile"}
_WRITE_FUNCS = {"shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.move", "shutil.rmtree", "os.remove",
                "os.unlink", "os.rename", "os.replace", "os.chmod", "os.rmdir", "os.makedirs", "os.mkdir"}


def _call_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def audit_agent_script(path, station):
    """Flag imports of this module, every file write (and writes into snapshots/), and key values typed in.

    Comments and docstrings are skipped. Prints line numbers only, so it does not reveal the key.
    The hash comparison before and after the run is the authority on whether raw data changed.
    """
    s = _station(station)
    src = Path(path).read_text()
    T = _TRUTH[s]()
    numbers = {
        "0": lambda: [T["peak_cfs"], T["shape"][0][0], T["n_iv"], T["discharge_m3_per_s"]["max"]],
        "A": lambda: [T["shape"][0][0], T["q10"]["TMAX_C"] * 10, T["q10"]["TMAX_C"], T["q10"]["z"],
                      T["q10"]["jja_mean_C"], T["PRCP_trace_days"], T["AWND_missing"]],
        "B": lambda: [T["shape"][0][0], T["negative_depths"], T["mag_exactly_zero"], T["b"][1.5]["b_0.1"]],
        "D": lambda: [T["peak_counts"], T["npts"], T["sensitivity"]],
        "E": lambda: [T["cell_mean_C"], T["wrong_cell"][2], T["air_K"]["min"], T["air_K"]["max"], T["dims"]["time"]],
    }[s]()
    key_hashes = {h for kind in _KEY.get(s, {}).values() for it in kind for grp in it["all"] for h in grp}
    found = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        table = pd.DataFrame([(e.lineno, f"the script does not parse: {e.msg}")], columns=["line", "issue"])
        print(f"1 issue in {Path(path).name}")
        return table
    docs = _docstring_nodes(tree)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            if any("lab_s07_tools" in (n or "") for n in names):
                found.append((line, "imports the checker"))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            seg = ast.get_source_segment(src, node) or ""
            mode = next((a.value for a in node.args[1:2] if isinstance(a, ast.Constant)), None)
            mode = next((k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)), mode)
            writes = (name.split(".")[-1] in _WRITE_METHODS or name in _WRITE_FUNCS
                      or (name in ("open", "io.open") and isinstance(mode, str) and re.search(r"[wax+]", mode)))
            if re.search(r"lab_s07_tools|reveal_key|_truth_|_KEY\b", seg) and not writes:
                found.append((line, "uses the checker or key"))
            if writes:
                found.append((line, "writes inside snapshots/" if "snapshots" in seg
                              else f"writes a file ({name}): check that it is agent_{s}.json"))
        elif isinstance(node, ast.Constant) and id(node) not in docs:
            v = node.value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if any(abs(v - n) <= 1e-3 * max(1.0, abs(n)) and abs(n) >= 2 for n in numbers):
                    found.append((line, "a number from the file typed into the code"))
                elif abs(v) >= 1 and len(str(v)) >= 3 and _hashed({t for t in _tokens(str(v)) if t[0] in "nr"}) & key_hashes:
                    found.append((line, "a value from the key typed into the code"))
            elif isinstance(v, str):
                if "lab_s07_tools" in v:
                    found.append((line, "names the checker"))
                dates = {t for t in _tokens(v) if t.startswith("d:")}
                if dates and _hashed(dates) & key_hashes:
                    found.append((line, "a date from the key typed into the code"))
    table = pd.DataFrame(sorted(set(found)), columns=["line", "issue"])
    print(f"{len(table)} item(s) to read in {Path(path).name}" if len(table) else f"no issues found in {Path(path).name}")
    return table


# ---------------------------------------------------------------------------
# Lab workflow: predictions, prompt, scores, scoreboard line
# ---------------------------------------------------------------------------
PREDICTIONS = {}
PRED_FILE = Path("lab_s07_predictions.json")   # beside your notebook copy; survives a kernel restart


def _leaves(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from _leaves(v)
    else:
        yield d


def stamp(station, *blocks):
    """Record the time of your predictions, in memory and in lab_s07_predictions.json. Every None must be replaced first."""
    s = str(station)
    empty = sum(v is None for b in blocks for v in _leaves(b))
    if empty:
        print(f"not filled yet: {empty} prediction(s) are still None")
        return
    PREDICTIONS[s] = {"time": f"{dt.datetime.now():%H:%M:%S}", "values": copy.deepcopy(list(blocks))}
    try:
        old = json.loads(PRED_FILE.read_text()) if PRED_FILE.exists() else {}
        old[s] = PREDICTIONS[s]
        PRED_FILE.write_text(json.dumps(old, indent=1, default=str))
    except (OSError, ValueError):
        pass
    print(f"station {s}: predictions recorded at {PREDICTIONS[s]['time']}. The record keeps these values; "
          f"if the data change your mind, revise bounds_{s} with a reason that says what you saw.")


def predictions(station):
    """The stamped predictions for a station, from memory or from lab_s07_predictions.json."""
    s = str(station)
    if s not in PREDICTIONS and PRED_FILE.exists():
        try:
            PREDICTIONS.update(json.loads(PRED_FILE.read_text()))
        except (OSError, ValueError):
            pass
    return PREDICTIONS.get(s)


def bounds_revised(station, card):
    """'yes' if the card's bounds differ from the stamped predictions, 'no' if not, 'no stamp' if none was recorded."""
    p = predictions(station)
    if not p or not p["values"]:
        return "no stamp"
    stamped = p["values"][0]
    now = card.get("physical_bounds", {}) if isinstance(card, dict) else {}
    same = all(_num(_get(now, v, k)) == _num(_get(stamped, v, k)) for v in stamped for k in ("low", "high"))
    return "no" if same else "yes"


def variant_for(netid):
    """W-bare or W-full, from the driver's UW NetID: a fixed coin flip, so no seat sheet is needed."""
    if not isinstance(netid, str) or not netid.strip() or netid.strip().upper() == "UWNETID":
        return None
    return "W-full" if int(hashlib.sha256(netid.strip().lower().encode()).hexdigest(), 16) % 2 == 0 else "W-bare"


ADD_ON = {
    "A": "In question10, report the hottest day and how many standard deviations it lies above the June "
         "to August mean of daily maximum temperature.",
    "B": "In question10, report Aki's b-value at magnitudes of completeness 1.0, 1.5 and 2.0, with the "
         "magnitude bin width you used and why.",
    "D": "In question10, report the peak ground velocity and the largest spectral peak of the noise record.",
    "E": "In question10, report the grid cell nearest SeaTac airport (47.4447 N, 122.3144 W), its mean, and the "
         "two largest spectral peaks.",
}
CHAT_MODE = "chat writes, we run"
PROMPT = """You are working in my clone of the course book repository, mlgeo-book.{where}
The data file is snapshots/{data} in that folder; treat it as read only.{scope}{context}

Write a Python script agent_{s}.py in that folder that reads the data file and saves agent_{s}.json in the \
same folder: one JSON object with exactly the keys and nesting of the card below, with every None replaced \
by your answer. The comments say what each entry means. {run}

At the top of the script, before any line reads the data, set physical_bounds from physics alone, each with \
its reason. If the data later make you change a bound, say so in its reason and name the value you saw.

Compute every other value from the files; do not type any number from the data into the script. {add_on}

Do not open, import or run lab_s07_tools.py or any notebook in the folder. Do not modify, move, copy or \
convert anything in snapshots/; write only agent_{s}.py and agent_{s}.json, and do not install packages. \
If an entry cannot be determined from the files, set it to a string that starts with "unanswerable:" and \
gives the reason; stopping and saying so is an acceptable answer.

{card}
"""
WHERE_AGENT = """ Work only in this folder:
    {folder}
If your working directory is not inside {repo}, stop and tell me to restart you there. Do not read or \
change any other repository on this computer."""
WHERE_CHAT = " The folder is book/Chapter2-DataManipulation."
RUN_AGENT = "Run the script from that folder with `pixi run python agent_{s}.py` and show me its full output."
RUN_CHAT = ("You cannot open or run my files. Below the card I paste what I printed from the file. I will run "
            "your script myself and paste its output back; do not write what the output will be.")
SCOPE_BARE = " Use only the data file: do not open any other file in snapshots/ or elsewhere in the repository."
CONTEXT = """

Before you start, read the context below and follow it as you would an AGENTS.md file.

Context:
- The provenance record of the data file is snapshots/{prov}. Read it first. It names the provider, \
the units of each column, the meaning of each flag, and links to the provider's documentation.
- The raw file is evidence. Never drop, fill, round or "clean" a value. If a value looks wrong, report \
it with a count and a reason.
- Give the source (file, attribute or document) of every unit you report.
- When a choice changes a number (units, bin width, which rows count), say which choice you made and why."""


def _prov_name(station):
    data = FILES[station]
    return (RATT[:-1] + ".provenance.json") if station == "D" else data.rsplit(".", 1)[0] + ".provenance.json"


def make_prompt(station, variant, mode=None, folder=None):
    """The fixed prompt for one pair. mode CHAT_MODE gives the chat-only version (the assistant cannot run code)."""
    s = _station(station)
    folder = Path(folder or Path.cwd()).resolve()
    repo = folder.parents[1] if folder.parent.name == "book" and len(folder.parents) > 1 else folder
    chat = mode == CHAT_MODE
    return PROMPT.format(
        where=WHERE_CHAT if chat else WHERE_AGENT.format(folder=folder, repo=repo),
        data=FILES[s], s=s, add_on=ADD_ON[s], card=agent_card_text(s),
        scope="" if variant == "W-full" else SCOPE_BARE,
        context=CONTEXT.format(prov=_prov_name(s)) if variant == "W-full" else "",
        run=(RUN_CHAT if chat else RUN_AGENT.format(s=s)))


def purple_printout(station, variant, folder=SNAP):
    """What a chat-only assistant cannot open itself: the head of the file, and the provenance for W-full."""
    s = _station(station)
    data = FILES[s]
    out = [f"First 30 lines of snapshots/{data}:"] + (Path(folder) / data).read_text().splitlines()[:30]
    if variant == "W-full":
        prov = _prov_name(s)
        out += ["", f"Contents of snapshots/{prov}:", (Path(folder) / prov).read_text()]
    return "\n".join(out)


GROUPS = {"S3": ("data_model", "reader", "shape", "time_base"), "S4": ("units",), "S5": ("missing",),
          "S6": ("bounds", "min_max", "real_or_error")}


def suggested_scores(station, card):
    """S3 to S6 from the checker, on the 0/1/2 scale of the scoring table. You confirm or override them."""
    t = check_card(station, card, quiet=True)
    out = {}
    for s, prefixes in GROUPS.items():
        rows = t[t["check"].str.startswith(prefixes)]
        n_ok = int((rows["pass"] == "pass").sum())
        n_bad = len(rows) - n_ok
        if s in ("S3", "S4"):
            score = 2 if n_bad == 0 else 1 if n_bad == 1 else 0
        elif s == "S5":
            score = 2 if n_bad == 0 else 1 if n_ok >= len(rows) / 2 else 0
        else:
            real_lost = ((rows["check"] == "real_or_error: real") & (rows["pass"] == "FAIL")).any() or \
                rows["note"].str.contains("throw away real values").any()
            score = 0 if real_lost else 2 if n_bad == 0 else 1
        out[s] = {"score": score, "evidence": f"{n_ok} of {len(rows)} checks pass"}
    return out


AUDIT_ITEMS = {
    "a_snapshots_unchanged": "Snapshot files unchanged (hash comparison, git status --short)",
    "b_no_data_numbers_typed": "No number seen in the data typed into the script, no import of the checker; "
                               "bounds typed from physics are allowed",
    "c_nothing_invented": "No invented column, unit, file, DOI or citation",
    "d_drops_reported": "Every dropped, filled or 'cleaned' value reported, with counts in and out",
    "e_claims_in_output": "Every claim in the chat (a number, a command, a file read) appears in the printed "
                          "output or the tool calls",
    "f_ambiguity_flagged": "Ambiguity flagged where a careful person would hesitate",
    "g_order_and_scope": "Bounds set from physics before the data were read; lab_s07_tools.py and the "
                         "notebooks never opened",
}


def empty_audit():
    return {k: {"answer": None, "evidence": None} for k in AUDIT_ITEMS}


def audit_summary(audit):
    """(answered, yes, with evidence) counts for an AUDIT dict."""
    vals = [v if isinstance(v, dict) else {"answer": v, "evidence": None} for v in audit.values()]
    answered = sum(str(v.get("answer")).strip().lower() in ("yes", "no") for v in vals)
    yes = sum(str(v.get("answer")).strip().lower() == "yes" for v in vals)
    ev = sum(isinstance(v.get("evidence"), str) and len(v["evidence"].strip()) >= 5 for v in vals)
    return answered, yes, ev


def scoreboard_line(netid, station, variant, run, audit, scores, hand_card, agent_card=None,
                    snapshots_unchanged=None, checker_unchanged=None):
    """One line for the class form. A run without an agent card (timeout, no JSON) still gets a line."""
    needed = {"netid": netid, "station": station, "tool": run.get("tool"), "mode": run.get("mode"),
              "wall_time_min": run.get("wall_time_min")}
    missing = [k for k, v in needed.items() if v is None]
    answered, yes, ev = audit_summary(audit)
    if missing:
        return "not filled yet: " + ", ".join(missing)
    s = _station(station)
    h = check_card(s, hand_card, quiet=True)
    h = h[~h["check"].isin(["station"]) & (h["pass"] != "ignored")]
    hc, hf = _summary(h["pass"])
    fmt = lambda d: ",".join("-" if d.get(k) is None else str(d[k]) for k in sorted(d))  # noqa: E731
    fails = lambda t: ";".join(t.loc[t["pass"] == "FAIL", "check"]) or "none"  # noqa: E731
    if agent_card is None:
        why = "timeout" if str(run.get("wall_time_min")).lower().startswith("timeout") else "no card"
        agent_part = f"agent n/a ({why})"
        agent_fails = "n/a"
    else:
        a = check_card(s, agent_card, quiet=True)
        a = a[~a["check"].isin(["station"]) & (a["pass"] != "ignored")]
        ac, af = _summary(a["pass"])
        agent_part = f"agent {ac['pass']} pass/{af} filled/{len(a)}"
        agent_fails = fails(a)
    log = check_log(s)
    wt = run.get("wall_time_min")
    wt = f"{wt} min" if _num(wt) is not None else str(wt)
    flag = lambda v: "-" if v is None else ("unchanged" if v else "CHANGED")  # noqa: E731
    return (f"S07 | {netid} | station {s} | {variant} | {run.get('tool')} ({run.get('model')}) | {run.get('mode')} | "
            f"{wt} | {run.get('turns')} turns | hand {hc['pass']} pass/{hf} filled/{len(h)} | "
            f"{agent_part} | hand checked {log['n']}x, last {log['last']} | bounds revised {bounds_revised(s, hand_card)} | "
            f"snapshots {flag(snapshots_unchanged)} | checker {flag(checker_unchanged)} | "
            f"audit {answered}/7 answered, {yes} yes, {ev} with evidence | agent S1-S8 {fmt(scores.get('agent', {}))} | "
            f"hand S3-S6 {fmt(scores.get('hand', {}))} | hand FAIL {fails(h)} | agent FAIL {agent_fails}")


# ---------------------------------------------------------------------------
# The key, for the instructor at 11:13. The text lives outside the book repository.
# ---------------------------------------------------------------------------

def reveal_key(station, key_file=None):
    """Print the key for one station from the instructor's key file (path in key_file or $LAB_S07_KEY)."""
    s = _station(station)
    path = key_file or os.environ.get("LAB_S07_KEY")
    if not path or not Path(path).expanduser().is_file():
        print("The key is not part of the book. The instructor runs it from a separate file in class at 11:13.")
        return None
    spec = importlib.util.spec_from_file_location("lab_s07_key", Path(path).expanduser())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import sys
    return mod.reveal_key(s, lab=sys.modules[__name__])
