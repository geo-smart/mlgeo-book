"""Session 7, land-surface track: the real data, the synthetic twin with planted problems, and the scorer.

Used by lab_s07_land_surface_model.ipynb; the self-checks and the session summary are in lab_s07_lsm_checks.py.

1. The real data. ``fetch_whole_and_cut`` downloads PSL's whole soil-water and precipitation files, checks
   that each arrived complete, cuts the box and the period, and writes the subset with ``write_grid``, the
   function that also wrote the course copy (snapshots/make_cpc_soilw_precl_westUS_1991-2020.py), so the two
   files have the same bytes when the values agree. ``file_facts`` gives size, counts, first and last records
   and the SHA-256 fingerprint. ``fetch_from_provider`` (the OPeNDAP subset request) is kept for the make
   script; it has not succeeded from our machines.
2. The synthetic twin (``make_twin``): our own small leaky-bucket model, in the spirit of Huang et al. (1996)
   but not CPC's code, run on synthetic weather built around the real 1991-2020 climatology of the Pacific
   Northwest cells. The truth is known; the file you receive carries planted problems chosen and placed from
   a seed derived from your UW NetID. ``score_findings`` says how many of them your findings locate.

Nothing here downloads when the environment variable CI is set or MLGEO_OFFLINE=1.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
SNAP = HERE / "snapshots"
TAG = "cpc_soilw_precl_westUS_1991-2020"
GRID_FILE = SNAP / f"{TAG}.nc"
STATIC_FILE = SNAP / "cpc_westUS_static_1991-2020.nc"
TABLE_FILE = SNAP / "cpc_soilw_precl_spokane_cell_1991-2020.csv"
PROV_FILE = SNAP / f"{TAG}.provenance.json"
USCRN_FILE = SNAP / "uscrn_WA_Spokane_17_SSW_2015-2024.txt"
USCRN_HEADERS = SNAP / "uscrn_daily01_HEADERS.txt"
MAP_FILE = SNAP / "naturalearth_pnw_map.csv"

OPENDAP = {
    "soilw": "https://psl.noaa.gov/thredds/dodsC/Datasets/cpcsoil/soilw.mon.mean.v2.nc",
    "precip": "https://psl.noaa.gov/thredds/dodsC/Datasets/precl/0.5deg/precip.mon.mean.0.5x0.5.nc",
}
BOX = {"lat": (31.0, 49.0), "lon": (235.0, 251.0)}      # degrees north; degrees east (0 to 360)
PERIOD = ("1991-01-01", "2020-12-31")
FILL = np.float32(-9.96921e36)                          # the provider's missing value
KEEP_ATTRS = ("long_name", "units", "var_desc", "dataset", "level_desc", "statistic", "parent_stat")
W_MAX = 760.0                                           # mm, CPC's holding capacity
CLASS_SEED_NAME = "ess469-autumn-2026"
NOT_ANSWERED = "not answered yet"
_LAST = {}                                               # the last file comparison, read by the self-check of Q1.1


# --------------------------------------------------------------------------------------------------
# 1. The real data
# --------------------------------------------------------------------------------------------------
def offline():
    """True on the book build (CI set) or when MLGEO_OFFLINE=1: no download is attempted."""
    return bool(os.environ.get("CI")) or os.environ.get("MLGEO_OFFLINE") == "1"


def output_dir():
    """Where your own files go: s07_my_files/ beside the notebook, or a temporary folder on the book build."""
    d = Path(tempfile.gettempdir()) / "mlgeo_s07_lsm" if offline() else HERE / "s07_my_files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def where(path):
    """A short, readable name for a file location: snapshots/..., s07_my_files/... or the build's temporary folder."""
    path = Path(path)
    try:
        return str(path.relative_to(HERE))
    except ValueError:
        return f"(temporary folder of the book build)/{path.name}"


def fingerprint(path):
    """SHA-256 of the file's bytes, as 64 hexadecimal characters."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _days_since_1800(time_values):
    t = pd.DatetimeIndex(time_values)
    return ((t - pd.Timestamp("1800-01-01")) / pd.Timedelta(days=1)).to_numpy(dtype="f8")


def write_grid(soilw, precip, path):
    """Write soil moisture and precipitation (xarray DataArrays on the same grid) to a NetCDF-3 file.

    Only values, coordinates and a fixed list of the provider's text attributes go in, nothing that depends on
    the day or the machine, so the same values always give the same bytes.
    """
    import netCDF4

    path = Path(path)
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as nc:
        nc.setncattr("title", "CPC soil moisture (V2) and PREC/L precipitation, monthly, 0.5 degree, "
                              "western United States 31N-49N, 235E-251E, 1991-2020")
        nc.setncattr("source", "NOAA PSL, https://psl.noaa.gov/data/gridded/data.cpcsoil.html and "
                               "https://psl.noaa.gov/data/gridded/data.precl.html")
        nc.setncattr("subset_written_by", "lab_s07_lsm_synthetic.write_grid")
        nc.createDimension("time", soilw.sizes["time"])
        nc.createDimension("lat", soilw.sizes["lat"])
        nc.createDimension("lon", soilw.sizes["lon"])
        t = nc.createVariable("time", "f8", ("time",))
        t.setncattr("units", "days since 1800-01-01 00:00:00")
        t.setncattr("calendar", "standard")
        t.setncattr("long_name", "first day of the month")
        t[:] = _days_since_1800(soilw["time"].values)
        for name, units in (("lat", "degrees_north"), ("lon", "degrees_east")):
            v = nc.createVariable(name, "f4", (name,))
            v.setncattr("units", units)
            v[:] = soilw[name].values.astype("f4")
        for name, da in (("soilw", soilw), ("precip", precip)):
            v = nc.createVariable(name, "f4", ("time", "lat", "lon"), fill_value=FILL)
            for k in KEEP_ATTRS:
                if isinstance(da.attrs.get(k), str):
                    v.setncattr(k, da.attrs[k])
            v.setncattr("missing_value", FILL)
            v.set_auto_maskandscale(False)
            a = da.transpose("time", "lat", "lon").values.astype("f4")
            v[:] = np.where(np.isfinite(a), a, FILL).astype("f4")


def open_grid(path=GRID_FILE):
    """Open a grid file written by write_grid (missing values become NaN)."""
    return xr.open_dataset(path).load()


def spokane_station():
    """USCRN Spokane 17 SSW coordinates from NCEI's station list, as recorded in the USCRN provenance file."""
    import json
    row = json.loads((SNAP / "uscrn_WA_Spokane_17_SSW_2015-2024.provenance.json").read_text())["station_list_row"]
    return float(row["LATITUDE"]), float(row["LONGITUDE"])


def file_facts(path):
    """The facts that tell two copies of a grid file apart: bytes, counts, first and last records, fingerprint."""
    ds = open_grid(path)
    lat, lon = spokane_station()
    cell = ds["soilw"].sel(lat=lat, lon=lon % 360, method="nearest")
    land = ds["soilw"].notnull().any("time")
    t = pd.DatetimeIndex(ds["time"].values)
    return {
        "bytes": Path(path).stat().st_size,
        "n_months": int(ds.sizes["time"]), "n_lat": int(ds.sizes["lat"]), "n_lon": int(ds.sizes["lon"]),
        "n_land_cells": int(land.sum()),
        "first_month": t[0].strftime("%Y-%m"), "last_month": t[-1].strftime("%Y-%m"),
        "first_soilw_at_spokane_cell_mm": round(float(cell[0]), 3),
        "last_soilw_at_spokane_cell_mm": round(float(cell[-1]), 3),
        "sha256": fingerprint(path),
    }


def _subset(da):
    lat0, lat1 = BOX["lat"]
    lon0, lon1 = BOX["lon"]
    ilat = np.flatnonzero((da["lat"].values > lat0) & (da["lat"].values < lat1))
    ilon = np.flatnonzero((da["lon"].values > lon0) & (da["lon"].values < lon1))
    t = pd.DatetimeIndex(da["time"].values)
    itime = np.flatnonzero((t >= PERIOD[0]) & (t <= PERIOD[1]))
    sub = da.isel(time=slice(itime[0], itime[-1] + 1), lat=slice(ilat[0], ilat[-1] + 1),
                  lon=slice(ilon[0], ilon[-1] + 1))
    query = f"{da.name}[{itime[0]}:1:{itime[-1]}][{ilat[0]}:1:{ilat[-1]}][{ilon[0]}:1:{ilon[-1]}]"
    return sub.load().sortby("lat", ascending=False), query


def fetch_from_provider(path, timeout=60):
    """Ask the PSL THREDDS server for the subset through OPeNDAP and write it with write_grid.

    Used by the make script only: from our machines the server has answered "NetCDF: Access failure" or
    HTTP 429 (too many requests) every time. The notebook downloads the whole files instead
    (fetch_whole_and_cut). Returns a short message with the two queries; raises on any failure or after
    `timeout` seconds.
    """
    if offline():
        raise RuntimeError("downloads are switched off (offline mode)")

    def work():
        out, queries = {}, []
        for name, url in OPENDAP.items():
            with xr.open_dataset(url, engine="netcdf4") as ds:
                out[name], q = _subset(ds[name])
                queries.append(f"{url}.ascii?{q}")
        return out, queries

    # The NetCDF and HDF5 C libraries print long diagnostics on a failed request; keep them out of the notebook.
    sys.stderr.flush()
    devnull, saved = os.open(os.devnull, os.O_WRONLY), os.dup(2)
    os.dup2(devnull, 2)
    pool = cf.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(work)
    try:
        out, queries = future.result(timeout=timeout)
    except cf.TimeoutError:
        raise TimeoutError(f"no complete answer within {timeout} s") from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
    write_grid(out["soilw"], out["precip"], path)
    return "the server answered these two subset queries (OPeNDAP constraint expressions):\n  " + "\n  ".join(queries)


def provenance():
    """The course copy's provenance record (URLs, retrieval time, fingerprints, terms, citations)."""
    import json
    return json.loads(PROV_FILE.read_text())


def _download(key, timeout=30, progress=None):
    """Download one whole provider file named in the provenance record into output_dir().

    Returns (path, bytes the server announced, the server's Last-Modified date). Raises on any failure, and
    gives up when the download has taken more than `timeout` seconds (wall clock, not only per network read).
    """
    import requests
    if offline():
        raise RuntimeError("downloads are switched off (offline mode)")
    info = provenance()["served_whole_files"][key]
    out = output_dir() / info["url"].rsplit("/", 1)[1]
    part = out.with_suffix(out.suffix + ".part")
    t0 = time.time()
    with requests.get(info["url"], timeout=min(timeout, 60), stream=True,
                      headers={"User-Agent": "mlgeo-book session 7"}) as r:
        r.raise_for_status()
        announced = int(r.headers.get("Content-Length", -1))
        if progress is not None:
            progress.update(file=out.name, total=announced, done=0)
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                if progress is not None:
                    progress["done"] += len(chunk)
                if time.time() - t0 > timeout:
                    f.close()
                    part.unlink()
                    raise TimeoutError(f"{out.name}: stopped after {timeout} s; the network is too slow right now")
    part.replace(out)
    return out, announced, r.headers.get("Last-Modified", "")


def fetch_whole_file(key="landperc", timeout=30):
    """Download one whole provider file named in the provenance record; returns the local path."""
    return _download(key, timeout)[0]


def cut_box(da):
    """The western United States box and 1991-2020, latitude from north to south: the make script's cut."""
    lat0, lat1 = BOX["lat"]
    lon0, lon1 = BOX["lon"]
    da = da.sel(lon=slice(lon0, lon1)).sortby("lat", ascending=False).sel(lat=slice(lat1, lat0))
    return da.sel(time=slice(*PERIOD))


WHOLE_KEYS = ("soilw", "precip", "landperc")      # the two files that hold the grid, and the land fraction


def fetch_whole_and_cut(path, timeout=1200, keep=False, say=print, progress=None):
    """Download PSL's whole files, check each is complete, cut the box and 1991-2020, write it with write_grid.

    Prints the download size first (about 545 MB) and stops if the disk lacks room. Each file gets `timeout`
    seconds of wall-clock time. After the cut, the two large files are deleted unless keep=True; landperc.nc
    (1 MB) stays. Returns a table with, for each file, the size the server announced, the bytes received, the
    server's date, and whether the fingerprint matches the record kept when the course copy was built.
    """
    if offline():
        raise RuntimeError("downloads are switched off (offline mode)")
    record = provenance()["served_whole_files"]
    total = sum(record[k]["bytes"] for k in WHOLE_KEYS)
    free = shutil.disk_usage(output_dir()).free
    say(f"Downloading about {total / 1e6:.0f} MB from downloads.psl.noaa.gov ({free / 1e9:.1f} GB free on this "
        f"disk), at most {timeout / 60:.0f} min per file.")
    if free < 1.2 * total:
        raise OSError(f"not enough free disk space: about {total / 1e6:.0f} MB needed")
    rows, paths = {}, {}
    for k in WHOLE_KEYS:
        t0 = time.time()
        p, announced, modified = _download(k, timeout=timeout, progress=progress)
        got = p.stat().st_size
        say(f"{p.name}: {got / 1e6:.1f} MB in {time.time() - t0:.0f} s")
        rows[p.name] = {"announced_bytes": announced, "received_bytes": got, "complete": got == announced,
                        "server_date": modified, "record_date": record[k]["last_modified"],
                        "same_fingerprint_as_record": fingerprint(p) == record[k]["sha256"]}
        if got != announced:
            raise OSError(f"{p.name} is incomplete: {got} of {announced} bytes")
        paths[k] = p
    with xr.open_dataset(paths["soilw"]) as s, xr.open_dataset(paths["precip"]) as q:
        soil, prec = cut_box(s["soilw"]).load(), cut_box(q["precip"]).load()
    write_grid(soil, prec, path)
    if not keep:
        for k in ("soilw", "precip"):
            paths[k].unlink()
    return pd.DataFrame(rows).T


class Download:
    """fetch_whole_and_cut running in the background, so you keep working while about 544 MB arrive.

    status() says where it stands; when it has finished, `table` holds the per-file checks and `error` any
    failure. A kernel restart stops it.
    """

    def __init__(self, path, timeout=1200):
        import threading
        self.path, self.timeout, self.t0 = Path(path), timeout, time.time()
        self.log, self.progress, self.table, self.error, self.done = [], {}, None, None, False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self.table = fetch_whole_and_cut(self.path, timeout=self.timeout, say=self.log.append,
                                             progress=self.progress)
        except Exception as e:  # noqa: BLE001  reported by status()
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
        self.done = True

    def status(self):
        minutes = (time.time() - self.t0) / 60
        if self.error:
            return f"download stopped after {minutes:.0f} min ({self.error})"
        if self.done:
            return f"download finished in {minutes:.0f} min; your cut is {where(self.path)}"
        p, n_done = self.progress, len([m for m in self.log if " MB in " in m])
        if not p:
            return f"waiting for the server ({minutes:.0f} min)"
        if n_done == len(WHOLE_KEYS):
            return f"all files downloaded; cutting the box and writing your file ({minutes:.0f} min)"
        return (f"downloading {p['file']}: {p['done'] / 1e6:.0f} of {p['total'] / 1e6:.0f} MB "
                f"after {minutes:.0f} min ({n_done} of {len(WHOLE_KEYS)} files done)")


def start_download(path, timeout=1200):
    """Start fetch_whole_and_cut in the background and return the Download that tracks it."""
    if offline():
        raise RuntimeError("downloads are switched off (offline mode)")
    return Download(path, timeout)


def rewrite_course_copy(path):
    """Write the course copy's numbers again with write_grid: what a download would give if no value had changed."""
    g = open_grid(GRID_FILE)
    write_grid(g["soilw"], g["precip"], path)
    return Path(path)


def read_uscrn():
    """The USCRN Spokane 17 SSW daily file, with the provider's column names; missing codes left as served."""
    names = USCRN_HEADERS.read_text().split("\n")[1].split()
    df = pd.read_csv(USCRN_FILE, sep=r"\s+", header=None, names=names, dtype={"CRX_VN": str})
    df["date"] = pd.to_datetime(df["LST_DATE"].astype(str))
    return df.set_index("date")


def read_map():
    """Natural Earth coastline, borders and places clipped to the Pacific Northwest (public domain)."""
    return pd.read_csv(MAP_FILE)


def draw_map(ax, color="0.25", lw=0.7, places=True):
    """Draw the Natural Earth lines on a matplotlib axis in degrees east (-180 to 180)."""
    m = read_map()
    for (kind, part), g in m[m["kind"] != "place"].groupby(["kind", "part"]):
        ax.plot(g["lon"], g["lat"], color=color, lw=lw if kind == "coastline" else lw * 0.6,
                ls="-" if kind == "coastline" else "--")
    if places:
        for _, p in m[m["kind"] == "place"].iterrows():
            ax.plot(p["lon"], p["lat"], "o", color=color, ms=3)
            ax.annotate(p["name"], (p["lon"], p["lat"]), xytext=(3, 3), textcoords="offset points",
                        fontsize=8, color=color)


def compare_files(course, yours):
    """Side-by-side facts for the course copy and your download; remembers whether the bytes are the same."""
    a, b = file_facts(course), file_facts(yours)
    _LAST["same_bytes"] = a["sha256"] == b["sha256"]
    return pd.DataFrame({"course copy (snapshots/)": a, "yours": b})


def uscrn_daily():
    """USCRN Spokane 17 SSW soil moisture (m3/m3) and soil temperature (degC) at five depths, missing codes as NaN."""
    u = read_uscrn()
    sm = [c for c in u.columns if c.startswith("SOIL_MOISTURE")]
    st = [c for c in u.columns if c.startswith("SOIL_TEMP")]
    out = u[sm + st + ["P_DAILY_CALC"]].copy()
    out[sm] = out[sm].where(out[sm] > -99.0)
    out[st + ["P_DAILY_CALC"]] = out[st + ["P_DAILY_CALC"]].where(out[st + ["P_DAILY_CALC"]] > -9999.0)
    return out


# Layer thicknesses (mm) that each USCRN probe stands for in the top 100 cm: 0-7.5, 7.5-15, 15-35, 35-75, 75-100 cm.
USCRN_LAYERS_MM = {"SOIL_MOISTURE_5_DAILY": 75.0, "SOIL_MOISTURE_10_DAILY": 75.0, "SOIL_MOISTURE_20_DAILY": 200.0,
                   "SOIL_MOISTURE_50_DAILY": 400.0, "SOIL_MOISTURE_100_DAILY": 250.0}


def uscrn_monthly_storage(min_days=20):
    """Monthly soil water in the top 100 cm at the USCRN station (mm), from the five probes.

    Each probe stands for one layer (USCRN_LAYERS_MM); a day counts only when all five depths have a value,
    and a month only when it has at least `min_days` such days.
    """
    d = uscrn_daily()
    cols = list(USCRN_LAYERS_MM)
    store = sum(d[c] * USCRN_LAYERS_MM[c] for c in cols)
    store = store.where(d[cols].notna().all(axis=1))
    g = store.resample("MS")
    out = pd.DataFrame({"insitu_top100cm_mm": g.mean(), "n_days": g.count()})
    out.loc[out["n_days"] < min_days, "insitu_top100cm_mm"] = np.nan
    out.index = out.index.strftime("%Y-%m")
    return out


TWIN_README = """Synthetic twin of the CPC soil moisture grid, Pacific Northwest, monthly, January 1991 to December 2020.
Columns: time (first day of the month), lat (degrees north, cell centre), lon (degrees east from 0 to 360,
cell centre), soil_moisture_mm (monthly mean water in a one-layer 1.6 m soil column, holding capacity 760 mm),
precip_mm_per_day (monthly mean precipitation rate from a gauge analysis over land only).
Grid: 0.5 degree. Rows exist where at least one of the two variables has a value."""

TOOL_INSTRUCTIONS = """I attach a CSV file. Here is what its provider says it contains:

{readme}

Please (1) list every problem you find in this file: for each, the variable, the first and last month
affected (YYYY-MM), the latitude and longitude range affected, what is wrong, and why it is physically
implausible; (2) say which values you would keep as real extremes of the climate; (3) compute the percent
of cell-months in 2011-2020 that are drought months, where a drought month has soil moisture below the
20th percentile of the same cell and calendar month over 1991-2010, first on the file as it is and then
after your cleaning; (4) give your findings as a Python list of dictionaries with keys variable, start,
end, lat, lon, what, why (lat and lon as (min, max) tuples, or None for the whole grid)."""


# --------------------------------------------------------------------------------------------------
# 2. The synthetic twin
# --------------------------------------------------------------------------------------------------
SYN_LAT = (42.0, 49.0)            # the Pacific Northwest part of the grid
SYN_LON = (235.0, 245.0)          # 125 W to 115 W
START_YEAR, END_YEAR, SPIN_YEARS = 1991, 2020, 10

CATEGORY_WORDS = {
    "spin_up": "the model was started from a wrong initial soil water and had not settled (spin-up drift)",
    "unit_change": "a block of months stored in other units (metres, or a volumetric fraction in m3/m3) instead of millimetres",
    "zero_fill": "a missing month filled with zeros",
    "ocean_leak": "ocean cells that carry soil moisture as if they were land",
    "lon_convention": "longitudes switched from 0-360 degrees east to -180-180 part way through the record",
    "duplicate": "a month written twice",
    "gauge_decimal": "one rain-gauge value entered ten times too large, and the soil's response to it",
    "network_step": "a jump in precipitation when the gauge network changed, and the soil's response",
    "precip_time_shift": "precipitation labelled one month late",
    "grid_offset": "the precipitation grid offset by one cell (half a degree) from the soil grid",
}
CORE = ["spin_up", "unit_change", "zero_fill", "ocean_leak", "lon_convention", "duplicate"]
EXTRA = ["gauge_decimal", "network_step", "precip_time_shift", "grid_offset"]
N_CORE, N_EXTRA = 4, 1         # each twin carries 5 problems: 4 of the 6 core kinds and 1 of the 4 extra kinds


def choose_problems(rng):
    """Which kinds of problem a twin carries: 15 x 4 = 60 possible sets, drawn from the seed."""
    core = set(rng.choice(CORE, size=N_CORE, replace=False).tolist())
    extra = set(rng.choice(EXTRA, size=N_EXTRA, replace=False).tolist())
    return core | extra


def seed_from_netid(netid):
    """An integer seed from a UW NetID (lower case, spaces removed); the class seed when netid is empty."""
    name = (netid or "").strip().lower() or CLASS_SEED_NAME
    return int(hashlib.sha256(name.encode()).hexdigest()[:12], 16)


def _thornthwaite(T, days, I, a):
    """Potential evaporation (mm per month) from monthly mean temperature (degC), Thornthwaite (1948)."""
    Tp = np.clip(T, 0.0, 26.0)
    return 16.0 * (days / 30.0) * np.where(Tp > 0, (10.0 * Tp / np.maximum(I, 1e-6)) ** a, 0.0)


def run_bucket(P, T, days, I, a, W0, S0=None, nsub=6):
    """A one-layer leaky bucket, our own simplified version of the CPC idea (not CPC's code).

    P: precipitation, mm/day (monthly mean rate), T: air temperature degC, both shaped (month, cell).
    Snow falls below +2 degC and melts with a degree-day factor; runoff grows with wetness; evaporation is
    potential evaporation times W / W_MAX; a small leak to groundwater. Returns monthly mean soil water (mm).
    """
    n, m = P.shape
    W = np.full(m, W0, dtype="f8") if np.isscalar(W0) else W0.astype("f8").copy()
    S = np.zeros(m) if S0 is None else S0.copy()
    out = np.empty((n, m))
    for i in range(n):
        d = days[i]
        snow_frac = np.clip((2.0 - T[i]) / 4.0, 0.0, 1.0)
        pe = _thornthwaite(T[i], d, I, a) / nsub
        acc = np.zeros(m)
        for _ in range(nsub):
            p = P[i] * d / nsub
            S += p * snow_frac
            melt = np.minimum(S, 3.0 * np.maximum(T[i], 0.0) * d / nsub)
            S -= melt
            inflow = p * (1.0 - snow_frac) + melt
            wet = W / W_MAX
            W = W + inflow * (1.0 - wet ** 2) - pe * wet - 0.0006 * W * d / nsub
            W = np.clip(W, 0.0, W_MAX)
            acc += W
        out[i] = acc / nsub
    return out


@dataclass
class Twin:
    """A synthetic twin: the truth, the file you receive, and the planted problems (kept out of sight)."""
    netid: str
    seed: int
    path: Path
    truth: pd.DataFrame
    lat: np.ndarray
    lon: np.ndarray
    land: np.ndarray
    months: pd.DatetimeIndex
    _planted: list = field(repr=False, default_factory=list)

    def __repr__(self):
        return (f"Twin(netid={self.netid!r}, file={self.path.name}, {len(self._planted)} planted problems, "
                f"{len(self.lat)} x {len(self.lon)} cells, {len(self.months)} months)")


def _real_inputs():
    g = open_grid(GRID_FILE)
    s = xr.open_dataset(STATIC_FILE).load()
    sel = dict(lat=slice(SYN_LAT[1], SYN_LAT[0]), lon=slice(SYN_LON[0], SYN_LON[1] + 0.6))
    soil = g["soilw"].sel(**sel)
    land = soil.notnull().all("time").values
    pclim = g["precip"].sel(**sel).groupby("time.month").mean().values            # (12, lat, lon) mm/day
    tclim = s["air_clim"].sel(**sel).values                                       # (12, lat, lon)
    if np.nanmean(tclim) > 100:                                                   # kelvin to celsius
        tclim = tclim - 273.15
    return soil["lat"].values, soil["lon"].values, land, pclim, tclim


def _ar1(rng, n, phi, shape=()):
    z = np.empty((n,) + shape)
    z[0] = rng.standard_normal(shape)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(shape)
    return z


def _weather(rng, lat, lon, land, pclim, tclim, months):
    """Synthetic monthly precipitation (mm/day) and temperature (degC) around the real climatology."""
    n = len(months)
    mo = months.month.values - 1
    ny, nx = len(lat), len(lon)
    yrs = months.year.values + (months.month.values - 0.5) / 12
    big = _ar1(rng, n, 0.35)[:, None, None]
    dip = _ar1(rng, n, 0.30)[:, None, None] * ((lat - 45.5) / 3.5)[None, :, None]
    cell = rng.standard_normal((n, ny, nx))
    z = np.clip(0.75 * big + 0.35 * dip + 0.55 * cell, -2.6, 2.6)
    dry = 1.0 - 0.12 * np.exp(-((yrs - 2015.2) / 1.3) ** 2)                       # a multi-year dry spell
    sig = 0.42
    pc = np.nan_to_num(pclim[mo])
    P = pc * np.exp(sig * z - sig ** 2 / 2) * dry[:, None, None]
    tt = _ar1(rng, n, 0.5)[:, None, None]
    T = (np.nan_to_num(tclim[mo]) + 1.2 * tt + 0.3 * rng.standard_normal((n, ny, nx))
         + 0.02 * (yrs - 2005)[:, None, None] + 0.6 * np.exp(-((yrs - 2015.4) / 0.6) ** 2)[:, None, None])
    P = np.where(land[None], P, np.nan)
    return P, T


def _model(P, T, months, land, tclim, W0=None, spin=None):
    """Run the bucket on land cells; with spin=(P, T) for the spin-up years, start from their end state."""
    ny, nx = land.shape
    idx = np.flatnonzero(land.ravel())
    Tc = np.nan_to_num(tclim).reshape(12, -1)[:, idx]
    I = np.sum((np.clip(Tc, 0, None) / 5.0) ** 1.514, axis=0)
    a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
    days = months.days_in_month.values.astype(float)

    def flat(X):
        return np.nan_to_num(X.reshape(len(X), -1)[:, idx])

    if spin is not None:
        Ps, Ts, ms = spin
        Ws = run_bucket(flat(Ps), flat(Ts), ms.days_in_month.values.astype(float), I, a, 0.5 * W_MAX)
        W0 = Ws[-1]
    W = run_bucket(flat(P), flat(T), days, I, a, W0)
    out = np.full((len(months), ny * nx), np.nan)
    out[:, idx] = W
    return out.reshape(len(months), ny, nx)


def _footprint(diff, land, thresh=5.0):
    """Months and cells where |diff| exceeds thresh mm."""
    big = np.abs(np.nan_to_num(diff)) > thresh
    t = np.flatnonzero(big.any(axis=(1, 2)))
    cells = np.argwhere(big.any(axis=0))
    return (int(t.min()), int(t.max())) if len(t) else (0, 0), {tuple(c) for c in cells}


def make_twin(netid=None, outdir=None):
    """Build your synthetic twin and write the file you will inspect (CSV, one row per month and cell)."""
    seed = seed_from_netid(netid)
    rng = np.random.default_rng(seed)
    lat, lon, land, pclim, tclim = _real_inputs()          # lon has one extra column on the east
    months = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-01", freq="MS")
    spin_months = pd.date_range(f"{START_YEAR - SPIN_YEARS}-01-01", f"{START_YEAR - 1}-12-01", freq="MS")
    allm = spin_months.append(months)
    P_all, T_all = _weather(rng, lat, lon, land, pclim, tclim, allm)
    ns = len(spin_months)
    P, T = P_all[ns:], T_all[ns:]
    spin = (P_all[:ns], T_all[:ns], spin_months)
    W_true = _model(P, T, months, land, tclim, spin=spin)
    n = len(months)
    nx = len(lon) - 1                                        # the file keeps 20 columns
    landf = land[:, :nx]
    all_land = {tuple(c) for c in np.argwhere(landf)}
    planted = []

    def plant(cat, var, t0, t1, cells):
        planted.append({"category": cat, "variable": var, "t0": int(t0), "t1": int(t1),
                        "cells": {c for c in cells if c[1] < nx}})

    kinds = choose_problems(rng)

    # Model-level problems: forcing errors and the spin-up, then one run of the model that carries them.
    P_file = P.copy()
    if "gauge_decimal" in kinds:
        winter = [i for i in range(24, n - 6) if months[i].month in (11, 12, 1, 2)]
        i0 = int(rng.choice(winter))
        east = np.argwhere(landf & (lon[None, :nx] > 239.0))
        cy, cx = east[rng.integers(len(east))]
        P_file[i0, cy, cx] *= 10.0
    if "network_step" in kinds:
        y0 = int(rng.integers(1998, 2011))
        n0 = (y0 - START_YEAR) * 12
        east_cols = lon > 239.0
        P_file[n0:, :, east_cols] *= 1.3
    W_forced = _model(P_file, T, months, land, tclim, spin=(spin[0], spin[1], spin[2]))
    if "spin_up" in kinds:
        w0 = float(rng.choice([0.03, 1.0])) * W_MAX
        W_file = _model(P_file, T, months, land, tclim, W0=w0)
        (s0, s1), _ = _footprint((W_file - W_forced)[:, :, :nx], landf, thresh=10.0)
        plant("spin_up", "soil", 0, s1, all_land)
    else:
        W_file, s1 = W_forced, -3
    if "gauge_decimal" in kinds:
        P_gauge_only = P.copy()
        P_gauge_only[i0, cy, cx] *= 10.0
        W_gauge_only = _model(P_gauge_only, T, months, land, tclim, spin=(spin[0], spin[1], spin[2]))
        (f0, f1), cells = _footprint((W_gauge_only - W_true)[:, :, :nx], landf)
        plant("gauge_decimal", "both", i0, max(f1, i0), cells | {(int(cy), int(cx))})
    if "network_step" in kinds:
        plant("network_step", "both", n0, n - 1, {c for c in all_land if lon[c[1]] > 239.0})

    soil = W_file.copy()
    taken = set(range(0, s1 + 3))
    if "unit_change" in kinds:
        # A block of months in other units: metres (divide by 1000) or a volumetric fraction of the 1.6 m column
        # (divide by 1600 mm), the two commonest confusions between land-surface products.
        L = int(rng.integers(6, 19))
        u0 = int(rng.choice([i for i in range(36, n - L - 1) if not taken & set(range(i - 2, i + L + 2))]))
        soil[u0:u0 + L] /= float(rng.choice([1000.0, 1600.0]))
        taken |= set(range(u0 - 2, u0 + L + 2))
        plant("unit_change", "soil", u0, u0 + L - 1, all_land)
    if "zero_fill" in kinds:
        z0 = int(rng.choice([i for i in range(36, n - 2) if i not in taken]))
        soil[z0] = np.where(land, 0.0, np.nan)
        taken |= {z0 - 1, z0, z0 + 1}
        plant("zero_fill", "soil", z0, z0, all_land)
    if "ocean_leak" in kinds:
        # Ocean cells that leak in, with a constant value: 0 (PSL's "ocean has 0's") or the capacity.
        coast = [tuple(c) for c in np.argwhere(~landf) if
                 landf[max(c[0] - 1, 0):c[0] + 2, max(c[1] - 1, 0):c[1] + 2].any()]
        k = int(rng.integers(4, 9))
        start = int(rng.integers(0, max(1, len(coast) - k)))
        coast.sort()
        leak = coast[start:start + k]
        leak_value = float(rng.choice([0.0, W_MAX]))
        for (cy_, cx_) in leak:
            soil[:, cy_, cx_] = leak_value
        plant("ocean_leak", "soil", 0, n - 1, set(leak))
    # Precipitation problems that do not touch the soil (the model saw the right forcing).
    if "grid_offset" in kinds:
        Pl = np.where(np.isnan(P_file), 0.0, P_file)
        shifted = np.full_like(P_file, np.nan)
        shifted[:, :, :-1] = Pl[:, :, 1:]
        both_nan = np.isnan(P_file[:, :, :-1]) & np.isnan(P_file[:, :, 1:])
        shifted[:, :, :-1][both_nan] = np.nan
        P_file = shifted
        plant("grid_offset", "precip", 0, n - 1,
              all_land | {tuple(c) for c in np.argwhere(np.isfinite(P_file[0, :, :nx]))})
    if "precip_time_shift" in kinds:
        P_file = np.concatenate([np.full_like(P_file[:1], np.nan), P_file[:-1]])
        plant("precip_time_shift", "precip", 0, n - 1, all_land)

    # The table: one row per month and cell where either variable has a value.
    soil, P_file = soil[:, :, :nx], P_file[:, :, :nx]
    lon_f = lon[:nx]
    tt, yy, xx = np.meshgrid(np.arange(n), np.arange(len(lat)), np.arange(nx), indexing="ij")
    df = pd.DataFrame({"t": tt.ravel(), "iy": yy.ravel(), "ix": xx.ravel(),
                       "soil_moisture_mm": soil.ravel(), "precip_mm_per_day": P_file.ravel()})
    df = df[df["soil_moisture_mm"].notna() | df["precip_mm_per_day"].notna()].copy()
    df["time"] = months.strftime("%Y-%m-%d").values[df["t"].values]
    df["lat"] = lat[df["iy"].values]
    df["lon"] = lon_f[df["ix"].values]
    if "lon_convention" in kinds:
        c0 = int(rng.integers(60, 229))
        df.loc[df["t"] >= c0, "lon"] = df.loc[df["t"] >= c0, "lon"] - 360.0
        plant("lon_convention", "both", c0, n - 1, all_land)
    if "duplicate" in kinds:
        d0 = int(rng.choice([i for i in range(24, n - 1) if i not in taken]))
        dup = df[df["t"] == d0]
        df = pd.concat([df[df["t"] <= d0], dup, df[df["t"] > d0]], ignore_index=True)
        plant("duplicate", "both", d0, d0, all_land)

    out = df[["time", "lat", "lon", "soil_moisture_mm", "precip_mm_per_day"]]
    outdir = Path(outdir) if outdir else output_dir()
    who = re.sub(r"[^a-z0-9]", "", (netid or "").lower()) or "class"
    path = outdir / f"lsm_twin_{who}.csv"
    out.to_csv(path, index=False, float_format="%.4f")

    tr = pd.DataFrame({"time": months.strftime("%Y-%m-%d").values[tt.ravel()],
                       "lat": lat[yy.ravel()], "lon": lon_f[xx.ravel()],
                       "soil_moisture_mm": W_true[:, :, :nx].ravel(),
                       "precip_mm_per_day": P[:, :, :nx].ravel()})
    tr = tr[tr["soil_moisture_mm"].notna()].reset_index(drop=True)
    twin = Twin(netid or "", seed, path, tr, lat, lon_f, landf, months, planted)
    twin._lat_lookup = {round(float(v), 2): i for i, v in enumerate(lat)}
    twin._lon_lookup = {round(float(v), 2): i for i, v in enumerate(lon_f)}
    return twin


def read_twin(path):
    """Read a twin file as it is, without fixing anything."""
    return pd.read_csv(path)


def drought_frequency(df, col="soil_moisture_mm"):
    """Percent of cell-months in 2011-2020 that are drought months.

    A drought month: soil moisture below the 20th percentile of the same cell and calendar month over the
    baseline 1991-2010. Works on any table with columns time, lat, lon and `col`, whatever is wrong with it:
    every row counts, and cells without a baseline are left out.
    """
    d = df[["time", "lat", "lon", col]].dropna(subset=[col]).copy()
    t = pd.to_datetime(d["time"])
    d["year"], d["month"] = t.dt.year.values, t.dt.month.values
    base = (d[(d["year"] >= 1991) & (d["year"] <= 2010)]
            .groupby(["lat", "lon", "month"])[col].quantile(0.2).rename("p20"))
    test = d[(d["year"] >= 2011) & (d["year"] <= 2020)].join(base, on=["lat", "lon", "month"]).dropna(subset=["p20"])
    if test.empty:
        print("drought_frequency: no cell-month of 2011-2020 has a baseline in 1991-2010 for the same lat, lon")
        return float("nan")
    return float(100.0 * (test[col] < test["p20"]).mean())


# Matching your findings to the planted problems -------------------------------------------------
def _month_index(twin, s):
    t = pd.Timestamp(str(s)[:7] + "-01")
    return (t.year - START_YEAR) * 12 + t.month - 1


def _cells_in(twin, lat_rng, lon_rng):
    lat, lon = twin.lat, twin.lon
    ys = np.arange(len(lat)) if lat_rng is None else np.flatnonzero(
        (lat >= min(lat_rng) - 0.26) & (lat <= max(lat_rng) + 0.26))
    if lon_rng is None:
        xs = np.arange(len(lon))
    else:
        lo = [v % 360 for v in lon_rng]
        xs = np.flatnonzero((lon >= min(lo) - 0.26) & (lon <= max(lo) + 0.26))
    return {(int(y), int(x)) for y in ys for x in xs}


def _var_ok(f_var, p_var):
    f = str(f_var or "both").lower()
    fv = "both" if ("both" in f or ("soil" in f and "precip" in f)) else ("soil" if "soil" in f else
                                                                         ("precip" if "precip" in f or "rain" in f else "both"))
    return fv == "both" or p_var == "both" or fv == p_var


LONG_PROBLEMS = {"network_step", "lon_convention", "ocean_leak", "precip_time_shift", "grid_offset"}
REGIONAL_PROBLEMS = {"network_step", "ocean_leak"}
ONSET_TOLERANCE = {"unit_change": 2, "zero_fill": 1, "duplicate": 1, "lon_convention": 2, "network_step": 12,
                   "gauge_decimal": 2, "spin_up": 6}


# Three kinds of problem cover the whole record (and, for two of them, every land cell), so where a finding
# sits cannot tell them apart from a finding about something else, such as the wettest coastal winters
# flagged by a z-score rule. For these the finding must also name the mechanism, in any words of this list.
MECHANISM_WORDS = {
    "ocean_leak": [r"ocean", r"\bsea\b", r"constant", r"never (change|vary|move)", r"(does not|doesn't|do not) (change|vary)",
                   r"unchang", r"\bflat\b", r"same value", r"land mask", r"\bmask", r"not land", r"zero variance",
                   r"no (precipitation|rain)"],
    "precip_time_shift": [r"shift", r"\blag", r"\blate\b", r"\bearly\b", r"\blead", r"label", r"delay", r"behind",
                          r"ahead", r"align", r"timing", r"time ?stamp", r"one month", r"offset"],
    "grid_offset": [r"offset", r"shift", r"displac", r"align", r"neighbo", r"one cell", r"half a degree",
                    r"0\.5 degree", r"ocean", r"\bsea\b", r"land mask", r"\bmask", r"position", r"misplac"],
}


def _names_mechanism(f, category):
    words = MECHANISM_WORDS.get(category)
    if words is None:
        return True
    text = f"{f.get('what', '')} {f.get('why', '')}".lower()
    return any(re.search(w, text) for w in words)


def _matches(twin, f, p):
    """Does finding f locate planted problem p? Variable, months and cells must overlap, a problem that
    starts at a given month must be found starting near that month, and a problem that spans the whole
    record must be named by its mechanism (MECHANISM_WORDS)."""
    if not str(f.get("what", "")).strip():
        return False
    if not _var_ok(f.get("variable"), p["variable"]):
        return False
    if not _names_mechanism(f, p["category"]):
        return False
    try:
        a = _month_index(twin, f.get("start") or f"{START_YEAR}-01")
        b = _month_index(twin, f.get("end") or f.get("start") or f"{END_YEAR}-12")
    except ValueError:
        return False
    a, b = min(a, b), max(a, b)
    ov = min(b, p["t1"] + 1) - max(a, p["t0"] - 1) + 1
    if ov < 1 or ov / (b - a + 1) < 0.2:
        return False
    tol = ONSET_TOLERANCE.get(p["category"])
    if tol is not None and abs(a - p["t0"]) > tol:
        return False
    if p["category"] in LONG_PROBLEMS and ov / (p["t1"] - p["t0"] + 1) < 0.5:
        return False                     # a problem that lasts years is not located by a few months of it
    cells = _cells_in(twin, f.get("lat"), f.get("lon"))
    oc = len(cells & p["cells"])
    if p["category"] in REGIONAL_PROBLEMS and oc / max(len(p["cells"]), 1) < 0.25:
        return False                     # a problem over a whole region is not located by one cell of it
    return oc >= 1 and oc / max(len(cells), 1) >= 0.1


def _assign(findings, twin):
    """One finding locates at most one problem: a maximum matching between findings and problems."""
    cand = [[i for i, f in enumerate(findings) if _matches(twin, f, p)] for p in twin._planted]
    owner = {}

    def augment(j, seen):
        for i in cand[j]:
            if i in seen:
                continue
            seen.add(i)
            if i not in owner or augment(owner[i], seen):
                owner[i] = j
                return True
        return False

    for j in range(len(cand)):
        augment(j, set())
    return owner          # finding index -> problem index


def count_found(findings, twin):
    """Number of planted problems located, each by its own finding."""
    return len(_assign(findings, twin))


def score_findings(findings, twin, reveal=False, who="you"):
    """Print how many planted problems the findings locate (variable, months and cells overlap)."""
    if not findings:
        print(NOT_ANSWERED)
        return
    owner = _assign(findings, twin)
    found = [j in owner.values() for j in range(len(twin._planted))]
    unmatched = [i for i in range(len(findings)) if i not in owner]
    print(f"{who[0].upper() + who[1:]} located {sum(found)} of the {len(found)} planted problems.")
    if unmatched:
        print(f"Finding(s) {unmatched} (counting from 0) locate no planted problem. Possible reasons: real "
              "structure of the synthetic climate; a second finding for a problem already located; the wrong "
              "variable, start month or cells; or a whole-record problem whose 'what' and 'why' do not say what went wrong.")
    if sum(found) < len(found) and not reveal:
        print("To see which kinds you missed (not where they are), run the same call with reveal=True.")
    if reveal:
        missed = [CATEGORY_WORDS[p["category"]] for p, ok in zip(twin._planted, found) if not ok]
        print("Missed kinds:" if missed else "Nothing missed.")
        for m in missed:
            print("  -", m)




# --------------------------------------------------------------------------------------------------
# 3. Quantities the notebook and the self-checks share (the self-checks are in lab_s07_lsm_checks.py)
# --------------------------------------------------------------------------------------------------
def spokane_table():
    """The course table: one row per month at the model cell nearest USCRN Spokane 17 SSW."""
    return pd.read_csv(TABLE_FILE)


def spokane_anomalies():
    """Soil moisture and precipitation anomalies (month minus its 1991-2020 calendar-month mean) at the cell."""
    t = spokane_table()
    m = pd.to_datetime(t["month"]).dt.month
    out = pd.DataFrame({"month": t["month"]})
    for c in ("soilw_mm", "precip_mm_per_day"):
        out[c] = t[c] - t.groupby(m)[c].transform("mean")
    return out


def gauge_totals():
    """Mean number of PREC/L gauges per month over the box's land cells, 1991-1996 and 1997-2020."""
    g = xr.open_dataset(STATIC_FILE)["gauges_total"].to_series()
    y = pd.DatetimeIndex(g.index).year
    return float(g[y <= 1996].mean()), float(g[y >= 1997].mean())


def real_drought_labels():
    """Drought months on the real grid: below the 1991-2010 20th percentile of the cell and calendar month."""
    g = open_grid(GRID_FILE)["soilw"]
    base = g.sel(time=slice("1991", "2010"))
    p20 = base.groupby("time.month").quantile(0.2, dim="time")
    lab_ = (g.groupby("time.month") < p20)
    return lab_.where(g.notnull()), p20
