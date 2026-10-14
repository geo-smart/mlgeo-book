"""Build the committed course copies for the Session 7 land-surface track.

System: the Climate Prediction Center (CPC) "leaky bucket" land model, a one-layer
water balance that turns observed monthly precipitation and temperature into soil
water (Huang et al. 1996; Fan and van den Dool 2004). NOAA's Physical Sciences
Laboratory (PSL) serves its monthly output on a 0.5 degree grid, together with the
gauge-based precipitation analysis that drives it (CPC PRECipitation REConstruction
over Land, PREC/L; Chen et al. 2002) and the temperature analysis (Global Historical
Climatology Network plus Climate Anomaly Monitoring System, GHCN+CAMS; Fan and van den
Dool 2008).

Subset: the western United States box 31 N to 49 N, 125 W to 109 W (235 E to 251 E in
the files' convention), January 1991 to December 2020 (the 1991-2020 climate normal).

The script downloads the provider's whole files from https://downloads.psl.noaa.gov
(about 800 MB in total, kept in a cache folder outside the book), records their
size, Last-Modified date and SHA-256 fingerprint, cuts the subset, and writes:

- cpc_soilw_precl_westUS_1991-2020.nc: soil moisture (soilw, mm) and precipitation
  (precip, mm/day) exactly as served, on the provider's grid, written by
  lab_s07_lsm_synthetic.write_grid. Students write their own live download with the
  same function, so the two files have the same bytes when the values agree.
- cpc_westUS_static_1991-2020.nc: supporting fields: the provider's percent of
  each cell that is land, the provider's 1991-2020 soil moisture climatology, the
  1991-2020 monthly climatology of GHCN+CAMS air temperature (computed here), and
  three summaries of the PREC/L gauge counts (computed here): mean gauges per cell,
  fraction of months with a gauge, and the monthly total over the box's land cells.
- cpc_soilw_precl_spokane_cell_1991-2020.csv: one row per month at the model cell
  nearest the U.S. Climate Reference Network (USCRN) station Spokane 17 SSW.
- cpc_soilw_precl_westUS_1991-2020.provenance.json: URLs, retrieval time, fingerprints,
  counts, terms of use quoted from the provider, citations with resolved DOIs, and,
  when the PSL THREDDS server answers, whether its OPeNDAP subset matches.

Regenerate (network needed, about 10 to 30 minutes for the downloads):
  pixi run python book/Chapter2-DataManipulation/snapshots/make_cpc_soilw_precl_westUS_1991-2020.py [CACHE_DIR]
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lab_s07_lsm_synthetic as lab  # noqa: E402  (the canonical writer and the box definition)

CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache" / "mlgeo_cpc_lsm"
CACHE.mkdir(parents=True, exist_ok=True)
DL = "https://downloads.psl.noaa.gov/Datasets"
SOURCES = {
    "soilw": f"{DL}/cpcsoil/soilw.mon.mean.v2.nc",
    "soilw_ltm": f"{DL}/cpcsoil/soilw.mon.ltm.1991-2020.v2.nc",
    "landperc": f"{DL}/cpcsoil/landperc.nc",
    "precip": f"{DL}/precl/0.5deg/precip.mon.mean.0.5x0.5.nc",
    "gauges": f"{DL}/precl/0.5deg/precip.mon.number_gauges.0.5x0.5.nc",
    "air": f"{DL}/ghcncams/air.mon.mean.nc",
}
USCRN_PROV = HERE / "uscrn_WA_Spokane_17_SSW_2015-2024.provenance.json"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url):
    """Download url into the cache unless a complete copy is there; return path and server facts."""
    out = CACHE / url.rsplit("/", 1)[1]
    head = requests.head(url, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    if not out.exists() or out.stat().st_size != size:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    assert out.stat().st_size == size, f"incomplete download of {url}"
    return out, {"url": url, "bytes": size, "last_modified": head.headers.get("Last-Modified"),
                 "sha256": sha256_file(out)}


def box(da):
    """The western United States box and the 1991-2020 period, lat from north to south."""
    lat0, lat1 = lab.BOX["lat"]
    lon0, lon1 = lab.BOX["lon"]
    da = da.sel(lon=slice(lon0, lon1)).sortby("lat", ascending=False).sel(lat=slice(lat1, lat0))
    if "time" in da.dims and da["time"].dtype.kind == "M":
        da = da.sel(time=slice(*lab.PERIOD))
    return da


paths, served = {}, {}
for key, url in SOURCES.items():
    paths[key], served[key] = fetch(url)
    print(f"{key:9s} {served[key]['bytes']:>11,d} bytes  {served[key]['last_modified']}  {served[key]['sha256'][:16]}")
retrieved = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")

# 1. The grid file: soil moisture and precipitation as served.
src = {k: xr.open_dataset(paths[k]) for k in ("soilw", "precip")}
soil, prec = box(src["soilw"]["soilw"]), box(src["precip"]["precip"])
assert np.array_equal(soil.lat, prec.lat) and np.array_equal(soil.lon, prec.lon), "grids differ"
assert np.array_equal(soil.time, prec.time), "time axes differ"
grid = HERE / f"{lab.TAG}.nc"
lab.write_grid(soil, prec, grid)
facts = lab.file_facts(grid)
print(json.dumps(facts, indent=1))

# 2. Static fields.
landperc = box(xr.open_dataset(paths["landperc"])["landperc"].isel(time=0, drop=True))
ltm = box(xr.open_dataset(paths["soilw_ltm"], decode_times=False)["soilw"])   # year-1 climatology dates
assert ltm.sizes["time"] == 12, "the climatology file should hold 12 calendar months"
ltm = ltm.assign_coords(time=np.arange(1, 13)).rename(time="month")
air = box(xr.open_dataset(paths["air"])["air"])
air_units = air.attrs.get("units", "")
air_clim = air.groupby("time.month").mean("time", skipna=False)
gauges = box(xr.open_dataset(paths["gauges"])["precip"])          # the gauge-count variable is named precip
static = xr.Dataset({
    "landperc": landperc.astype("f4").assign_attrs(
        long_name="percent of the cell that is land (provider file landperc.nc)", units="percent"),
    "soilw_ltm": ltm.astype("f4").assign_attrs(
        long_name="provider's 1991-2020 long-term monthly mean soil moisture (soilw.mon.ltm.1991-2020.v2.nc)", units="mm"),
    "air_clim": air_clim.astype("f4").assign_attrs(
        long_name="1991-2020 monthly mean of GHCN+CAMS air temperature, computed by the make script", units=air_units),
    "gauges_mean": gauges.mean("time").astype("f4").assign_attrs(
        long_name="mean number of rain gauges per month in the cell, 1991-2020, computed from "
                  "precip.mon.number_gauges.0.5x0.5.nc", units="gauges"),
    "gauges_frac_months_with_gauge": (gauges > 0).mean("time").astype("f4").assign_attrs(
        long_name="fraction of months 1991-2020 with at least one gauge in the cell", units="1"),
    "gauges_total": gauges.where(soil.notnull().any("time")).sum(["lat", "lon"]).astype("f4").assign_attrs(
        long_name="number of rain gauges in all model land cells of the box, each month, computed from "
                  "precip.mon.number_gauges.0.5x0.5.nc", units="gauges"),
})
static.attrs = {"title": "Static fields for the CPC soil moisture subset, western United States, 1991-2020",
                "source": "see cpc_soilw_precl_westUS_1991-2020.provenance.json"}
static_path = HERE / "cpc_westUS_static_1991-2020.nc"
enc = {v: {"zlib": False, "_FillValue": np.float32(np.nan)} for v in static.data_vars}
static.to_netcdf(static_path, format="NETCDF3_CLASSIC", encoding=enc)

# 3. The table at the model cell nearest the USCRN Spokane station (station coordinates from NCEI's station list).
row = json.loads(USCRN_PROV.read_text())["station_list_row"]
st_lat, st_lon = float(row["LATITUDE"]), float(row["LONGITUDE"])
cell = soil.sel(lat=st_lat, lon=st_lon % 360, method="nearest")
c_lat, c_lon = float(cell.lat), float(cell.lon)
pick = dict(lat=c_lat, lon=c_lon)
months = pd.DatetimeIndex(soil.time.values)
table = pd.DataFrame({
    "month": months.strftime("%Y-%m"),
    "cell_lat": c_lat, "cell_lon_east": c_lon,
    "soilw_mm": soil.sel(pick).values.round(3),
    "precip_mm_per_day": prec.sel(pick).values.round(4),
    "n_gauges": gauges.sel(pick).values.astype(int),
    f"air_temp_{air_units.replace(' ', '_') or 'units'}": air.sel(pick).values.round(3),
    "soilw_ltm_1991_2020_mm": ltm.sel(pick).sel(month=months.month).values.round(3),
})
table_path = HERE / "cpc_soilw_precl_spokane_cell_1991-2020.csv"
table.to_csv(table_path, index=False)

# 4. When the THREDDS server answers, check that the OPeNDAP subset (the students' query) gives the same file.
opendap = {"urls": lab.OPENDAP}
try:
    tmp = CACHE / "opendap_check.nc"
    msg = lab.fetch_from_provider(tmp, timeout=300)
    same = lab.fingerprint(tmp) == facts["sha256"]
    opendap.update(checked_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                   result="same bytes as the course copy" if same else "different bytes: " + msg)
except Exception as e:  # noqa: BLE001  the record says what happened
    opendap.update(checked_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                   result=f"not checked: {type(e).__name__}: {str(e)[:200]}")

files = {}
for p in (grid, static_path, table_path):
    files[p.name] = {"bytes": p.stat().st_size, "sha256": lab.fingerprint(p)}

prov = {
    "description": "CPC leaky-bucket soil moisture (V2) and the PREC/L precipitation that drives it, monthly, "
                   "0.5 degree, western United States box, 1991-2020, as served by NOAA PSL",
    "retrieved_utc": retrieved,
    "box": {"lat_north": lab.BOX["lat"][1], "lat_south": lab.BOX["lat"][0],
            "lon_west_east": lab.BOX["lon"][0], "lon_east_east": lab.BOX["lon"][1],
            "lon_convention": "degrees east, 0 to 360, as in the provider files"},
    "period": list(lab.PERIOD),
    "file": grid.name, "sha256": facts["sha256"], "bytes": facts["bytes"],
    "n_months": facts["n_months"], "n_lat": facts["n_lat"], "n_lon": facts["n_lon"],
    "n_land_cells": facts["n_land_cells"],
    "first_month": facts["first_month"], "last_month": facts["last_month"],
    "missing_value": "-9.96921e+36 (cells the provider masks as ocean); read as NaN by xarray",
    "units": {"soilw": soil.attrs.get("units"), "precip": prec.attrs.get("units"), "air": air_units},
    "served_whole_files": served,
    "files": files,
    "spokane_cell": {"station": "USCRN WA Spokane 17 SSW (WBAN 04136)", "station_lat": st_lat,
                     "station_lon": st_lon, "station_coords_source": "NCEI USCRN stations.tsv, see "
                     "uscrn_WA_Spokane_17_SSW_2015-2024.provenance.json", "cell_lat": c_lat, "cell_lon_east": c_lon},
    "opendap_check": opendap,
    "provider_revisions": [
        "CPC, https://www.cpc.ncep.noaa.gov/soilmst/descrip.htm: \"The soil moisture in the current month will be "
        "re-calculated with monthly data on the 8th of next month.\"",
        "CPC, same page: the PREC/L analysis \"will undergo the the following improvements in the near future ... "
        "Soil moisture calculations will be repeated when such improvements are implemented.\"",
        "PSL dataset description (cpcsoil): \"The dataset is now V2. There are some differences with the previous "
        "version, particularly over Africa. The V2 version also has the landmask applied to the datavalues.\"",
    ],
    "fingerprints_published_by_provider": "none found: the PSL download directories list file names, dates and "
                                          "sizes only (checked on the retrieval date)",
    "license": "PSL dataset descriptions for cpcsoil, precl and ghcncams list \"Usage Restrictions: None\". "
               "PSL asks: \"If you acquire data products from PSL, we ask that you acknowledge us in your use of the "
               "data. This may be done by including text such as data provided by the NOAA PSL, Boulder, Colorado, "
               "USA, from their website at https://psl.noaa.gov in any documents or publications using these data.\" "
               "NOAA data are in the public domain in the United States.",
    "license_sources": ["https://psl.noaa.gov/cgi-bin/mddb2/mddb2.py?action=getDataset&datasetID=46",
                        "https://psl.noaa.gov/cgi-bin/mddb2/mddb2.py?action=getDataset&datasetID=39",
                        "https://psl.noaa.gov/cgi-bin/mddb2/mddb2.py?action=getDataset&datasetID=16",
                        "https://psl.noaa.gov/data/gridded/data.cpcsoil.html"],
    "acknowledgement": "CPC Soil Moisture, PREC/L and GHCN+CAMS data provided by the NOAA PSL, Boulder, Colorado, "
                       "USA, from their website at https://psl.noaa.gov",
    "citations": [
        "van den Dool, H., J. Huang, and Y. Fan (2003), Performance and analysis of the constructed analogue method "
        "applied to U.S. soil moisture over 1981-2001, J. Geophys. Res., 108(D16), 8617, "
        "https://doi.org/10.1029/2002JD003114 (the citation PSL requests for this dataset)",
        "Fan, Y., and H. van den Dool (2004), Climate Prediction Center global monthly soil moisture data set at "
        "0.5 degree resolution for 1948 to present, J. Geophys. Res., 109, D10102, https://doi.org/10.1029/2003JD004345",
        "Huang, J., H. M. van den Dool, and K. P. Georgakakos (1996), Analysis of model-calculated soil moisture over "
        "the United States (1931-1993) and applications to long-range temperature forecasts, J. Climate, 9, "
        "1350-1362, https://doi.org/10.1175/1520-0442(1996)009<1350:AOMCSM>2.0.CO;2",
        "Chen, M., P. Xie, J. E. Janowiak, and P. A. Arkin (2002), Global land precipitation: A 50-yr monthly "
        "analysis based on gauge observations, J. Hydrometeor., 3, 249-266, "
        "https://doi.org/10.1175/1525-7541(2002)003<0249:GLPAYM>2.0.CO;2",
        "Fan, Y., and H. van den Dool (2008), A global monthly land surface air temperature analysis for "
        "1948-present, J. Geophys. Res., 113, D01103, https://doi.org/10.1029/2007JD008470",
    ],
    "dois_resolved": "each DOI above resolved through https://doi.org and api.crossref.org on the retrieval date",
}
prov_path = HERE / f"{lab.TAG}.provenance.json"
prov_path.write_text(json.dumps(prov, indent=1) + "\n")
print(json.dumps({k: prov[k] for k in ("file", "sha256", "bytes", "n_months", "n_land_cells", "files",
                                       "spokane_cell", "opendap_check")}, indent=1))
