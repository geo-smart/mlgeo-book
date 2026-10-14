"""Build the committed NCEP/NCAR Reanalysis 1 snapshot used in the Session 7 lab (station E).

Air temperature at the lowest model level (sigma 0.995), every 6 hours, 2013-01-01
to 2014-12-31, on the 2.5 degree grid over North America (lat 75 to 15 N, lon 200
to 330 E). The bytes are the file notebook 2.5 pins: `air_temperature.nc` from
the Xarray project's example-data repository (pydata/xarray-data) at commit
2baf0c2, a subset the Xarray project cut from the NOAA Physical Sciences
Laboratory (PSL) archive. The file is copied unchanged; this script refuses
bytes whose SHA-256 differs from the one 2.5 pins. The lab reads the file and
checks its SHA-256 against provenance.json, so it never depends on a live download.

Regenerate (network needed): pixi run python book/Chapter2-DataManipulation/snapshots/make_ncep_air_2013-2014.py
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

import requests
import xarray as xr

COMMIT = "2baf0c22d9671a2058415f17e71b3fe06239a3dd"
URL = f"https://github.com/pydata/xarray-data/raw/{COMMIT}/air_temperature.nc"
SHA256 = "c606b89c35970a2983b914b76df4adbb409003ef34aa7cfd7f582e41f307482b"   # pinned in 2.5_Arrays.ipynb
HERE = Path(__file__).resolve().parent
OUT = HERE / "ncep_air_2013-2014.nc"

resp = requests.get(URL, timeout=300)
resp.raise_for_status()
sha = hashlib.sha256(resp.content).hexdigest()
if sha != SHA256:
    raise SystemExit(f"downloaded bytes differ from the pinned file: {sha} != {SHA256}")
OUT.write_bytes(resp.content)

ds = xr.open_dataset(OUT)
raw = xr.open_dataset(OUT, mask_and_scale=False)
air = ds["air"]
prov = {
    "dataset": "NCEP/NCAR Reanalysis 1, 4x daily air temperature at sigma 0.995 (variable air.sig995)",
    "provider": "NOAA Physical Sciences Laboratory (PSL), Boulder, Colorado, USA",
    "provider_page": "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html",
    "distributor": "Xarray project example data, https://github.com/pydata/xarray-data "
                   "(subset in time and space cut by the Xarray project; the repository states no license)",
    "url": URL,
    "commit": COMMIT,
    "commit_date_utc": "2015-08-21T20:27:11Z",
    "pinned_in": "book/Chapter2-DataManipulation/2.5_Arrays.ipynb (RAW_URL, RAW_SHA256)",
    "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    "file": OUT.name, "sha256": sha, "bytes": OUT.stat().st_size,
    "format": "netCDF classic (NETCDF3_CLASSIC), COARDS conventions",
    "dims": {k: int(v) for k, v in air.sizes.items()},
    "time": {"first": str(ds.time.values[0])[:19], "last": str(ds.time.values[-1])[:19],
             "step": "6 h", "units_on_disk": ds.time.encoding.get("units")},
    "lat_first_last_deg_north": [float(ds.lat[0]), float(ds.lat[-1])],
    "lon_first_last_deg_east": [float(ds.lon[0]), float(ds.lon[-1])],
    "air_on_disk": {"dtype": str(raw["air"].dtype), "scale_factor": float(raw["air"].attrs["scale_factor"]),
                    "add_offset": raw["air"].attrs.get("add_offset"),
                    "_FillValue": raw["air"].attrs.get("_FillValue"),
                    "missing_value": raw["air"].attrs.get("missing_value"),
                    "units": raw["air"].attrs["units"],
                    "actual_range_attribute": [round(float(v), 2) for v in raw["air"].attrs["actual_range"]]},
    "air_decoded_range_K": [round(float(air.min()), 2), round(float(air.max()), 2)],
    "global_attributes": {k: str(v) for k, v in ds.attrs.items()},
    "usage": "PSL dataset page, USAGE & CITATION panel (https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html, "
             "text served by https://psl.noaa.gov/cgi-bin/mddb2/mddb2.py?action=getDataset&datasetID=29, read 2026-10-14 UTC): "
             "\"Usage Restrictions: None\". Acknowledgement requested on the same page: \"If you acquire [NCEP-NCAR "
             "Reanalysis 1] data products from PSL, we ask that you acknowledge us in your use of the data. This may be "
             "done by including text such as [NCEP-NCAR Reanalysis 1] data provided by the NOAA PSL, Boulder, Colorado, "
             "USA, from their website at https://psl.noaa.gov in any documents or publications using these data.\"",
    "citation_psl": "Kalnay et al.,The NCEP/NCAR 40-year reanalysis project, Bull. Amer. Meteor. Soc., 77, 437-470, 1996 "
                    "(as given on the PSL page; Crossref lists pages 437-471)",
    "citation": "Kalnay, E., M. Kanamitsu, R. Kistler, et al. (22 authors) (1996). The NCEP/NCAR 40-Year Reanalysis "
                "Project. Bulletin of the American Meteorological Society, 77(3), 437-471. "
                "doi:10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2",
    "doi_verified": "resolved on Crossref (https://api.crossref.org/works/10.1175/1520-0477(1996)077%3C0437:TNYRP%3E2.0.CO;2) "
                    "on 2026-10-14 UTC: title, journal, volume 77, issue 3, pages 437-471, March 1996, 22 authors",
    "notes": [
        "air is stored as int16 with scale_factor 0.01 and no add_offset, _FillValue or missing_value attribute; "
        "xarray decodes it to float64 kelvin.",
        "actual_range (185.16 to 322.1 K) is wider than the values in this subset; it most likely carries over from the PSL source file and does not describe this subset.",
        "Longitudes run 0 to 360 degrees east; latitudes run north to south.",
    ],
}
(HERE / "ncep_air_2013-2014.provenance.json").write_text(json.dumps(prov, indent=1, default=str) + "\n")
print(json.dumps(prov, indent=1, default=str))
