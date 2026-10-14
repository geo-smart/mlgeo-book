"""Build the committed GHCN-Daily snapshot used in notebooks 2.7 and 2.8 and the lecture 7 figures.

Seattle-Tacoma International Airport (GHCN-Daily station USW00024233), daily
TMAX, TMIN, PRCP and AWND with their measurement (M), quality (Q) and source
(S) flags, 1948-01-01 to 2024-12-31, as served by the NOAA Open Data
Dissemination (NODD) bucket on AWS. Values stay in native GHCN units (tenths
of degC, tenths of mm, tenths of m/s), so the unit check in the notebook is
real. One row per date that has at least one of the four elements; dates with
none of them are absent, not filled. The notebooks read the CSV and check its
SHA-256 against provenance.json, so the build never depends on a live download.

Regenerate (network needed): pixi run python book/Chapter2-DataManipulation/snapshots/make_ghcnd_USW00024233_1948-2024.py
"""
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests

STATION, START, END = "USW00024233", "1948-01-01", "2024-12-31"
ELEMENTS = ["TMAX", "TMIN", "PRCP", "AWND"]
BUCKET = "https://noaa-ghcn-pds.s3.amazonaws.com"
HERE = Path(__file__).resolve().parent

data_url = f"{BUCKET}/csv.gz/by_station/{STATION}.csv.gz"
stations_url = f"{BUCKET}/ghcnd-stations.txt"

resp = requests.get(data_url, timeout=300)
resp.raise_for_status()
raw = pd.read_csv(io.BytesIO(gzip.decompress(resp.content)), header=None, dtype=str, keep_default_na=False,
                  names=["id", "date", "element", "value", "m", "q", "s", "obs_time"])
raw = raw[raw["element"].isin(ELEMENTS)]
raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d")
raw = raw[(raw["date"] >= START) & (raw["date"] <= END)]

wide = raw.pivot(index="date", columns="element", values=["value", "m", "q", "s"])
cols = {}
for el in ELEMENTS:
    cols[el] = pd.to_numeric(wide[("value", el)]).astype("Int64")
    for f in ["m", "q", "s"]:
        cols[f"{el}_{f.upper()}"] = wide[(f, el)].replace("", pd.NA)
table = pd.DataFrame(cols).sort_index()
table.index = table.index.strftime("%Y-%m-%d")
table.index.name = "date"

out = HERE / f"ghcnd_{STATION}_1948-2024.csv"
table.to_csv(out)
sha = hashlib.sha256(out.read_bytes()).hexdigest()



def station_line_from(url, station, chunk=2048):
    """Find one line of the 11 MB ghcnd-stations.txt with HTTP Range requests.

    The file is sorted by station ID, so a binary search on byte offsets reads
    a few kB instead of the whole file.
    """
    size = int(requests.head(url, timeout=60).headers["Content-Length"])
    lo, hi = 0, size
    while hi - lo > chunk:
        mid = (lo + hi) // 2
        text = requests.get(url, headers={"Range": f"bytes={mid}-{mid + chunk}"}, timeout=60).text
        first_full = text.split("\n")[1]
        if first_full[:11] < station:
            lo = mid
        else:
            hi = mid
    text = requests.get(url, headers={"Range": f"bytes={lo}-{hi + 2 * chunk}"}, timeout=60).text
    return next(ln for ln in text.split("\n") if ln.startswith(station))


station_line = station_line_from(stations_url, STATION)

all_days = pd.date_range(START, END, freq="D")
prov = {
    "station": STATION,
    "station_line_ghcnd_stations_txt": station_line,
    "station_line_format": "ID(1-11) LATITUDE(13-20) LONGITUDE(22-30) ELEVATION_m(32-37) STATE(39-40) NAME(42-71) GSN(73-75) HCN/CRN(77-79) WMO(81-85), from ghcnd-stations.txt",
    "latitude": float(station_line[12:20]), "longitude": float(station_line[21:30]),
    "elevation_m": float(station_line[31:37]), "name": station_line[41:71].strip(),
    "urls": {"data": data_url, "stations": stations_url,
             "documentation": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt"},
    "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    "source_last_modified": resp.headers.get("Last-Modified"),
    "file": out.name, "sha256": sha, "n_rows": len(table),
    "date_range": [table.index[0], table.index[-1]],
    "n_calendar_days_in_range": len(all_days),
    "n_calendar_days_absent": len(all_days) - len(table),
    "n_values": {el: int(table[el].notna().sum()) for el in ELEMENTS},
    "first_date_with_value": {el: table[el].dropna().index[0] for el in ELEMENTS},
    "columns": {
        "date": "observation day, YYYY-MM-DD, as given in the GHCN DATE field",
        "TMAX": "daily maximum temperature, tenths of degC (native GHCN units)",
        "TMIN": "daily minimum temperature, tenths of degC (native GHCN units)",
        "PRCP": "daily precipitation, tenths of mm (native GHCN units)",
        "AWND": "average daily wind speed, tenths of m/s (native GHCN units)",
        "<EL>_M": "measurement flag, from readme.txt: blank = no measurement information applicable; T = trace of precipitation (the only non-blank M flag for these four elements at this station)",
        "<EL>_Q": "quality flag, from readme.txt: blank = did not fail any quality assurance check; O = failed climatological outlier check (the only non-blank Q flag here)",
        "<EL>_S": "source flag, from readme.txt: 0 = U.S. Cooperative Summary of the Day (NCDC DSI-3200); D = short time delay US National Weather Service CF6 daily summaries; W = WBAN/ASOS Summary of the Day from NCDC's Integrated Surface Data; X = U.S. First-Order Summary of the Day (NCDC DSI-3210); Z = Datzilla official additions or replacements",
        "missing value": "empty cell = element not reported that day",
    },
    "license": "CC0-1.0. Quoted from the Registry of Open Data on AWS page https://registry.opendata.aws/noaa-ghcn/ : "
               "\"NOAA data disseminated through NODD is made available under the Creative Commons 1.0 Universal Public Domain "
               "Dedication (CC0-1.0) license, which is well-known and internationally recognized. There are no restrictions "
               "on the use of the data.\" NOAA requests attribution for unaltered data and asks that modified data not be "
               "presented as original, unaltered NOAA data. This file is a reshaped subset (long to wide, four elements, date window).",
    "citation": "Menne, M.J., I. Durre, B. Korzeniewski, S. McNeal, K. Thomas, X. Yin, S. Anthony, R. Ray, R.S. Vose, B.E. Gleason, "
                "and T.G. Houston (2012): Global Historical Climatology Network - Daily (GHCN-Daily), Version 3. NOAA National "
                "Climatic Data Center. doi:10.7289/V5D21VHZ",
}
(HERE / f"ghcnd_{STATION}_1948-2024.provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
print(json.dumps(prov, indent=1))
