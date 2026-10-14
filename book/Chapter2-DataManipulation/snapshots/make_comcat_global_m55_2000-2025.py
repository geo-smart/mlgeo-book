"""Build the committed worldwide M >= 5.5 ComCat snapshot used in notebook 2.7, Level 3.

USGS Advanced National Seismic System Comprehensive Catalog (ComCat), FDSN
event web service: all events of type earthquake with magnitude >= 5.5
worldwide, origin times 2000-01-01 to 2026-01-01 (end exclusive). The service
caps one query at 20,000 events, so the request runs in chunks; for each chunk
the script records the /count answer and stops if the rows returned differ
from it. Columns are trimmed to time, latitude, longitude, depth, mag,
magType, net, id; values are written exactly as the service returns them (no
rounding). The notebook reads the CSV and checks its SHA-256 against
provenance.json, so the build never depends on a live download. ComCat is
revised continuously, so a rebuild can differ.

Regenerate (network needed): pixi run python book/Chapter2-DataManipulation/snapshots/make_comcat_global_m55_2000-2025.py
"""
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

NAME = "comcat_global_m55_2000-2025"
BASE = "https://earthquake.usgs.gov/fdsnws/event/1"
QUERY = dict(eventtype="earthquake", minmagnitude=5.5)
CHUNKS = [("2000-01-01", "2013-01-01"), ("2013-01-01", "2026-01-01")]
COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "magType", "net", "id"]
MW_FAMILY = {"mw", "mwb", "mwc", "mwr", "mww", "mwp", "mi", "mwbe", "mws"}
HERE = Path(__file__).resolve().parent

parts, chunk_log = [], []
for start, end in CHUNKS:
    q = dict(QUERY, starttime=start, endtime=end)
    count_url = f"{BASE}/count?{urlencode(q)}"
    query_url = f"{BASE}/query?{urlencode(dict(q, format='csv', orderby='time-asc'))}"
    n_count = int(requests.get(count_url, timeout=120).text.strip())
    if n_count >= 20000:
        raise RuntimeError(f"chunk {start} to {end} has {n_count} events, at or above the 20,000 limit")
    r = requests.get(query_url, timeout=600)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), dtype=str, keep_default_na=False)[COLUMNS]
    if len(df) != n_count:
        raise RuntimeError(f"chunk {start} to {end}: /count says {n_count}, query returned {len(df)}")
    parts.append(df)
    chunk_log.append({"starttime": start, "endtime": end, "count": n_count, "rows": len(df),
                      "count_url": count_url, "query_url": query_url})
    print(start, end, n_count, len(df))

cat = pd.concat(parts, ignore_index=True)
if cat["id"].duplicated().any():
    raise RuntimeError(f"{cat['id'].duplicated().sum()} duplicate ids across chunk boundaries")
total_url = f"{BASE}/count?{urlencode(dict(QUERY, starttime=CHUNKS[0][0], endtime=CHUNKS[-1][1]))}"
n_total = int(requests.get(total_url, timeout=120).text.strip())

out = HERE / f"{NAME}.csv"
cat.to_csv(out, index=False)
sha = hashlib.sha256(out.read_bytes()).hexdigest()
mt = cat["magType"].str.lower()
prov = {
    "catalog": "USGS ANSS Comprehensive Earthquake Catalog (ComCat), FDSN event web service",
    "query": dict(QUERY, starttime=CHUNKS[0][0], endtime=CHUNKS[-1][1], endtime_note="exclusive", orderby="time-asc"),
    "chunks": chunk_log,
    "total_count_url": total_url, "total_count": n_total,
    "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    "file": out.name, "sha256": sha, "n_rows": len(cat),
    "rows_match_count": len(cat) == n_total,
    "n_negative_depth": int((pd.to_numeric(cat["depth"]) < 0).sum()),
    "magType_counts": {k: int(v) for k, v in cat["magType"].replace("", "(blank)").value_counts().items()},
    "n_mw_family": int(mt.isin(MW_FAMILY).sum()),
    "mw_family_definition": "magType (lower case) in " + ", ".join(sorted(MW_FAMILY)),
    "columns": {
        "time": "origin time, UTC, ISO 8601",
        "latitude": "epicenter latitude, decimal degrees (WGS84)",
        "longitude": "epicenter longitude, decimal degrees (WGS84)",
        "depth": "hypocenter depth, km, positive down, as served; negative values place the hypocenter above the depth datum and are kept",
        "mag": "preferred magnitude, as served (not rounded)",
        "magType": "magnitude type of the preferred magnitude (e.g. mww W-phase moment magnitude, mb body-wave, ms surface-wave)",
        "net": "network that contributed the preferred origin (e.g. us = USGS National Earthquake Information Center)",
        "id": "ComCat event id (net + network event code)",
    },
    "documentation": "https://earthquake.usgs.gov/fdsnws/event/1/ and https://earthquake.usgs.gov/data/comcat/",
    "license": "USGS data are in the public domain (https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)",
    "note": "ComCat is revised continuously (relocations, magnitude updates, deletions); a rebuild on another day can change rows and the hash.",
}
(HERE / f"{NAME}.provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
print(json.dumps({k: v for k, v in prov.items() if k != "chunks"}, indent=1))
