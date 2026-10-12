"""Build the committed USGS snapshot used in notebook 2.6, section 3.1b.

Nooksack River at Ferndale, WA (USGS 12213100), discharge (parameter 00060),
1-30 November 2021: instantaneous values (iv) and daily mean values (dv,
statistic 00003), as served by USGS Water Services, plus the site location.
The notebook reads the CSV and checks its SHA-256 against provenance.json,
so the build never depends on a live download.

Regenerate (network needed): pixi run python book/Chapter2-DataManipulation/snapshots/make_usgs_12213100_2021-11.py
"""
import datetime as dt
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests

SITE, START, END = "12213100", "2021-11-01", "2021-11-30"
BASE = "https://nwis.waterservices.usgs.gov/nwis"
HERE = Path(__file__).resolve().parent


def rdb(url):
    lines = [ln for ln in requests.get(url, timeout=120).text.splitlines() if not ln.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)


iv_url = f"{BASE}/iv/?format=rdb&sites={SITE}&startDT={START}&endDT={END}&parameterCd=00060"
dv_url = f"{BASE}/dv/?format=rdb&sites={SITE}&startDT={START}&endDT={END}&parameterCd=00060&statCd=00003"
site_url = f"{BASE}/site/?format=rdb&sites={SITE}&siteOutput=expanded"

iv, dv, site = rdb(iv_url), rdb(dv_url), rdb(site_url)
qiv = [c for c in iv.columns if c.endswith("_00060")][0]
qdv = [c for c in dv.columns if c.endswith("_00060_00003")][0]
rows = pd.concat([
    pd.DataFrame({"product": "instantaneous", "time_local": iv["datetime"], "tz": iv["tz_cd"],
                  "discharge_cfs": iv[qiv].astype(float), "approval": iv[qiv + "_cd"]}),
    pd.DataFrame({"product": "daily_mean", "time_local": dv["datetime"], "tz": "local day",
                  "discharge_cfs": dv[qdv].astype(float), "approval": dv[qdv + "_cd"]}),
])
out = HERE / f"usgs_{SITE}_2021-11.csv"
rows.to_csv(out, index=False)
sha = hashlib.sha256(out.read_bytes()).hexdigest()
prov = {
    "site": SITE, "site_name": site["station_nm"][0],
    "latitude": float(site["dec_lat_va"][0]), "longitude": float(site["dec_long_va"][0]),
    "datum": site["dec_coord_datum_cd"][0], "drainage_area_mi2": float(site["drain_area_va"][0]),
    "parameter": "00060 discharge, cubic feet per second",
    "urls": {"instantaneous": iv_url, "daily_mean": dv_url, "site": site_url},
    "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    "file": out.name, "sha256": sha, "n_instantaneous": int((rows["product"] == "instantaneous").sum()),
    "n_daily": int((rows["product"] == "daily_mean").sum()),
    "license": "USGS data are in the public domain (https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)",
}
(HERE / "usgs_12213100_2021-11.provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
print(json.dumps(prov, indent=1))
