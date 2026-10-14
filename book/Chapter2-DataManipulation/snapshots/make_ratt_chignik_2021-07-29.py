"""Build the committed RATT snapshot used in notebook 2.8.

Station UW.RATT (Pacific Northwest Seismic Network, PNSN) recorded the
2021-07-29 Mww 8.2 Chignik, Alaska, earthquake. The script downloads from the
EarthScope (formerly IRIS) DMC FDSN web services and from the USGS event
service, and writes:

- raw counts (no response removal) for UW.RATT.--.HHZ, 04:15-06:15 UTC
  (pre-event noise) and 06:15-08:15 UTC (event), as MiniSEED STEIM2 int32;
- a 2-minute three-component window (HHE, HHN, HHZ) from 06:26:30 UTC;
- the RATT StationXML at response level, as served (bytes untouched);
- the USGS event record (GeoJSON), as served;
- a provenance JSON with the URLs, sha256 per file, sample counts,
  gaps, the UW network DOI and the data-center terms.

Each MiniSEED file is read back after writing and compared sample by sample
with the downloaded trace; the script stops if any sample differs. The notebook
reads these files and checks their SHA-256 against the provenance JSON, so the
build never depends on a live download.

Regenerate (network needed): pixi run python book/Chapter2-DataManipulation/snapshots/make_ratt_chignik_2021-07-29.py
"""
import datetime as dt
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
from obspy import UTCDateTime, read, read_inventory

HERE = Path(__file__).resolve().parent
TAG = "ratt_chignik_2021-07-29"
FDSN = "https://service.earthscope.org/fdsnws"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
NET, STA, LOC = "UW", "RATT", "--"
WINDOWS = {  # name: (channels, start, end)
    "HHZ_0415-0615": ("HHZ", "2021-07-29T04:15:00", "2021-07-29T06:15:00"),
    "HHZ_0615-0815": ("HHZ", "2021-07-29T06:15:00", "2021-07-29T08:15:00"),
    "3C_062630-062830": ("HHE,HHN,HHZ", "2021-07-29T06:26:30", "2021-07-29T06:28:30"),
}
FULL_SCALE_24BIT = 2**23  # digitizer counts limit for a 24-bit ADC


def get(url):
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


files, waveforms = {}, {}

# 1. Waveforms, raw counts, round trip through STEIM2.
for name, (cha, t1, t2) in WINDOWS.items():
    url = f"{FDSN}/dataselect/1/query?" + urlencode(
        dict(net=NET, sta=STA, loc=LOC, cha=cha, start=t1, end=t2, nodata=404), safe=",:")
    downloaded = read(io.BytesIO(get(url)), format="MSEED")
    gaps = downloaded.get_gaps()
    downloaded.sort(keys=["channel"])
    for tr in downloaded:
        assert tr.data.dtype == np.int32, (tr.id, tr.data.dtype)
    out = HERE / f"{TAG}_{NET}.{STA}.{name}.mseed"
    downloaded.write(str(out), format="MSEED", encoding="STEIM2", reclen=4096)
    back = read(str(out), format="MSEED")
    back.sort(keys=["channel"])
    assert len(back) == len(downloaded)
    for a, b in zip(downloaded, back):
        assert a.id == b.id and a.stats.starttime == b.stats.starttime
        assert a.stats.sampling_rate == b.stats.sampling_rate
        assert b.data.dtype == np.int32 and np.array_equal(a.data, b.data), a.id
        assert b.stats.mseed.encoding == "STEIM2", b.stats.mseed.encoding
    traces = []
    for tr in back:
        # No peak values here: the peak and its share of full scale are what the Session 7 lab asks students
        # to find, and agents read this record first.
        traces.append({
            "id": tr.id, "starttime": str(tr.stats.starttime), "endtime": str(tr.stats.endtime),
            "sampling_rate_hz": tr.stats.sampling_rate, "npts": int(tr.stats.npts),
        })
    files[out.name] = {"url": url, "sha256": sha256(out), "bytes": out.stat().st_size}
    waveforms[out.name] = {
        "requested": [t1, t2], "encoding": "STEIM2 int32, 4096-byte records, raw counts",
        "round_trip_identical": True, "downloaded_encoding": sorted({str(t.stats.mseed.encoding) for t in downloaded}), "n_traces": len(back), "gaps": [list(map(str, g)) for g in gaps],
        "traces": traces,
    }

# 2. StationXML at response level, three HH channels, saved byte for byte.
sx_url = f"{FDSN}/station/1/query?" + urlencode(
    dict(net=NET, sta=STA, loc=LOC, cha="HH?", start="2021-07-29T04:15:00",
         end="2021-07-29T08:15:00", level="response", nodata=404), safe=",:?")
sx = HERE / f"{TAG}_{NET}.{STA}.stationxml.xml"
sx.write_bytes(get(sx_url))
inv = read_inventory(str(sx))
st0 = inv[0][0]
zch = inv.select(channel="HHZ")[0][0][0]
files[sx.name] = {"url": sx_url, "sha256": sha256(sx), "bytes": sx.stat().st_size}

# 3. USGS event record: search, then save the event's own GeoJSON as served.
search_url = USGS + "?" + urlencode(dict(format="geojson", starttime="2021-07-29T06:00:00",
                                         endtime="2021-07-29T07:00:00", minmagnitude=8))
found = json.loads(get(search_url))["features"]
assert len(found) == 1, [f["id"] for f in found]
event_id = found[0]["id"]
ev_url = USGS + "?" + urlencode(dict(eventid=event_id, format="geojson"))
ev = HERE / f"{TAG}_usgs_{event_id}.geojson"
ev.write_bytes(get(ev_url))
evj = json.loads(ev.read_bytes())
lon, lat, depth_km = evj["geometry"]["coordinates"]
files[ev.name] = {"url": ev_url, "sha256": sha256(ev), "bytes": ev.stat().st_size}

# 4. Network DOI, checked on the FDSN network page and resolved through doi.org.
fdsn_page = "https://www.fdsn.org/networks/detail/UW/"
dois = sorted(set(re.findall(r"10\.7914/[A-Za-z0-9/._-]+", get(fdsn_page).decode())))
assert dois == ["10.7914/SN/UW"], dois
citation = requests.get("https://doi.org/10.7914/SN/UW", timeout=60,
                        headers={"Accept": "text/x-bibliography; style=apa"}).text.strip()
citation = re.sub(r"</?i>", "", citation)

prov = {
    "description": "UW.RATT raw counts around the 2021-07-29 Mww 8.2 Chignik, Alaska, earthquake",
    "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    "data_center": "EarthScope Consortium DMC (formerly IRIS DMC), FDSN web services " + FDSN,
    "station": {
        "network": NET, "station": STA, "site": st0.site.name,
        "latitude": st0.latitude, "longitude": st0.longitude, "elevation_m": st0.elevation,
        "source": "StationXML from the station service (provider values)",
        "HHZ_sensitivity_counts_per_m_per_s": zch.response.instrument_sensitivity.value,
        "HHZ_sensitivity_frequency_hz": zch.response.instrument_sensitivity.frequency,
        "HHZ_sensor": zch.sensor.description if zch.sensor else None,
    },
    "network_doi": {
        "doi": "10.7914/SN/UW", "verified_on": fdsn_page,
        "dois_found_on_page": dois, "citation_from_doi_org": citation,
    },
    "event": {
        "usgs_event_id": event_id, "search_url": search_url,
        "title": evj["properties"]["title"],
        "magnitude": evj["properties"]["mag"], "magnitude_type": evj["properties"]["magType"],
        "origin_time_utc": str(UTCDateTime(evj["properties"]["time"] / 1000)),
        "latitude": lat, "longitude": lon, "depth_km": depth_km,
        "usgs_page": evj["properties"]["url"],
    },
    "waveforms": waveforms,
    "files": files,
    "terms_of_use": {
        "waveforms_and_metadata": (
            "Open FDSN data. Cite the network by its DOI (citation above) and acknowledge the "
            "facility: 'Data were accessed from the NSF NGF data archive operated by EarthScope "
            "Consortium (NSF award 2435260).' (The NSF SAGE facility became the NSF National "
            "Geophysical Facility, NGF, on 2025-10-01.)"),
        "earthscope_how_to_cite": "https://www.earthscope.org/how-to-cite/",
        "earthscope_terms_of_service": "https://www.earthscope.org/terms-of-service/",
        "usgs_event": "USGS data are in the public domain (https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)",
    },
}
(HERE / f"{TAG}.provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
print(json.dumps(prov, indent=1))
