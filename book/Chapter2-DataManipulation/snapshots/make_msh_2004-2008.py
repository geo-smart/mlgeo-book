"""Build the committed Mount St Helens snapshot used in the Session 7 volcano lab.

Mount St Helens (Washington) erupted again from September 2004, building a new
lava dome while repetitive shallow "drumbeat" earthquakes shook the crater
(Moran and others, 2008, USGS Professional Paper 1750, chapter 2). The script
downloads, and saves byte for byte as served (no parsing, no rewriting):

- comcat_msh_2004-2008.csv: every ComCat event in a box around the volcano,
  2004-01-01 to 2009-01-01 (end exclusive), all columns, from the USGS FDSN
  event service. ComCat is revised continuously, so a rebuild can differ.
- msh_UW.HSR.EHZ_2004-11-29T15-21.mseed: six hours of the vertical
  short-period seismometer HSR (South Ridge, 2.5 km south of the 2004 vent),
  raw counts, from the EarthScope (formerly IRIS) FDSN dataselect service.
- msh_UW.HSR.EHZ_response.xml: HSR's StationXML at response level for that day.
- msh_UW.HSR.EHZ_epochs.txt: every metadata epoch of HSR EHZ (text format), which
  records the factor-10 gain reduction of October 2004.
- msh_UW.EDM-JUN-SHW.EHZ_2004-11-29T2000.mseed: two minutes at three other
  crater-area stations, for checking whether a burst seen at HSR travelled
  through the ground.
- msh_2004-2008.provenance.json: URLs, retrieval time, SHA-256 and size of every
  file, row and sample counts, the ComCat /count answer, resolved DOIs, quoted
  terms of use and quoted documentation.

Each file is read back after writing (pandas, obspy) to check that it parses.
The notebook reads these files and compares their SHA-256 with this record,
so the book build never depends on a live download.

Regenerate (network needed): pixi run python book/Chapter2-DataManipulation/snapshots/make_msh_2004-2008.py
"""
import datetime as dt
import hashlib
import html
import io
import json
import re
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
from obspy import read, read_inventory

HERE = Path(__file__).resolve().parent
TAG = "msh_2004-2008"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1"
FDSN = "https://service.earthscope.org/fdsnws"
BOX = dict(minlatitude=46.10, maxlatitude=46.30, minlongitude=-122.32, maxlongitude=-122.08)
CAT_TIME = dict(starttime="2004-01-01", endtime="2009-01-01")
WAVE = ("2004-11-29T15:00:00", "2004-11-29T21:00:00")
BURST = ("2004-11-29T20:00:00", "2004-11-29T20:02:00")
COMCAT_TERMS = "https://web.archive.org/web/2023id_/https://earthquake.usgs.gov/data/comcat/index.php"


def get(url, timeout=300):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "mlgeo-book snapshot builder"})
    r.raise_for_status()
    return r


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def save(name, url, content, what):
    path = HERE / name
    path.write_bytes(content)
    files[name] = {"url": url, "sha256": sha256(content), "bytes": len(content), "what": what}
    return path


files = {}

# 1. ComCat, as served. /count first, then the query; stop if they disagree.
q = dict(BOX, **CAT_TIME)
count_url = f"{USGS}/count?{urlencode(q)}"
query_url = f"{USGS}/query?{urlencode(dict(q, format='csv', orderby='time-asc'))}"
n_count = int(get(count_url).text.strip())
cat_path = save("comcat_msh_2004-2008.csv", query_url, get(query_url, 600).content,
                "ComCat events in the box, all columns, bytes as served")
cat = pd.read_csv(cat_path)
assert len(cat) == n_count, (len(cat), n_count)

# 2. HSR waveform, as served (the archive's own MiniSEED records, not re-encoded).
def dataselect(sta, start, end):
    return f"{FDSN}/dataselect/1/query?" + urlencode(
        dict(net="UW", sta=sta, loc="--", cha="EHZ", start=start, end=end, nodata=404), safe=",:")


wave_url = dataselect("HSR", *WAVE)
wave_path = save("msh_UW.HSR.EHZ_2004-11-29T15-21.mseed", wave_url, get(wave_url).content,
                 "UW.HSR..EHZ raw counts, 2004-11-29 15:00 to 21:00 UTC, MiniSEED as served")
st = read(str(wave_path))
burst_url = dataselect("EDM,JUN,SHW", *BURST)
burst_path = save("msh_UW.EDM-JUN-SHW.EHZ_2004-11-29T2000.mseed", burst_url, get(burst_url).content,
                  "UW.EDM, UW.JUN, UW.SHW EHZ raw counts, 2004-11-29 20:00 to 20:02 UTC, as served")
st_burst = read(str(burst_path))

# 3. Station metadata, as served.
resp_url = f"{FDSN}/station/1/query?" + urlencode(
    dict(net="UW", sta="HSR", loc="--", cha="EHZ", start=WAVE[0], end=WAVE[1], level="response", nodata=404), safe=",:")
resp_path = save("msh_UW.HSR.EHZ_response.xml", resp_url, get(resp_url).content,
                 "StationXML, response level, for the waveform window")
inv = read_inventory(str(resp_path))
cha = inv[0][0][0]
epochs_url = f"{FDSN}/station/1/query?" + urlencode(
    dict(net="UW", sta="HSR", cha="EHZ", level="channel", format="text", nodata=404), safe=",:")
save("msh_UW.HSR.EHZ_epochs.txt", epochs_url, get(epochs_url).content,
     "every metadata epoch of UW.HSR..EHZ, FDSN station text format")
sites_url = f"{FDSN}/station/1/query?" + urlencode(
    dict(net="UW", sta="HSR,EDM,JUN,SHW,SEP,YEL", level="station", format="text", nodata=404), safe=",:")
sites = get(sites_url).text

# 4. Documentation and citations.
dois = {}
for doi in ["10.3133/pp1750", "10.3133/pp17502", "10.1007/s00445-015-0973-4", "10.5066/F7MS3QZH", "10.7914/SN/UW"]:
    txt = requests.get(f"https://doi.org/{doi}", timeout=60,
                       headers={"Accept": "text/x-bibliography; style=apa"}).content.decode("utf-8").strip()
    dois[doi] = re.sub(r"</?i>", "", txt)
terms_html = get(COMCAT_TERMS).text
terms_txt = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", " ", terms_html, flags=re.S))))


def term(name, n=420):
    i = terms_txt.find(f" {name} Data Type")
    return terms_txt[i + 1:i + 1 + n].strip() if i >= 0 else None


# Quotes that the notebook relies on, with the printed page of Professional Paper 1750, chapter 2.
QUOTES_PP17502 = [
    (27, "by October 16, spacing between events had become so regular that we dubbed the earthquakes “drumbeats.” Through the end of 2005 seismicity was dominated by these drumbeats"),
    (27, "Changes in drumbeat character did not correspond to variations in magma flux at the conduit, indicating that drumbeat size and spacing may be more a function of the mechanics of extrusion than of the extrusion rate."),
    (28, "More than one million earthquakes were recorded by the end of 2005"),
    (28, "roughly 8,000 events were located by the PNSN through the end of 2005"),
    (29, "Depths are relative to a datum corresponding to average altitude of stations in Mount St. Helens network, roughly 1.1 km above sea level."),
    (30, "By September 24, earthquakes had become so numerous that the PNSN began locating only a subset of them"),
    (31, "Note that peak amplitudes clip at ~3,400 counts for station HSR."),
    (33, "It also became increasingly difficult to manually pick earthquakes and determine accurate coda magnitudes"),
    (35, "The October 1 explosion destroyed seismic station SEP, the closest station to the earthquakes"),
    (37, "Seismicity in this phase was dominated by these events, which occurred at such remarkably constant intervals that we dubbed them “drumbeat” earthquakes."),
    (37, "The drumbeats consisted of repetitive LF and hybrid earthquakes"),
    (37, "The drumbeats accompanied the regular rate of extrusion of several dacite spines"),
    (37, "with daily averages ranging between 40 and 80 s"),
    (37, "HSR (2.5 km south of vent; fig. 1)"),
    (39, "between October 16 and November 12, 2004, events were occurring only ~20 s apart and were small enough that many went undetected by our detection algorithm"),
    (39, "After November 12, event spacing increased to ~40 s, and visual inspections show that most events were detected."),
    (39, "on July 20 it became irregular, with spacing progressively increasing. In early August the events had become too small to be detected reliably by our algorithm on station HSR."),
    (39, "By early October 2005, they had become so infrequent, emergent, and small that we began questioning whether the seismic events we were seeing were earthquakes or other surface sources"),
    (39, "Roughly 370,000 events were detected during this time period, including occasional noise glitches and other false triggers."),
    (40, "In general, increases in event spacing corresponded to increases in event size and vice versa (fig. 15)."),
    (42, "was dominated by LF events (peak frequencies in the 2–3-Hz range) through early January 2005"),
    (44, "drumbeats tend to have long codas that yield overly large Md values relative to magnitudes determined using amplitudes"),
    (44, "We therefore arbitrarily define larger events as those post-October 2004 events with amplitudes large enough to clip on station HSR."),
    (45, "Note that no change in drumbeat spacing or size occurs before or after the larger earthquakes, indicating that source processes are separate and distinct from one another."),
    (50, "the limited dynamic range of the short-period analog stations was frequently exceeded at seismometers in the crater and on the flanks"),
    (60, "after a gain reduction on October 4, 2004, no changes were made to the site"),
    (60, "the 100-Hz sample rate used by the PNSN and CVO for all short-period stations"),
    (60, "We chose to look only at spectra for the first 2.56 s (256 samples) following the event trigger"),
    (60, "We only show spectra for events with amplitudes greater than 300 counts (Md ~1.0), because our algorithm commonly triggered on the S-wave for events smaller than this threshold."),
    (60, "noise glitches (including telemetry and/or Internet dropouts, radio interference, and calibration pulses)"),
]
QUOTES_DZURISIN_2015 = {
    "source": "abstract of Dzurisin and others (2015), USGS Publications Warehouse record 70157103, https://pubs.usgs.gov/publication/70157103",
    "quotes": [
        "This was followed by a brief return of repetitive “drumbeat” earthquakes that characterized much of the eruption.",
        "By January 31, however, seismicity had declined to 1–2 earthquakes per day, a rate similar to pre-eruption levels."
    ]
}

prov = {
    "description": "Mount St Helens 2004-2008: ComCat events in a box around the volcano, and six hours "
                   "of the crater-area seismometer UW.HSR during dome building (29 November 2004)",
    "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    "files": files,
    "catalog": {
        "service": "USGS ANSS Comprehensive Earthquake Catalog (ComCat), FDSN event web service " + USGS,
        "query": dict(q, endtime_note="exclusive", orderby="time-asc", eventtype="all types (no filter)",
                      minmagnitude="none"),
        "count_url": count_url, "count": n_count, "n_rows": len(cat),
        "rows_match_count": len(cat) == n_count,
        "columns": list(cat.columns),
        "updated_years": {str(k): int(v) for k, v in
                          pd.to_datetime(cat["updated"]).dt.year.value_counts().sort_index().items()},
        "note": "ComCat is revised continuously (relocations, magnitude updates, deletions); the column "
                "'updated' gives each row's last revision, and a rebuild on another day can change rows "
                "and the SHA-256.",
    },
    "waveform": {
        "data_center": "EarthScope Consortium DMC (formerly IRIS DMC), FDSN web services " + FDSN,
        "requested": list(WAVE),
        "traces": [{"id": tr.id, "starttime": str(tr.stats.starttime), "endtime": str(tr.stats.endtime),
                    "sampling_rate_hz": tr.stats.sampling_rate, "npts": int(tr.stats.npts),
                    "encoding": str(tr.stats.mseed.encoding)} for tr in st],
        "gaps": [[str(g[4]), str(g[5]), round(float(g[6]), 2)] for g in st.get_gaps()],
        "burst_file_traces": [tr.id for tr in st_burst],
        "units": "raw digitizer counts; no instrument correction",
    },
    "station": {
        "source": "StationXML and station text from the station service (provider values)",
        "HSR_site": inv[0][0].site.name, "HSR_latitude": inv[0][0].latitude,
        "HSR_longitude": inv[0][0].longitude, "HSR_elevation_m": inv[0][0].elevation,
        "EHZ_sensor": cha.sensor.description if cha.sensor else None,
        "EHZ_sample_rate_hz": cha.sample_rate,
        "EHZ_sensitivity_counts_per_m_per_s": cha.response.instrument_sensitivity.value,
        "EHZ_sensitivity_frequency_hz": cha.response.instrument_sensitivity.frequency,
        "station_list_url": sites_url, "station_list": sites.strip().splitlines(),
    },
    "citations": {
        "doi_org_apa": dois,
        "note_on_pp17502": "The doi.org record lists the first author as 'Morgan, S. C.'; the chapter itself "
                           "names Seth C. Moran. Cite it as Moran, S.C., Malone, S.D., Qamar, A.I., Thelen, W.A., "
                           "Wright, A.K., and Caplan-Auerbach, J., 2008, Seismicity associated with renewed dome "
                           "building at Mount St. Helens, 2004-2005, chap. 2 of Sherrod, D.R., Scott, W.E., and "
                           "Stauffer, P.H., eds., A volcano rekindled: the renewed eruption of Mount St. Helens, "
                           "2004-2006: U.S. Geological Survey Professional Paper 1750, p. 27-60, "
                           "https://doi.org/10.3133/pp17502",
        "note_on_dzurisin_2015": "Crossref gives volume 77, issue 10, article number 89.",
        "quotes_from_pp17502": [{"page": p, "quote": q} for p, q in QUOTES_PP17502],
        "quotes_note": ("Page numbers are those printed in Professional Paper 1750; every quote was checked word for word "
                        "against the PDF of chapter 2 (pages 27 to 60)."),
        "quotes_from_dzurisin_2015": QUOTES_DZURISIN_2015,
    },
    "comcat_column_documentation": {
        "url": COMCAT_TERMS,
        "note": "The live ComCat event-terms page (https://earthquake.usgs.gov/data/comcat/data-eventterms.php) now "
                "redirects; the definitions below are quoted from the Internet Archive copy of the ComCat "
                "documentation page.",
        **{k: term(k) for k in ["time", "depth", "mag", "magType", "nst", "gap", "dmin", "rms",
                                "horizontalError", "depthError", "updated", "type"]},
    },
    "terms_of_use": {
        "comcat": "USGS data are in the public domain (https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)",
        "waveforms_and_metadata": (
            "Open FDSN data. Cite the network by its DOI (10.7914/SN/UW, citation above) and acknowledge the "
            "facility: 'Data were accessed from the NSF NGF data archive operated by EarthScope "
            "Consortium (NSF award 2435260).' (The NSF SAGE facility became the NSF National "
            "Geophysical Facility, NGF, on 2025-10-01.)"),
        "earthscope_how_to_cite": "https://www.earthscope.org/how-to-cite/",
        "earthscope_terms_of_service": "https://www.earthscope.org/terms-of-service/",
    },
}
(HERE / f"{TAG}.provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in prov.items() if k not in ("comcat_column_documentation",)}, indent=1,
                 ensure_ascii=False)[:6000])
