"""Build the committed USCRN soil moisture snapshot used in the Session 7 soil moisture lab.

Station WA Spokane 17 SSW (WBAN 04136) of the U.S. Climate Reference Network
(USCRN), operated by NOAA's National Centers for Environmental Information
(NCEI), sits at the headquarters of Turnbull National Wildlife Refuge, south
of Spokane, Washington. It reports daily air temperature, precipitation and
soil moisture and temperature at 5, 10, 20, 50 and 100 cm.

The script writes:

- uscrn_WA_Spokane_17_SSW_2015-2024.txt: the ten yearly daily01 files
  (2015 to 2024) exactly as served, concatenated in year order. No byte is
  changed; the file has no header line because the provider's files have none.
- uscrn_daily01_HEADERS.txt and uscrn_daily01_readme.txt: the provider's
  column names, units and notes (including the missing-value codes), as served.
- naturalearth_pnw_map.csv: Natural Earth 1:50 million coastlines, state and
  province borders and three populated places, clipped to the Pacific
  Northwest, so the lab can draw a map offline with matplotlib only.
- uscrn_WA_Spokane_17_SSW_2015-2024.provenance.json: URLs, retrieval time,
  sha256 of every served file and of every written file, row counts, the
  station's line in the network station list, the terms of use quoted
  from the provider pages, and the sensor documentation the lab quotes
  (URL, sha256, and whether each quote was found in the page text).

Regenerate (network needed): pixi run python book/Chapter2-DataManipulation/snapshots/make_uscrn_WA_Spokane_17_SSW_2015-2024.py
"""
import datetime as dt
import hashlib
import html
import io
import json
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box

HERE = Path(__file__).resolve().parent
STATION, WBAN = "WA_Spokane_17_SSW", "04136"
YEARS = range(2015, 2025)
TAG = f"uscrn_{STATION}_{YEARS[0]}-{YEARS[-1]}"
BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products"
ISO = "https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00959/xml"
POLICY = ("https://www.ncei.noaa.gov/sites/default/files/2023-12/"
          "NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf")
NE = "https://naciscdn.org/naturalearth/50m/cultural/{}.zip"
NE_COAST = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"
NE_TERMS = "https://www.naturalearthdata.com/about/terms-of-use/"
MAP_BOX = (-126.0, 44.0, -114.0, 50.5)        # lon min, lat min, lon max, lat max
PLACES = ["Seattle", "Spokane", "Portland"]


def get(url):
    r = requests.get(url, timeout=300, headers={"User-Agent": "mlgeo-book snapshot builder"})
    r.raise_for_status()
    return r


def sha256(b):
    return hashlib.sha256(b).hexdigest()


files, served = {}, {}

# 1. The ten yearly files, as served, then concatenated without change.
chunks, rows_per_year = [], {}
for y in YEARS:
    url = f"{BASE}/daily01/{y}/CRND0103-{y}-{STATION}.txt"
    r = get(url)
    lines = r.content.decode("ascii").splitlines()
    assert all(ln[:5] == WBAN and ln[6:10] == str(y) for ln in lines), url
    n_days = 366 if y % 4 == 0 else 365
    assert len(lines) == n_days, (url, len(lines))
    rows_per_year[y] = len(lines)
    served[url] = {"sha256": sha256(r.content), "bytes": len(r.content),
                   "last_modified": r.headers.get("Last-Modified")}
    chunks.append(r.content)
data = HERE / f"{TAG}.txt"
data.write_bytes(b"".join(chunks))
files[data.name] = {"sha256": sha256(data.read_bytes()), "bytes": data.stat().st_size,
                    "what": "the ten served yearly files, concatenated in year order, bytes unchanged"}

# 2. Provider documentation, as served.
for name, out in [("HEADERS.txt", "uscrn_daily01_HEADERS.txt"), ("README.txt", "uscrn_daily01_readme.txt")]:
    r = get(f"{BASE}/daily01/{name}")
    (HERE / out).write_bytes(r.content)
    files[out] = {"sha256": sha256(r.content), "bytes": len(r.content), "url": f"{BASE}/daily01/{name}"}
readme = (HERE / "uscrn_daily01_readme.txt").read_text()
missing_note = re.search(r"C\.\s+Missing data are indicated.*?places\.", readme, re.S).group(0)
flag_note = re.search(r"E\.\s+There are no quality flags.*?missing\.", readme, re.S).group(0)

# 3. The station's line in the network station list (provider values for name and place).
st_url = f"{BASE}/stations.tsv"
st_lines = get(st_url).text.splitlines()
header = st_lines[0].split("\t")
row = next(ln.split("\t") for ln in st_lines[1:] if ln.startswith(WBAN + "\t"))
station_row = dict(zip(header, row))

# 4. Terms of use, quoted from the provider pages.
iso = get(ISO).text
constraints = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m)).strip()
               for m in re.findall(r"<gmd:otherConstraints>(.*?)</gmd:otherConstraints>", iso, re.S)]
cite = next(c for c in constraints if c.startswith("Cite as:"))
liability = [c for c in constraints if "liability" in c.lower()]
policy_pdf = get(POLICY).content
POLICY_QUOTE = ("Environmental data and information produced by NOAA or any Federal agency are available fully "
                "and openly to data users unless explicitly exempt from release by Federal law, regulation or policy. "
                "These data are in the public domain in the United States. To serve our international partners and "
                "customers, NCEI will work to ensure a Creative Commons CC0 license is applied.")

# 5. Natural Earth map lines and places, clipped to the Pacific Northwest.
def ne_read(url, known=None):
    """Download a Natural Earth zip (hash-checked when a hash is known) and read its shapefile."""
    blob = get(url).content
    if known:
        assert sha256(blob) == known, url
    served[url] = {"sha256": sha256(blob), "bytes": len(blob)}
    shp = next(n for n in zipfile.ZipFile(io.BytesIO(blob)).namelist() if n.endswith(".shp"))
    tmp = HERE / ".ne_tmp.zip"
    tmp.write_bytes(blob)
    g = gpd.read_file(f"zip://{tmp}!{shp}")
    tmp.unlink()
    return g


clip = box(*MAP_BOX)
map_rows = []
for kind, url, known in [
    ("coastline", NE_COAST, "640f805509b822f57f4840a2e18d9ff2412cf1cf6976124701c2789436166fde"),
    ("border", NE.format("ne_50m_admin_1_states_provinces_lines"), None),
    ("border", NE.format("ne_50m_admin_0_boundary_lines_land"), None),
]:
    g = ne_read(url, known)
    for geom in g.geometry.intersection(clip):
        parts = getattr(geom, "geoms", [geom])
        for part in parts:
            if part.is_empty or part.geom_type != "LineString":
                continue
            pid = len({r[1] for r in map_rows})
            map_rows += [(kind, pid, round(x, 4), round(y, 4), "") for x, y in part.coords]
places = ne_read(NE.format("ne_50m_populated_places_simple"))
for name in PLACES:
    p = places[(places["name"] == name) & (places["adm0name"] == "United States of America")].iloc[0]
    map_rows.append(("place", -1, round(p.geometry.x, 4), round(p.geometry.y, 4), name))
mp = HERE / "naturalearth_pnw_map.csv"
mp.write_text("kind,part,lon,lat,name\n" + "".join(f"{k},{i},{x},{y},{n}\n" for k, i, x, y, n in map_rows))
files[mp.name] = {"sha256": sha256(mp.read_bytes()), "bytes": mp.stat().st_size,
                  "what": f"Natural Earth 1:50m coastline, state/province and country borders clipped to "
                          f"lon {MAP_BOX[0]} to {MAP_BOX[2]}, lat {MAP_BOX[1]} to {MAP_BOX[3]}; places {PLACES}"}
try:
    ne_terms_page = get(NE_TERMS).text
    ne_terms_checked = "in the public domain" in ne_terms_page
except requests.RequestException as e:
    ne_terms_checked = f"page not reachable from this script ({type(e).__name__}); quote checked in a browser"

# 6. Sensor documentation quoted in the Session 7 soil moisture lab (not stored; URL, hash and quotes only).
DOCS = {
    "ncei_measurements_page": ("https://www.ncei.noaa.gov/access/crn/measurements.html", [
        "a total of 15 Stevens Water Monitoring Systems, Inc., Hydra Probe II (SDI-12) units are placed in the "
        "ground in three plots at five depths (5, 10, 20, 50, and 100 cm)",
        "All 15 probes are interrogated every two minutes and their measurements averaged over 5-minute periods "
        "for output purposes."]),
    "hourly02_readme": (f"{BASE}/hourly02/README.txt", [
        "USCRN stations have multiple co-located soil sensors that record independent measurements at 5-minute "
        "intervals."]),
    "ncei_soil_moisture_correction_2022": (
        "https://www.ncei.noaa.gov/pub/data/uscrn/publications/events/"
        "Correction%20to%20Volumetric%20Soil%20Moisture%2010202022.pdf", [
            "Volumetric = (0.109 * SQRT(Dielectric)) -0.179",
            "rerun for all stations and the period of record in our database"]),
    "uscrn_tech_memo_19_001": (
        "https://www.ncei.noaa.gov/pub/data/uscrn/documentation/research/NOAA-Tech-Memo-Soil-FINAL-AUGUST%202019.pdf", [
            "The Hydra probe lacks the ability to measure water content in frozen soils (printed page 16)"]),
    "stevens_hydraprobe_page": ("https://stevenswater.com/products/hydraprobe/", [
        'Sensing volume (cylindrical region) Length: 2.2" (5.7cm) Diameter: 1.2" (3.0cm)']),
}


# Title registered with DataCite for doi:10.7289/V5MS3QR9 (https://api.datacite.org/dois/10.7289/V5MS3QR9, checked
# 2026-10-14); the notebook cites this title, while "citation" keeps the dataset record's own "Cite as" text.
DATACITE_TITLE = ("United States Climate Reference Network (USCRN) Processed Data from the Version 2 USCRN Database "
                  "(title registered with DataCite for doi:10.7289/V5MS3QR9, checked at "
                  "https://api.datacite.org/dois/10.7289/V5MS3QR9 on 2026-10-14; the notebook cites this title, while "
                  "'citation' above keeps the dataset record's own 'Cite as' text)")


def page_text(blob):
    t = blob.decode("utf-8", errors="ignore")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


documentation = {}
for key, (url, quotes) in DOCS.items():
    try:
        r = requests.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0 (mlgeo-book snapshot builder)"})
        r.raise_for_status()
        rec = {"url": url, "sha256": sha256(r.content), "bytes": len(r.content), "quotes": quotes}
        if url.endswith(".pdf"):
            rec["quotes_checked"] = "PDF: read with pdftotext on 2026-10-14 (not checked by this script)"
        else:
            text = page_text(r.content)
            rec["quotes_found_in_page_text"] = [q.split(" (printed")[0] in text for q in quotes]
    except requests.RequestException as e:
        rec = {"url": url, "quotes": quotes, "error": f"{type(e).__name__}: not reachable from this script"}
    documentation[key] = rec

prov = {
    "description": f"USCRN daily01 files for {STATION} (WBAN {WBAN}), {YEARS[0]} to {YEARS[-1]}, as served",
    "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    "file": data.name, "sha256": files[data.name]["sha256"],
    "n_rows": sum(rows_per_year.values()), "rows_per_year": rows_per_year,
    "format": "fixed-width text, one row per Local Standard Time (LST) day, 28 fields, no header line; "
              "names and units in uscrn_daily01_HEADERS.txt, definitions in uscrn_daily01_readme.txt",
    "missing_value_note_from_readme": re.sub(r"\s+", " ", missing_note),
    "quality_flag_note_from_readme": re.sub(r"\s+", " ", flag_note),
    "station_list_url": st_url, "station_list_row": station_row,
    "served": served, "files": files,
    "license": "Public domain in the United States. Quoted from the NCEI Open Data Policy (PD-10-2-02), section a, "
               f"{POLICY.replace('%20', ' ')} : \"{POLICY_QUOTE}\" The dataset record adds: "
               + " ".join(f"\"{c}\"" for c in liability),
    "license_policy_pdf_sha256": sha256(policy_pdf),
    "license_quote_checked": "read in the PDF text (section 5a) on the retrieval date; the script records the PDF's sha256",
    "citation": cite.removeprefix("Cite as: ").replace("[indicate subset used]", f"[daily01, {STATION}, {YEARS[0]}-{YEARS[-1]}]")
                .replace("[indicate access date]", f"[accessed {dt.datetime.now(dt.timezone.utc).date().isoformat()}]"),
    "citation_datacite_title": DATACITE_TITLE,
    "citation_soil": "Bell, J.E., M.A. Palecki, C.B. Baker, W.G. Collins, J.H. Lawrimore, R.D. Leeper, M.E. Hall, "
                     "J. Kochendorfer, T.P. Meyers, T. Wilson, and H.J. Diamond (2013): U.S. Climate Reference Network "
                     "soil moisture and temperature observations. J. Hydrometeorology, 14, 977-988, "
                     "doi:10.1175/JHM-D-12-0146.1 (requested by the USCRN program in the dataset record " + ISO + ")",
    "map_license": "Natural Earth, " + NE_TERMS + ": \"All versions of Natural Earth raster + vector map data found on "
                   "this website are in the public domain.\" \"No permission is needed to use Natural Earth.\"",
    "map_terms_page_checked_by_script": ne_terms_checked,
    "sensor_documentation": documentation,
}
(HERE / f"{TAG}.provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
print(json.dumps({k: v for k, v in prov.items() if k not in ("served",)}, indent=1))
