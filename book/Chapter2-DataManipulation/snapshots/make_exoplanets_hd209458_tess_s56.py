"""Build the committed course copies for the Session 7 planetary science lab (lab_s07_planetary.ipynb).

The system: HD 209458, a Sun-like star 48 parsecs away in Pegasus, and its giant planet HD 209458 b, the
first planet seen to pass in front of its star (Charbonneau et al. 2000; Henry et al. 2000). NASA's
Transiting Exoplanet Survey Satellite (TESS) watched it for 28 days in sector 56 (September 2022).

The script writes, in this folder:

- nea_pscomppars_<date>.csv: the NASA Exoplanet Archive Planetary Systems Composite Parameters table
  (one row per confirmed planet), exactly as the archive's Table Access Protocol (TAP) service returns it
  for the query lab_s07_planetary_checks.PS_QUERY. The students' fetch cell sends the same query.
- nea_toi_<date>.csv: the TESS Objects of Interest (TOI) table, as served for TOI_QUERY.
- tess2022244194134-s0056-0000000420814525-0243-s_lc.fits: the 2-minute light curve of HD 209458
  (TESS Input Catalog number 420814525) from the Mikulski Archive for Space Telescopes (MAST), bytes
  unchanged. The notebook reads it with numpy (lab_s07_planetary_checks.read_fits), no astropy.
- tess_s0056_tic420814525_pixels_one_transit.nc: a cut of the target pixel file (the 13 x 11 pixel
  images behind the light curve) around one transit, BTJD 2840.879 +/- 0.3 days, as NetCDF. The full
  target pixel file is 58 MB and is not kept; its fingerprint is recorded.
- exoplanets_hd209458_tess_s56.provenance.json: URLs, retrieval time (UTC), SHA-256 fingerprints, row
  counts, header facts, column definitions and terms of use quoted from the provider pages, citations,
  and the limb-darkening fit used by the synthetic twin.

Regenerate (network needed, about 70 MB of downloads):
    pixi run python book/Chapter2-DataManipulation/snapshots/make_exoplanets_hd209458_tess_s56.py
"""
import datetime as dt
import hashlib
import html
import json
import re
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lab_s07_planetary_checks as ck          # noqa: E402  the same queries and reader the students use
import lab_s07_planetary_synthetic as syn      # noqa: E402  transit model for the limb-darkening fit

NOW = dt.datetime.now(dt.timezone.utc)
DATE = NOW.strftime("%Y-%m-%d")
TPF_NAME = ck.LC_NAME.replace("_lc.fits", "_tp.fits")
TPF_URL = ck.MAST_LC.replace(ck.LC_NAME, TPF_NAME)
FAST_NAME = ck.LC_NAME.replace("-s_lc.fits", "-a_fast-lc.fits")
DOCS = {
    "ps_columns": "https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html",
    "toi_columns": "https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html",
    "pscp_calc": "https://exoplanetarchive.ipac.caltech.edu/docs/pscp_calc.html",
    "criteria": "https://exoplanetarchive.ipac.caltech.edu/docs/exoplanet_criteria.html",
    "acknowledge": "https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html",
    "mast_data_use": "https://archive.stsci.edu/publishing/data-use",
    "mast_ack": "https://archive.stsci.edu/publishing/mission-acknowledgements",
}
SDPDD = "https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014-Rev-F.pdf"
HANDBOOK = ("https://archive.stsci.edu/files/live/sites/mast/files/home/missions-and-data/active-missions/"
            "tess/_documents/TESS_Instrument_Handbook_v0.1.pdf")
# Quoted from the PDFs above (read as text on the retrieval date; their SHA-256 is recorded below).
QUALITY_BITS = {  # SDPDD Rev F, section 9, table 32 "Data quality bits"
    1: "Attitude Tweak", 2: "Safe Mode", 4: "Spacecraft is in Coarse Point", 8: "Spacecraft is in Earth Point",
    16: "Argabrightening event", 32: "Reaction Wheel desaturation Event",
    64: "Cosmic Ray in Optimal Aperture pixel",
    128: "Manual Exclude. The cadence was excluded because of an anomaly.",
    256: "Discontinuity corrected between this cadence and the following one.",
    512: "Impulsive outlier removed before cotrending.",
    1024: "Cosmic ray detected on collateral pixel row or column.",
    2048: "Straylight from Earth or Moon in camera FOV (predicted).",
    4096: "Scattered Light Exclude (spoc-4.0.5 and later).", 8192: "Planet Search Exclude (spoc-4.0.5 and later).",
    16384: "Bad Calibration Exclude (spoc-4.0.14 and later).",
    32768: "Insufficient Targets for Error Correction Exclude (spoc-4.0.14 and later).",
}
APERTURE_BITS = {  # SDPDD Rev F, section 6, table 15 "Aperture mask image bits"
    1: "Pixel was collected by the spacecraft.", 2: "Pixel was in optimal aperture.",
    4: "Pixel was used in background calculation.", 8: "Pixel was used to calculate the flux weighted centroid.",
    16: "Pixel was used to calculate the PRF centroid.", 32: "Pixel is on CCD output A",
    64: "Pixel is on CCD output B", 128: "Pixel is on CCD output C", 256: "Pixel is on CCD output D",
}
LC_COLUMNS = {  # SDPDD Rev F, table 13 "Light curve binary table column summary"
    "TIME": "BJD - 2457000 (BTJD)", "TIMECORR": "light arrival time correction applied",
    "CADENCENO": "timestamp count from start of mission",
    "SAP_FLUX": "Simple aperture photometry light curve.",
    "SAP_BKG": "Estimated background flux contribution to the target aperture. Already subtracted from SAP_FLUX.",
    "PDCSAP_FLUX": "PDC corrected SAP light curve.", "QUALITY": "Each bit is a flag defined in table 32.",
    "MOM_CENTR1": "CCD column position of target's flux-weighted centroid.",
    "MOM_CENTR2": "CCD row position of target's flux-weighted centroid.",
}
SDPDD_QUOTES = [
    "Subtracting TIMECORR from TIME will give the light arrival time at the spacecraft rather than at the "
    "solar system barycenter.",
    "TPS and DV operate on light curves generated by Presearch Data Conditioning (PDC).",
    "RAW_CNTS ... ADU ... Raw, uncalibrated target image. FLUX ... e-/s ... Calibrated, cosmic ray removed, "
    "background subtracted target image.",
]
HANDBOOK_QUOTES = [
    "During the image stacking process, pixel values are examined in groups of N : the highest and lowest values "
    "are discarded, and the sum is used in creating the stack. [...] the cost is a reduction in the effective "
    "exposure time by (N-2)/N [...] Pre-launch simulations calculated that the optimal value for N was 10.",
    "During a momentum dump, the angular momentum is removed from the reaction wheels by firing the thrusters while "
    "the ACS keeps the spacecraft pointing stable. [...] During those five minutes, and for ~5-10 minutes "
    "thereafter, the spacecraft stability will be reduced.",
    "At perigee, the spacecraft reorientation required for data downlink changes the camera temperatures by "
    "1-2 deg C: camera temperatures return to nominal within 1.5-2 days after return to science orientation.",
    "Each camera is equipped with a lens hood to minimize the levels of stray light from the Earth or Moon "
    "striking the detectors.",
    "Pixel Size 15 um square; Lens Custom 146 mm, f/1.4; Bandpass 600-1040 nm; Camera FOV 24 x 24 deg "
    "(Table 2.1, Instrument Characteristics).",
]


def get(url, params=None, timeout=600):
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "mlgeo-book snapshot builder"})
    r.raise_for_status()
    return r


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def page_text(url):
    t = get(url).text
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " | ", t))
    return re.sub(r"(\|\s*)+", "| ", re.sub(r"\s+", " ", t))


def column_doc(text, col):
    if col.endswith("lim"):
        ok = re.search(r"\| " + re.escape(col) + r" \|", text) is not None
        return {"label": "listed in the 'Limit Column' of the definitions table, next to " + col[:-3],
                "description": "the page gives no definition of the values" if ok else "not found on the page"}
    if col == "tfopwg_disp":
        m = re.search(r"\| tfopwg_disp†? \| ([^|]+)\| (.+?=planetary candidate)", text)
        return {"label": m.group(1).strip(), "description": re.sub(r"\s*\|\s*", " ", m.group(2)).strip()}
    m = re.search(r"\| " + re.escape(col) + r"†? \| ([^|]+)\| ([^|]+)\|", text)
    return {"label": m.group(1).strip(), "description": m.group(2).strip()} if m else None


def quote(text, start, end):
    text = re.sub(r"\s+", " ", re.sub(r"\s*\|\s*", " ", text))
    i = text.index(start)
    j = text.index(end, i) + len(end)
    return re.sub(r"\s*\|\s*", " ", text[i:j]).strip()


files, served = {}, {}

# 1. The two archive tables, as served.
for kind, query, name in [("planets", ck.PS_QUERY, f"nea_pscomppars_{DATE}.csv"),
                          ("toi", ck.TOI_QUERY, f"nea_toi_{DATE}.csv")]:
    r = get(ck.TAP, {"query": query, "format": "csv"})
    (HERE / name).write_bytes(r.content)
    n_rows = r.content.count(b"\n") - 1
    files[name] = {"sha256": sha256(r.content), "bytes": len(r.content), "rows": n_rows, "url": r.url,
                   "query": query, "what": "archive TAP result, bytes unchanged"}

# 2. The light curve, as served (MAST redirects to its Amazon S3 copy, whose ETag is the file's MD5).
r = get(ck.MAST_LC)
(HERE / ck.LC_NAME).write_bytes(r.content)
files[ck.LC_NAME] = {"sha256": sha256(r.content), "md5": hashlib.md5(r.content).hexdigest(),
                     "bytes": len(r.content), "url": ck.MAST_LC, "served_from": r.url,
                     "etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified"),
                     "what": "SPOC 2-minute light curve FITS file, bytes unchanged"}
hdus = ck.read_fits(HERE / ck.LC_NAME)
h0, h1 = hdus[0]["header"], hdus[1]["header"]
keys0 = ["OBJECT", "TICID", "SECTOR", "CAMERA", "CCD", "RA_OBJ", "DEC_OBJ", "TESSMAG", "TEFF", "RADIUS",
         "DATA_REL", "PROCVER", "FILEVER", "DATE", "CRMITEN", "CRBLKSZ", "CRSPOC"]
keys1 = ["TIMEREF", "TASSIGN", "TIMESYS", "BJDREFI", "TSTART", "TSTOP", "DATE-OBS", "DATE-END", "TIMEDEL",
         "INT_TIME", "READTIME", "FRAMETIM", "NUM_FRM", "DEADC", "EXPOSURE", "TELAPSE", "CDPP0_5", "CDPP1_0",
         "CDPP2_0", "CROWDSAP", "FLFRCSAP", "PDCMETHD", "GAINA", "READNOIA", "BACKAPP"]
header_facts = {"primary": {k: h0.get(k) for k in keys0},
                "lightcurve": {k: [h1.get(k), hdus[1]["comments"].get(k)] for k in keys1},
                "aperture_npix_optimal": hdus[2]["header"].get("NPIXSAP")}
fast = requests.head(ck.MAST_LC.replace(ck.LC_NAME, FAST_NAME), allow_redirects=True, timeout=120)
other_cadences = {FAST_NAME: {"status": fast.status_code, "bytes": fast.headers.get("Content-Length"),
                              "note": "20-second light curve of the same sector; not used"}}

# 3. The target pixel file: keep a cut around one transit.
with tempfile.TemporaryDirectory() as tmp:
    r = get(TPF_URL)
    served[TPF_URL] = {"sha256": sha256(r.content), "bytes": len(r.content), "file": TPF_NAME,
                       "last_modified": r.headers.get("Last-Modified")}
    tpf_path = Path(tmp) / TPF_NAME
    tpf_path.write_bytes(r.content)
    tp = ck.read_fits(tpf_path)
pix, th, ap = tp[1]["data"], tp[1]["header"], tp[2]
row = pd.read_csv(HERE / f"nea_pscomppars_{DATE}.csv").set_index("pl_name").loc[ck.TARGET]
planet = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
P, T0 = float(planet["pl_orbper"]), float(planet["pl_tranmid"]) - 2457000.0
t = np.asarray(pix["TIME"], float)
tc = T0 + P * np.round((2840.88 - T0) / P)
win = np.isfinite(t) & (np.abs(t - tc) <= 0.3)
idx = np.flatnonzero(win)
col0, row0 = int(th["1CRV4P"]), int(th["2CRV4P"])
nrow, ncol = pix["FLUX"].shape[1:]
nc_name = "tess_s0056_tic420814525_pixels_one_transit.nc"
with netCDF4.Dataset(HERE / nc_name, "w", format="NETCDF4") as nc:
    nc.title = f"TESS sector 56 target pixels of TIC {ck.TARGET_TIC} (HD 209458) around the transit at BTJD {tc:.4f}"
    nc.source = f"{TPF_NAME} from MAST ({TPF_URL}), cut in time by make_exoplanets_hd209458_tess_s56.py"
    nc.time_window = f"BTJD {tc - 0.3:.4f} to {tc + 0.3:.4f} (transit mid-time from the archive ephemeris)"
    nc.pixel_scale_note = "one TESS pixel is 15 micrometres behind a 146 mm lens: 15e-6/0.146 rad = 21.2 arcsec"
    nc.createDimension("time", len(idx))
    nc.createDimension("row", nrow)
    nc.createDimension("column", ncol)
    v = nc.createVariable("time", "f8", ("time",))
    v.units, v.long_name = "BJD - 2457000, days (BTJD), TDB time system", "mid-time of each 2-minute cadence"
    v[:] = t[idx]
    v = nc.createVariable("cadenceno", "i4", ("time",))
    v.long_name = "timestamp count from start of mission"
    v[:] = np.asarray(pix["CADENCENO"])[idx]
    v = nc.createVariable("quality", "i4", ("time",))
    v.long_name = "bit flags, SDPDD table 32 (see provenance record)"
    v[:] = np.asarray(pix["QUALITY"])[idx]
    v = nc.createVariable("row", "i4", ("row",))
    v.long_name = "CCD row number of the pixel"
    v[:] = row0 + np.arange(nrow)
    v = nc.createVariable("column", "i4", ("column",))
    v.long_name = "CCD column number of the pixel"
    v[:] = col0 + np.arange(ncol)
    for name, src, typ, units, desc in [
        ("raw_cnts", "RAW_CNTS", "i4", "ADU (counts), sum of the 2-minute stack",
         "Raw, uncalibrated target image (SDPDD table 8)"),
        ("flux", "FLUX", "f4", "e-/s", "Calibrated, cosmic ray removed, background subtracted target image"),
        ("flux_bkg", "FLUX_BKG", "f4", "e-/s", "Estimate of the background at every pixel in the target image")]:
        v = nc.createVariable(name, typ, ("time", "row", "column"), zlib=True, complevel=6, shuffle=True)
        v.units, v.long_name = units, desc
        v[:] = np.asarray(pix[src])[idx]
    v = nc.createVariable("aperture", "i4", ("row", "column"))
    v.long_name = "aperture mask bits, SDPDD table 15: 2 = pixel in the optimal (photometry) aperture"
    v[:] = np.asarray(ap["data"])
files[nc_name] = {"sha256": sha256((HERE / nc_name).read_bytes()), "bytes": (HERE / nc_name).stat().st_size,
                  "what": f"{len(idx)} cadences x {nrow} rows x {ncol} columns cut from {TPF_NAME}",
                  "from": TPF_NAME}

# 4. Limb darkening for the synthetic twin, fitted to the real PDCSAP light curve with the archive geometry.
lc = hdus[1]["data"]
tt, ff = np.asarray(lc["TIME"], float), np.asarray(lc["PDCSAP_FLUX"], float)
ld = syn.fit_limb_darkening(tt, ff, T0, P, float(planet["pl_ratror"]), float(planet["pl_ratdor"]),
                            float(planet["pl_orbincl"]))

# 5. Documents: units, flags and terms, quoted.
ps_doc, toi_doc = page_text(DOCS["ps_columns"]), page_text(DOCS["toi_columns"])
calc, crit = page_text(DOCS["pscp_calc"]), page_text(DOCS["criteria"])
ack, mast_use, mast_ack = page_text(DOCS["acknowledge"]), page_text(DOCS["mast_data_use"]), page_text(DOCS["mast_ack"])
column_definitions = {
    "pscomppars": {c: column_doc(ps_doc, c) for c in
                   ["pl_name", "discoverymethod", "disc_facility", "tran_flag", "pl_controv_flag", "pl_orbper",
                    "pl_rade", "pl_radelim", "pl_bmasse", "pl_bmasselim", "pl_bmassprov", "pl_ratror", "pl_ratdor",
                    "pl_orbincl", "pl_trandep", "pl_tranmid", "st_teff", "st_rad", "sy_dist"]},
    "toi": {c: column_doc(toi_doc, c) for c in
            ["toi", "tfopwg_disp", "pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade",
             "st_tmag", "st_rad", "toi_created"]},
}
quotes = {
    "composite_table": quote(calc, "By contrast, the goal of the PSCompPars table",
                             "may not be internally or physically self-consistent."),
    "radius_from_mass": quote(calc, "For planets that do have an empirically determined value for the planet mass",
                              "to derive the planet radius"),
    "calculated_reference": quote(calc, "We do not report uncertainties", "\"Calculated Value,\""),
    "planet_criteria": quote(crit, "The mass (or minimum mass)", "30 Jupiter masses."),
    "archive_acknowledgement": quote(ack, "This research has made use of the NASA Exoplanet Archive",
                                     "Exoplanet Exploration Program."),
    "mast_data_use": quote(mast_use, "Most data hosted at MAST are in the public domain",
                           "do not have restrictions on use"),
    "tess_acknowledgement": quote(mast_ack, "This paper includes data collected with the TESS mission",
                                  "NASA Explorer Program."),
}
for url in (SDPDD, HANDBOOK):
    b = get(url).content
    served[url] = {"sha256": sha256(b), "bytes": len(b)}

prov = {
    "description": "HD 209458 b: NASA Exoplanet Archive composite planet table and TOI table (TAP results as "
                   "served), the TESS sector 56 2-minute light curve (FITS as served) and a cut of its target "
                   "pixel file around one transit",
    "retrieved_utc": NOW.strftime("%Y-%m-%d %H:%M"),
    "files": files,
    "served": served,
    "target": {"planet": ck.TARGET, "tic": ck.TARGET_TIC, "archive_row": planet},
    "lightcurve_header": header_facts,
    "other_cadences_at_mast": other_cadences,
    "quality_bits_sdpdd_table32": QUALITY_BITS,
    "aperture_bits_sdpdd_table15": APERTURE_BITS,
    "lightcurve_columns_sdpdd_table13": LC_COLUMNS,
    "sdpdd": {"url": SDPDD, "title": "TESS Science Data Products Description Document, EXP-TESS-ARC-ICD-0014 "
              "Rev F (Twicken et al. 2020, NASA/TM-20205008729)", "quotes": SDPDD_QUOTES},
    "instrument_handbook": {"url": HANDBOOK, "title": "TESS Instrument Handbook v0.1 (Vanderspek, Doty, "
                            "Fausnaugh et al., draft of 6 December 2018)", "quotes": HANDBOOK_QUOTES},
    "column_definitions": column_definitions,
    "quotes": quotes,
    "revisions": "The archive tables change: the archive adds planets and revises values as papers appear, so "
                 "a later download of the same query normally differs. The light curve is one release of the "
                 "Science Processing Operations Center (SPOC) pipeline (DATA_REL and PROCVER in the header); "
                 "MAST serves the same bytes until the sector is reprocessed.",
    "license": {
        "nasa_exoplanet_archive": "No license is stated on the archive pages checked. The archive asks for this "
                                  "acknowledgement: \"" + quotes["archive_acknowledgement"] + "\" (" + DOCS["acknowledge"] + ")",
        "mast_tess": "\"" + quotes["mast_data_use"] + "\" (" + DOCS["mast_data_use"] + "). Requested acknowledgement: \""
                     + quotes["tess_acknowledgement"] + "\" (" + DOCS["mast_ack"] + ")",
    },
    "citations": [
        "NASA Exoplanet Science Institute (2026). Planetary Systems Composite Parameters table, NASA Exoplanet "
        f"Archive, IPAC, accessed {DATE}. doi:10.26133/NEA13",
        "Christiansen, J. L., et al. (2025). The NASA Exoplanet Archive and Exoplanet Follow-up Observing Program: "
        "Data, Tools, and Usage. Planetary Science Journal 6. doi:10.3847/PSJ/ade3c2",
        "Guerrero, N. M., et al. (2021). The TESS Objects of Interest Catalog from the TESS Prime Mission. ApJS 254, "
        f"39. doi:10.3847/1538-4365/abefe1 (TOI table accessed {DATE} through the NASA Exoplanet Archive)",
        "TESS Team (2021). TESS Light Curves - All Sectors. MAST, STScI. doi:10.17909/t9-nmc8-f686",
        "Ricker, G. R., et al. (2015). Transiting Exoplanet Survey Satellite. JATIS 1, 014003. "
        "doi:10.1117/1.JATIS.1.1.014003",
        "Jenkins, J. M., et al. (2016). The TESS science processing operations center. Proc. SPIE 9913, 99133E. "
        "doi:10.1117/12.2233418",
    ],
    "synthetic_twin_inputs": {
        "limb_darkening_fit": ld,
        "how": "quadratic limb-darkening coefficients u1, u2 (and a flux scale) fitted by least squares to the "
               "real PDCSAP flux within 0.2 d of each transit, with the radius ratio, a/R* and inclination fixed "
               "to the archive row of HD 209458 b; transit model lab_s07_planetary_synthetic.transit_model",
    },
}
(HERE / "exoplanets_hd209458_tess_s56.provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
print(json.dumps({k: prov[k] for k in ("retrieved_utc", "files", "synthetic_twin_inputs")}, indent=1))
print("column definitions found:", sum(v is not None for d in column_definitions.values() for v in d.values()),
      "of", sum(len(d) for d in column_definitions.values()))
