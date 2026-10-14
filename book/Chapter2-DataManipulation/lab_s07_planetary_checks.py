"""Session 7, planetary science track: reading the real files, the provider queries, and the self-checks.

Used by lab_s07_planetary.ipynb, next to lab_s07_planetary_synthetic.py (the synthetic twin). Three groups:

1. Reading the real data without astropy: ``read_fits`` reads the FITS files that NASA's TESS mission
   delivers (headers are plain text, tables are big-endian binary), ``read_lightcurve``, ``read_pixels``,
   ``read_planets`` and ``read_toi``.
2. Pulling the data yourself: ``fetch_all`` asks the NASA Exoplanet Archive and the Mikulski Archive for
   Space Telescopes (MAST) for the same files as the course copies, and ``file_facts`` gives the size,
   counts, first and last records and SHA-256 fingerprint of each copy.
3. Self-checks (``check``): they print "correct" or a hint; they never print an answer.

Nothing here downloads when the environment variable CI is set or MLGEO_OFFLINE=1.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SNAP = HERE / "snapshots"
PROV_FILE = SNAP / "exoplanets_hd209458_tess_s56.provenance.json"
LC_NAME = "tess2022244194134-s0056-0000000420814525-0243-s_lc.fits"
LC_FILE = SNAP / LC_NAME
PIX_FILE = SNAP / "tess_s0056_tic420814525_pixels_one_transit.nc"
CLASS_SEED_NAME = "ess469-autumn-2026-98"   # chosen so the class twin holds problems of 7 kinds
NOT_ANSWERED = "not answered yet"

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
PS_QUERY = ("select pl_name,hostname,ra,dec,sy_dist,discoverymethod,disc_year,disc_facility,tran_flag,"
            "pl_controv_flag,pl_orbper,pl_rade,pl_radeerr1,pl_radelim,"
            "pl_bmasse,pl_bmasseerr1,pl_bmasselim,pl_bmassprov,"
            "pl_ratror,pl_ratdor,pl_orbincl,pl_trandep,pl_tranmid,st_teff,st_rad "
            "from pscomppars order by pl_name")
TOI_QUERY = ("select toi,tid,tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep,pl_rade,"
             "st_tmag,st_rad,toi_created from toi order by toi")
MAST_LC = f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/{LC_NAME}"

TARGET = "HD 209458 b"          # the planet of this track
TARGET_TIC = 420814525          # its star in the TESS Input Catalog


# --------------------------------------------------------------------------------------------------
# 1. Reading the real data
# --------------------------------------------------------------------------------------------------
def offline():
    """True on the book build (CI set) or when MLGEO_OFFLINE=1: no download is attempted."""
    return bool(os.environ.get("CI")) or os.environ.get("MLGEO_OFFLINE") == "1"


def output_dir():
    """Where your own files go: s07_my_files/ beside the notebook, or a temporary folder on the book build."""
    d = Path(tempfile.gettempdir()) / "mlgeo_s07_planetary" if offline() else HERE / "s07_my_files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def provenance():
    return json.loads(PROV_FILE.read_text())


def course_copy(kind):
    """Path of the course copy: kind is 'planets', 'toi' or 'lightcurve'."""
    files = provenance()["files"]
    want = {"planets": "nea_pscomppars_", "toi": "nea_toi_", "lightcurve": "tess2022244194134"}[kind]
    return SNAP / next(n for n in files if n.startswith(want))


def seed_from_netid(netid):
    """An integer seed from a UW NetID (lower case, spaces removed); the class seed when netid is empty."""
    name = (netid or "").strip().lower() or CLASS_SEED_NAME
    return int(hashlib.sha256(name.encode()).hexdigest()[:12], 16)


def fingerprint(path, algorithm="sha256"):
    """Fingerprint of the file's bytes (SHA-256 by default, as 64 hexadecimal characters)."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_FITS_CODES = {"L": "S1", "B": "u1", "I": ">i2", "J": ">i4", "K": ">i8", "E": ">f4", "D": ">f8"}


def _parse_card(card):
    key = card[:8].strip()
    if card[8:10] != "= ":
        return key, None, card[8:].strip()
    rest = card[10:]
    if rest.lstrip().startswith("'"):
        s = rest.lstrip()[1:]
        out, i = [], 0
        while i < len(s):
            if s[i] == "'":
                if s[i + 1:i + 2] == "'":
                    out.append("'")
                    i += 2
                    continue
                break
            out.append(s[i])
            i += 1
        comment = s[i + 1:].split("/", 1)[1].strip() if "/" in s[i + 1:] else ""
        return key, "".join(out).rstrip(), comment
    val, _, comment = rest.partition("/")
    val = val.strip()
    if val in ("T", "F"):
        v = val == "T"
    else:
        try:
            v = int(val)
        except ValueError:
            try:
                v = float(val.replace("D", "E"))
            except ValueError:
                v = val
    return key, v, comment.strip()


def read_fits(path):
    """Read every header-data unit (HDU) of a FITS file with numpy only.

    Returns a list of dicts with 'header' ({keyword: value}), 'comments' ({keyword: comment}) and 'data'
    (a numpy structured array for a binary table, an n-d array for an image, None when empty). Covers the
    column types TESS light curve and target pixel files use (L, B, I, J, K, E, D, A, with repeat counts).
    """
    raw = Path(path).read_bytes()
    pos, hdus = 0, []
    while pos + 2880 <= len(raw):
        header, comments, end = {}, {}, False
        while not end:
            block = raw[pos:pos + 2880].decode("ascii")
            pos += 2880
            for i in range(36):
                card = block[80 * i:80 * (i + 1)]
                if card.startswith("END") and card[3:].strip() == "":
                    end = True
                    break
                key, val, com = _parse_card(card)
                if val is not None and key not in header:
                    header[key], comments[key] = val, com
        naxis = [header.get(f"NAXIS{i}", 0) for i in range(1, header.get("NAXIS", 0) + 1)]
        nbytes = abs(header.get("BITPIX", 8)) // 8 * header.get("GCOUNT", 1) * (
            header.get("PCOUNT", 0) + (int(np.prod(naxis)) if naxis else 0))
        data = None
        if header.get("XTENSION") == "BINTABLE" and naxis[1] > 0:
            names, formats, shapes = [], [], []
            for k in range(1, header["TFIELDS"] + 1):
                m = re.match(r"(\d*)([A-Z])", header[f"TFORM{k}"])
                rep, code = int(m.group(1) or 1), m.group(2)
                names.append(header[f"TTYPE{k}"])
                if code == "A":
                    formats.append(f"S{rep}")
                    shapes.append(())
                    continue
                dim = header.get(f"TDIM{k}")
                shape = tuple(int(x) for x in dim.strip("()").split(","))[::-1] if dim else ((rep,) if rep > 1 else ())
                formats.append((_FITS_CODES[code], shape) if shape else _FITS_CODES[code])
            dtype = np.dtype({"names": names, "formats": formats, "itemsize": naxis[0]})
            data = np.frombuffer(raw, dtype=dtype, count=naxis[1], offset=pos)
        elif naxis and header.get("XTENSION", "IMAGE") == "IMAGE" and nbytes:
            code = {8: "u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}[header["BITPIX"]]
            data = np.frombuffer(raw, dtype=code, count=int(np.prod(naxis)), offset=pos).reshape(naxis[::-1])
        hdus.append({"header": header, "comments": comments, "data": data})
        pos += -(-nbytes // 2880) * 2880
    return hdus


def _native(a):
    a = np.asarray(a)
    return a.astype(a.dtype.newbyteorder("=")) if a.dtype.byteorder not in ("=", "|") else a


def read_lightcurve(path=None):
    """The TESS light curve as a table (one row per 2-minute cadence, the file's own column names and units).

    Returns (table, headers) where headers holds the primary header, the light curve header and the
    aperture image, as stored in the file.
    """
    hdus = read_fits(path or course_copy("lightcurve"))
    tab = hdus[1]["data"]
    lc = pd.DataFrame({n: _native(tab[n]) for n in tab.dtype.names})
    heads = {"primary": hdus[0]["header"], "lightcurve": hdus[1]["header"],
             "lightcurve_comments": hdus[1]["comments"], "aperture": _native(hdus[2]["data"])}
    return lc, heads


def read_pixels(path=PIX_FILE):
    """The pixel cutout around one transit: an xarray Dataset with dimensions time, row and column."""
    import xarray as xr
    return xr.open_dataset(path, engine="netcdf4").load()


def read_planets(path=None):
    """The Planetary Systems Composite Parameters table: one row per confirmed planet."""
    return pd.read_csv(path or course_copy("planets"))


def read_toi(path=None):
    """The TESS Objects of Interest table: one row per transit-like signal flagged by the TESS project."""
    return pd.read_csv(path or course_copy("toi"))


def sky_map(ax, planets, highlight=TARGET):
    """Where the known planets are on the sky (right ascension against declination), by discovery method."""
    groups = [("Transit", "#1f77b4"), ("Radial Velocity", "#d62728"), ("Microlensing", "#2ca02c"),
              ("Imaging", "#9467bd")]
    other = ~planets["discoverymethod"].isin([g for g, _ in groups])
    for name, color in groups:
        s = planets[planets["discoverymethod"] == name]
        ax.scatter(s["ra"], s["dec"], s=4, color=color, alpha=0.6, lw=0, label=f"{name} ({len(s)})")
    s = planets[other]
    ax.scatter(s["ra"], s["dec"], s=4, color="0.4", alpha=0.6, lw=0, label=f"other methods ({len(s)})")
    t = planets[planets["pl_name"] == highlight].iloc[0]
    ax.scatter(t["ra"], t["dec"], s=160, marker="*", color="gold", edgecolor="k", zorder=5,
               label=f"{t['hostname']} ({t['sy_dist']:.1f} pc)")
    ax.set_xlim(360, 0)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(0, 361, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("right ascension (degrees, increasing to the east, i.e. to the left)")
    ax.set_ylabel("declination (degrees)")
    ax.grid(alpha=0.3)
    leg = ax.legend(loc="lower left", fontsize=8, markerscale=2, framealpha=0.9)
    leg.legend_handles[-1].set_sizes([80])


# --------------------------------------------------------------------------------------------------
# 2. Pulling the data yourself
# --------------------------------------------------------------------------------------------------
def _get(url, params=None, timeout=60):
    import requests
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "mlgeo-book ESS469 lab"})
    r.raise_for_status()
    return r


def _with_timeout(fn, timeout):
    pool = cf.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        return fut.result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def fetch_all(timeout=90):
    """Ask the providers for the three files of this track; fall back to the course copies.

    Returns {kind: {"path": file to use, "source": "provider" or "course copy", "note": text, ...}}.
    Never downloads when CI is set or MLGEO_OFFLINE=1.
    """
    out, today = {}, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    jobs = {
        "planets": (TAP, {"query": PS_QUERY, "format": "csv"}, f"nea_pscomppars_{today}.csv"),
        "toi": (TAP, {"query": TOI_QUERY, "format": "csv"}, f"nea_toi_{today}.csv"),
        "lightcurve": (MAST_LC, None, LC_NAME),
    }
    for kind, (url, params, name) in jobs.items():
        rec = {"path": course_copy(kind), "source": "course copy"}
        if offline():
            rec["note"] = "no download (offline mode): the course copy is used"
        else:
            try:
                r = _with_timeout(lambda: _get(url, params, timeout), timeout + 5)
                p = output_dir() / name
                p.write_bytes(r.content)
                rec = {"path": p, "source": "provider", "url": r.url,
                       "note": f"downloaded {len(r.content):,} bytes from {url.split('/')[2]}",
                       "etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified")}
            except Exception as e:  # noqa: BLE001  any failure falls back to the course copy
                rec["note"] = f"download failed ({type(e).__name__}: {str(e)[:120]}); using the course copy"
        out[kind] = rec
    return out


def file_facts(path, kind):
    """Size, counts, first and last records and SHA-256 fingerprint of one copy of a file."""
    path = Path(path)
    facts = {"file": path.name, "bytes": path.stat().st_size}
    if kind == "lightcurve":
        lc, h = read_lightcurve(path)
        facts.update({"rows": len(lc), "first_cadence": int(lc["CADENCENO"].iloc[0]),
                      "last_cadence": int(lc["CADENCENO"].iloc[-1]),
                      "first_time_btjd": float(lc["TIME"].dropna().iloc[0]),
                      "last_time_btjd": float(lc["TIME"].dropna().iloc[-1]),
                      "file_made_on (DATE)": h["primary"].get("DATE"),
                      "pipeline (PROCVER)": h["primary"].get("PROCVER")})
    else:
        t = pd.read_csv(path)
        key = "pl_name" if kind == "planets" else "toi"
        facts.update({"rows": len(t), "columns": t.shape[1], "first_record": str(t[key].iloc[0]),
                      "last_record": str(t[key].iloc[-1])})
    facts["sha256"] = fingerprint(path)
    return facts


def compare_tables(course_path, your_path, key, cols):
    """What changed between two downloads of a table: rows added, rows removed, values that differ."""
    a = pd.read_csv(course_path).set_index(key)
    b = pd.read_csv(your_path).set_index(key)
    added, removed = b.index.difference(a.index), a.index.difference(b.index)
    common = a.index.intersection(b.index)
    changed = {}
    for c in cols:
        x, y = a.loc[common, c], b.loc[common, c]
        diff = ~((x == y) | (x.isna() & y.isna()))
        changed[c] = int(diff.sum())
    return {"rows_added": len(added), "rows_removed": len(removed), "examples_added": list(added[:5]),
            "values_changed_in_common_rows": changed}


# --------------------------------------------------------------------------------------------------
# Physical helpers used by the self-checks and the worked solutions
# --------------------------------------------------------------------------------------------------
EARTH_DENSITY = 5.514      # g/cm3, mean density of the Earth (NASA Earth fact sheet)
R_SUN_IN_R_EARTH = 695700.0 / 6378.1   # IAU 2015 nominal solar radius / nominal equatorial Earth radius
R_JUP_IN_R_EARTH = 71492.0 / 6378.1    # IAU 2015 nominal equatorial Jupiter radius / Earth radius


def bulk_density(mass_earth, radius_earth):
    """Bulk density in g/cm3 from mass and radius in Earth units (Earth: 5.514 g/cm3)."""
    return EARTH_DENSITY * np.asarray(mass_earth, float) / np.asarray(radius_earth, float) ** 3


def chen_kipping_radius(mass_earth):
    """Radius (Earth radii) the archive computes from a mass with Chen & Kipping (2017), as given on
    https://exoplanetarchive.ipac.caltech.edu/docs/pscp_calc.html."""
    m = np.asarray(mass_earth, float)
    lm = np.log10(m)
    c = np.select([m < 2.04, m < 132, m < 26600], [0.00346, -0.0925, 1.25], -2.85)
    s = np.select([m < 2.04, m < 132, m < 26600], [0.2790, 0.589, -0.044], 0.881)
    return 10 ** (c + s * lm)


def chen_kipping_mass(radius_earth):
    """Mass (Earth masses) the archive computes from a radius; NaN in the 11.1-14.3 Earth-radius range."""
    r = np.asarray(radius_earth, float)
    lr = np.log10(r)
    c = np.select([r < 1.23, r < 11.1, r < 14.3], [0.00346, -0.0925, np.nan], -2.85)
    s = np.select([r < 1.23, r < 11.1, r < 14.3], [0.2790, 0.589, np.nan], 0.881)
    return 10 ** ((lr - c) / s)


def measured_mass_and_radius(planets):
    """Rows where both mass and radius are measurements of this planet: it transits (so the radius is
    measured), the mass provenance is 'Mass', and neither value is a limit."""
    p = planets
    return p[(p["tran_flag"] == 1) & (p["pl_bmassprov"] == "Mass") & (p["pl_bmasselim"] == 0)
             & (p["pl_radelim"] == 0) & p["pl_bmasse"].notna() & p["pl_rade"].notna()]


def phase_days(time, period, t0):
    """Time from the nearest transit mid-time, in days (negative before, positive after)."""
    return ((np.asarray(time) - t0 + 0.5 * period) % period) - 0.5 * period


# --------------------------------------------------------------------------------------------------
# 3. Self-checks
# --------------------------------------------------------------------------------------------------
_LAST = {}


def remember(**kw):
    """Store facts computed in the notebook (for example whether your download matched) for the checks."""
    _LAST.update(kw)


def _ok(msg="correct"):
    print(msg)
    return True


def _hint(msg):
    print("not yet. Hint: " + msg)
    return False


def _words(s):
    return len(str(s or "").split())


def _has(s, *words):
    s = str(s or "").lower()
    return any(w in s for w in words)


def _num(a):
    return float(a)


def _close(a, b, rel=0.05, abs_=0.0):
    return abs(_num(a) - b) <= max(rel * abs(b), abs_)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _text_ok(a, min_words, hint, *need):
    if _words(a) < min_words:
        return _hint(f"write at least {min_words} words. {hint[:1].upper() + hint[1:]}")
    if need and not any(_has(a, *n) if isinstance(n, tuple) else _has(a, n) for n in need):
        return _hint(hint)
    return _ok("thanks: compare your words with the worked solution below once you are done")


def check(qid, answer):
    """Self-check for question `qid`. Prints 'correct' or a hint; never prints the answer."""
    if answer is None or (isinstance(answer, (str, list, dict, tuple)) and len(answer) == 0):
        print(NOT_ANSWERED)
        return None
    if isinstance(answer, dict) and all(v is None or v == "" for v in answer.values()):
        print(NOT_ANSWERED)
        return None
    try:
        return _CHECKS[qid](answer)
    except KeyError as e:
        return _hint(f"an entry is missing ({e}). Keep the keys of the template.")
    except Exception as e:  # noqa: BLE001  a malformed answer gets a hint, not a traceback
        return _hint(f"the answer has an unexpected form ({type(e).__name__}: {e}). Look at the template.")


def _c0(a):
    if _has(a, "telescope", "survey", "kepler", "where we looked", "pointed"):
        return _ok()
    if _has(a, "galaxy", "planet", "star"):
        return _hint("colour the points by disc_facility.")
    return _hint("answer 'telescope' or 'galaxy', with a few words of reason.")


def _c1_same(a):
    same_lc, same_tab = _LAST.get("same_lightcurve"), _LAST.get("same_planets")
    if same_lc is None:
        return _ok("no download was made on this computer (offline), so there is nothing to compare; "
                   "read the two worked cases in the solution below")
    ok = a.get("lightcurve_identical") == same_lc and a.get("planets_identical") == same_tab
    return _ok() if ok else _hint("compare the two 'sha256' lines for each file: identical bytes give "
                                  "identical fingerprints, and one changed byte changes the whole fingerprint.")


def _c1_change(a):
    if a.get("size") is False and a.get("rows") is False and a.get("fingerprint") is True:
        return _ok("correct: a revised number of the same width keeps the size and the row count, only the "
                   "fingerprint catches it")
    if a.get("fingerprint") is not True:
        return _hint("any change of any byte changes the SHA-256 fingerprint. Rerun the one-byte demonstration.")
    return _hint("the revision replaces '1.190000' by '1.230000': same number of characters, same rows.")


def _c1_exposure(a):
    _, h = read_lightcurve()
    lh = h["lightcurve"]
    expect = lh["TIMEDEL"] * 86400 * lh["DEADC"]
    if _close(a, expect, rel=0.01):
        return _ok()
    if _close(a, lh["TIMEDEL"] * 86400, rel=0.01):
        return _hint("120 s is the time between two values, not the time spent collecting light. Read the "
                     "comment of DEADC in the header and what the cosmic-ray step throws away.")
    if _close(a, lh["NUM_FRM"] * lh["INT_TIME"], rel=0.01):
        return _hint("you counted all 60 frames. The cosmic-ray step drops 2 of every CRBLKSZ frames.")
    return _hint("two header keys matter: the time between values, and the fraction of it spent collecting "
                 "light. Find the second among the keys printed above.")


def _c1_raw(a):
    ok = str(a.get("least_processed_column", "")).upper().strip() == "SAP_FLUX" and a.get("file_is_raw") is False
    if ok:
        return _ok()
    if str(a.get("least_processed_column", "")).upper().strip() == "RAW_CNTS":
        return _hint("RAW_CNTS is not in this file; it is in the target pixel file of Part 4.")
    if a.get("file_is_raw") is not False:
        return _hint("the CCD recorded 2-second frames in counts. Is anything in this file still in counts, "
                     "summed over one pixel, every 2 seconds?")
    return _hint("PDCSAP_FLUX went through every step SAP_FLUX went through, plus cotrending.")


def _c2_row(a):
    return _ok() if str(a).strip().lower().lstrip("(").startswith("b") else _hint(
        "read the first paragraph of the archive's page on how the composite table is built (quoted above): "
        "can one row mix values from different papers?")


def _c2_units(a):
    rade = str(a.get("pl_rade") or "").lower()
    ok = {"pl_rade": _has(rade, "earth") or _norm(rade) in ("re", "rearth", "rearthradii"),
          "pl_trandep_planets": _has(a.get("pl_trandep_planets"), "%", "percent", "per cent"),
          "pl_trandep_toi": _has(a.get("pl_trandep_toi"), "ppm", "part", "million")}
    tips = {"pl_rade": "pl_rade: read its label in prov['column_definitions']['pscomppars']",
            "pl_trandep_planets": "pl_trandep of the planet table: read its label in "
                                  "prov['column_definitions']['pscomppars']",
            "pl_trandep_toi": "pl_trandep of the TOI table: read its label in prov['column_definitions']['toi']"}
    bad = [k for k, v in ok.items() if not v]
    return _ok() if not bad else _hint("; ".join(tips[k] for k in bad))


def _c2_missing(a):
    p = read_planets()
    exp = {"no_mass": int(p["pl_bmasse"].isna().sum()), "no_radius": int(p["pl_rade"].isna().sum()),
           "mass_from_radius": int((p["pl_bmassprov"] == "M-R relationship").sum()),
           "mass_is_a_limit": int((p["pl_bmasselim"].fillna(0) != 0).sum())}
    bad = [k for k, v in exp.items() if int(a.get(k, -1)) != v]
    if not bad:
        return _ok()
    tips = {"no_mass": "count empty cells: planets['pl_bmasse'].isna().sum()",
            "no_radius": "count empty cells in pl_rade",
            "mass_from_radius": "value_counts() of pl_bmassprov: one category is not in the column definition",
            "mass_is_a_limit": "count rows where pl_bmasselim is not 0 (1 or -1: the number is a bound)"}
    return _hint("; ".join(tips[k] for k in bad))


def _c2_limits(a):
    lo, hi = a.get("pl_rade_low"), a.get("pl_rade_high")
    mhi = a.get("pl_bmasse_high")
    if None in (lo, hi, mhi):
        print(NOT_ANSWERED)
        return None
    off = [name for name, good in [("lower radius bound", 0 < float(lo) < 1),
                                   ("upper radius bound", 20 <= float(hi) <= 40),
                                   ("upper mass bound", 3000 <= float(mhi) <= 12000)] if not good]
    if not off:
        return _ok("reasonable bounds, written before looking")
    return _hint(f"rethink your {', '.join(off)}. Use the scales in the question text and the archive's "
                 "criterion in prov['quotes']['planet_criteria'].")


def _c2_extreme(a):
    e, r = str(a.get("error", "")).strip(), str(a.get("real_extreme", "")).strip()
    if _norm(e) == "kepler37e" and _norm(r) == "kepler51b":
        return _ok()
    if _norm(e) == "kepler51b":
        return _hint("you swapped them. Low density only needs a thick, hot gas envelope; can any material be "
                     "more than a hundred times denser than Earth?")
    return _hint("compute bulk_density(mass, radius) for the two rows and compare with Earth (5.5 g/cm3) and "
                 "with water (1 g/cm3).")


def _c3_interval(a):
    lc, h = read_lightcurve()
    exp = float(np.nanmedian(np.diff(lc["TIME"]))) * 86400
    return _ok() if _close(a, exp, rel=0, abs_=0.5) else _hint(
        "use np.nanmedian(np.diff(lc['TIME'])) and convert days to seconds.")


def _c3_gaps(a):
    lc, _ = read_lightcurve()
    nt = lc["TIME"].isna().to_numpy()
    blocks = int(np.sum(np.diff(np.r_[0, nt.astype(int)]) == 1))
    pdc_only = float(((lc["PDCSAP_FLUX"].isna()) & lc["SAP_FLUX"].notna()).sum() * 2 / 60)
    ok_b = int(a.get("blocks_without_time", -1)) == blocks
    ok_h = _close(a.get("hours_with_sap_but_no_pdcsap", -1), pdc_only, rel=0.1)
    if ok_b and ok_h:
        return _ok()
    msg = []
    if not ok_b:
        msg.append("count runs of consecutive rows where TIME is NaN, not the rows themselves")
    if not ok_h:
        msg.append("hours: count rows, then convert with the cadence")
    return _hint("; ".join(msg))


def _c3_transits(a):
    p = read_planets().set_index("pl_name").loc[TARGET]
    lc, _ = read_lightcurve()
    t = lc["TIME"].dropna()
    t0 = p["pl_tranmid"] - 2457000
    n = np.arange(np.ceil((t.min() - t0) / p["pl_orbper"]), np.floor((t.max() - t0) / p["pl_orbper"]) + 1)
    return _ok() if int(a) == len(n) else _hint(
        "sector length divided by the period gives a first guess; then count the dips in the PDCSAP plot.")


def _c3_sensor(a):
    lc, _ = read_lightcurve()
    tmax = float(lc.loc[lc["SAP_BKG"].idxmax(), "TIME"])
    t_ok = _close(a.get("time_of_max_background_btjd", 0), tmax, rel=0, abs_=0.5)
    c_ok = _has(a.get("cause"), "scatter", "earth", "moon", "stray")
    if t_ok and c_ok:
        return _ok()
    if not t_ok:
        return _hint("plot SAP_BKG against TIME and read the time of its maximum (lc.loc[lc['SAP_BKG'].idxmax()]).")
    return _hint("what bright body near TESS's orbit can shine light into a camera's lens hood? See the quality-flag "
                 "table (bit values 2048 and 4096). Neither flag is set in this file, so the background column "
                 "itself is your evidence.")


def _c4_shape(a):
    px = read_pixels()
    ok = (int(a.get("n_time", -1)) == px.sizes["time"] and int(a.get("n_row", -1)) == px.sizes["row"]
          and int(a.get("n_column", -1)) == px.sizes["column"])
    return _ok() if ok else _hint("print(px.sizes) gives the length of each dimension.")


def _c4_peak(a):
    px = read_pixels()
    img = px["flux"].median("time")
    r, c = np.unravel_index(int(np.nanargmax(img.values)), img.shape)
    ok = int(a.get("row", -1)) == int(px["row"][r]) and int(a.get("column", -1)) == int(px["column"][c])
    return _ok() if ok else _hint("take the median image over time, then np.unravel_index(argmax) and read the "
                                  "CCD row and column coordinates (not the 0-based index).")


def _c4_where(a):
    on_target = _has(a, "target", "hd 209458", "center", "centre", "same", "brightest", "star itself", "the star",
                     "not on a neigh", "not on the neigh", "not a neigh", "not the neigh")
    on_neighbour = _has(a, "neighbo") and not _has(a, "not on a neigh", "not on the neigh", "not a neigh",
                                                   "not the neigh", "no neigh")
    return _ok() if on_target and not on_neighbour else _hint(
        "look where the difference image (out of transit minus in transit) is largest. Does it coincide with "
        "the brightest pixels, or with the fainter sources at the upper right (row 360, column 1872) or lower "
        "left (row 354, column 1862)?")


def _c6_sign(a):
    return _ok() if int(a) == 1 else _hint(
        "think of the full range from rocks to gas giants.")


def _c6_corr(a):
    from scipy import stats
    p = read_planets()
    allr = p[(p["pl_bmasse"] > 0) & (p["pl_rade"] > 0)]
    meas = measured_mass_and_radius(p)
    exp = {"pearson_all": stats.pearsonr(np.log10(allr["pl_bmasse"]), np.log10(allr["pl_rade"]))[0],
           "spearman_all": stats.spearmanr(allr["pl_bmasse"], allr["pl_rade"])[0],
           "pearson_measured": stats.pearsonr(np.log10(meas["pl_bmasse"]), np.log10(meas["pl_rade"]))[0],
           "spearman_measured": stats.spearmanr(meas["pl_bmasse"], meas["pl_rade"])[0]}
    bad = [k for k, v in exp.items() if not _close(a.get(k, 9), v, rel=0, abs_=0.01)]
    if not bad:
        return _ok()
    return _hint(f"check {', '.join(bad)}: Pearson on log10 of both columns, Spearman on the raw values "
                 "(ranks do not care about the log); 'all' = every row with both values > 0, 'measured' = "
                 "ck.measured_mass_and_radius(planets).")


def _c6_trap(a):
    return _text_ok(a, 15, "say which rows sit exactly on a curve because the archive computed one value from "
                    "the other, and what that does to the correlation.", ("calculat", "computed", "chen",
                                                                          "m-r", "relationship", "circular"))


def _c7_task(a):
    s = str(a).strip().lower()
    return _ok() if "classif" in s else _hint(
        "the TFOPWG disposition is a label from a short list (planet, false positive, false alarm, candidate).")


def _c7_classes(a):
    t = read_toi()
    lab = t["tfopwg_disp"].map({"CP": "planet", "KP": "planet", "FP": "false_positive", "FA": "false_alarm"})
    frac = lab.value_counts(normalize=True) * 100
    bad = [k for k in ("planet", "false_positive", "false_alarm") if not _close(a.get(k, -1), frac[k], abs_=0.6)]
    if not bad:
        return _ok()
    return _hint("which rows have a verdict? Use percent.")


def _c7_balance(a):
    ok = a.get("balance_before_split") is False and _has(a.get("metric"), "precision", "recall", "pr", "f1",
                                                          "balanced accuracy")
    if ok:
        return _ok()
    if a.get("balance_before_split") is not False:
        return _hint("if you oversample before splitting, copies of the same false alarm land in both the "
                     "training and the test set. What does that do to the test score?")
    return _hint("with 3.5% false alarms, a model that never predicts one has 96.5% accuracy. Which metrics look "
                 "at the rare class only?")


def _c7_regression(a):
    p = read_planets()
    r = p.loc[(p["tran_flag"] == 1) & (p["pl_radelim"] == 0), "pl_rade"].dropna()
    exp = {"median": float(r.median()), "p90": float(r.quantile(0.9)),
           "pct_above_8": float((r > 8).mean() * 100)}
    got = dict(a)
    if got.get("pct_above_8") is not None and float(got["pct_above_8"]) < 1:   # a fraction instead of percent
        got["pct_above_8"] = float(got["pct_above_8"]) * 100
    bad = [k for k, v in exp.items() if not _close(got.get(k, -1), v, rel=0.03, abs_=0.05)]
    return _ok() if not bad else _hint(f"check {', '.join(bad)}: use transiting planets with a radius value "
                                       "(tran_flag == 1, pl_radelim == 0); the fraction is in percent.")


def _c7_chain(a):
    steps = [str(x) for x in a if str(x).strip()]
    if len(steps) < 4:
        return _hint("write at least four steps, each with its reason.")
    text = " ".join(steps)
    miss = []
    if not _has(text, "split", "star", "time"):
        miss.append("say how you split so that no star appears in both sets")
    if not _has(text, "flag", "keep", "kept"):
        miss.append("say which real extremes you keep, and what you flag rather than delete")
    if miss:
        return _hint("; ".join(miss) + ".")
    return _ok("thanks: compare your chain with the worked solution below")


def _c5_list(a):
    a = [str(x) for x in a if str(x).strip()]
    if len(a) < 4:
        return _hint("list at least four. The quality-flag table (prov['quality_bits_sdpdd_table32']) and the "
                     "instrument handbook quotes (prov['instrument_handbook']['quotes']) name several.")
    kinds = {"time": ("time", "clock", "barycent"), "pointing": ("dump", "pointing", "jitter", "wheel"),
             "light": ("scatter", "stray", "moon", "earth", "background"), "detector": ("cosmic", "spike", "satur",
             "bleed", "glitch", "readout", "outlier", "step", "discontinu"),
             "processing": ("gap", "zero", "duplicate", "stuck", "missing", "unit", "fill"),
             "star": ("flare", "spot", "variab", "neighbo", "blend")}
    hit = [k for k, w in kinds.items() if _has(" ".join(a), *w)]
    if len(hit) < 3:
        return _hint("your list covers few kinds of cause. Think of the clock, the pointing, stray light, the "
                     "detector, the processing, and the star itself.")
    return _ok("good list: keep it next to you while you hunt")


def _c5_reflect(a):
    return _text_ok(a, 30, "name at least one problem that moved the depth and one that cleaning cannot undo, "
                    "with the reason.", ("undo", "cannot", "lost", "unknown", "two values", "choose"))


def _c8_compare(a):
    return _text_ok(a, 20, "name one thing the AI tool missed or invented, and say whether it removed the "
                    "transits or the flare as 'outliers'.", ("miss", "invent", "transit", "flare", "outlier"))


def _c_wrap(a):
    return _text_ok(a, 20, "answer with a number or date from today's work and the physical reason behind it.")


_CHECKS = {
    "0": _c0, "1a": _c1_same, "1b": _c1_change, "1c": _c1_exposure, "1d": _c1_raw,
    "2a": _c2_row, "2b": _c2_units, "2c": _c2_missing, "2d": _c2_limits, "2e": _c2_extreme,
    "3a": _c3_interval, "3b": _c3_gaps, "3c": _c3_transits, "3d": _c3_sensor,
    "5a": _c5_list, "5d": _c5_reflect,
    "4a": _c4_shape, "4b": _c4_peak, "4c": _c4_where,
    "6a": _c6_sign, "6b": _c6_corr, "6c": _c6_trap,
    "7chain": _c7_chain, "7a": _c7_task, "7b": _c7_classes, "7c": _c7_balance, "7d": _c7_regression,
    "8": _c8_compare, "wrap": _c_wrap,
}


# --------------------------------------------------------------------------------------------------
# 4. The summary block for Canvas
# --------------------------------------------------------------------------------------------------
PARTS = [("P0", ["0"]), ("P1", ["1a", "1b", "1c", "1d"]), ("P2", ["2a", "2b", "2c", "2d", "2e"]),
         ("P3", ["3a", "3b", "3c", "3d"]), ("P4", ["4a", "4b", "4c"]), ("P5", ["5a", "5b", "5c", "5d"]),
         ("P6", ["6a", "6b", "6c"]), ("P7", ["7chain", "7a", "7b", "7c", "7d"]), ("P8", ["8tool", "8"])]
_VARS = {"0": "answer_0", "5b": "my_findings", "7chain": "my_chain", "8tool": "ai_findings", "8": "answer_8"}
_NO_CHECK = {"5b", "8tool"}     # findings are counted, not checked


def _quiet(fn, *args, **kw):
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return fn(*args, **kw)
        except Exception:  # noqa: BLE001  a broken answer counts as not passed
            return False


def _given(a):
    if a is None or (isinstance(a, (str, list, dict, tuple)) and len(a) == 0):
        return False
    if isinstance(a, dict) and all(v is None or v == "" for v in a.values()):
        return False
    if isinstance(a, list) and all(not str(x.get("what", "") if isinstance(x, dict) else x).strip() for x in a):
        return False
    return True


def print_summary(ns):
    """Print the Session 7 summary block from the answers in the notebook. Call: ck.print_summary(globals())."""
    import lab_s07_planetary_synthetic as syn
    twin = ns.get("twin")
    answered, passed, n_checks = [], 0, 0
    a5 = syn.assess_cleaning(ns.get("my_clean"), twin, ns.get("my_st_rad")) if twin is not None else None
    for part, qids in PARTS:
        n = 0
        for q in qids:
            if q == "5c":
                ok = a5 is not None and (a5["cleaned_ppm"] is not None or ns.get("my_st_rad") is not None)
                res = a5["passed"] if a5 is not None else None
            else:
                a = ns.get(_VARS.get(q, "answer_" + q))
                ok = _given(a)
                res = _quiet(check, q, a) if ok and q not in _NO_CHECK else None
            n += bool(ok)
            if q not in _NO_CHECK:
                n_checks += 1
                passed += res is True
        answered.append(f"{part} {n}/{len(qids)}")
    wraps = [ns.get(f"wrap_{i}") for i in (1, 2, 3)]
    for w in wraps:
        n_checks += 1
        passed += _given(w) and _quiet(check, "wrap", w) is True
    netid = str(ns.get("MY_NETID") or "").strip() or "not answered"
    if twin is not None:
        found = _quiet(syn.score_findings, ns.get("my_findings"), twin) if _given(ns.get("my_findings")) else None
        planted = f"{found or 0} of {len(twin._planted)} (seed {twin.seed})" if found is not None else \
            f"not answered (seed {twin.seed})"
        fmt = lambda x: "not answered" if x is None else f"{x:.0f}"
        derived = f"{fmt(a5['truth_ppm'])} / {fmt(a5['given_ppm'])} / {fmt(a5['cleaned_ppm'])} ppm"
    else:
        planted, derived = "not answered (run Part 5 first)", "not answered / not answered / not answered ppm"
    text = lambda w: " ".join(str(w).split())[:300] if _given(w) else "not answered"
    print("==== SESSION 7 SUMMARY ====")
    print("track: planetary")
    print(f"netid: {netid}")
    print("parts answered: " + " | ".join(answered))
    print(f"self-checks passed: {passed} of {n_checks}")
    print(f"planted problems found: {planted}")
    print(f"derived quantity (truth / as given / after cleaning): {derived}")
    for i, w in enumerate(wraps, 1):
        print(f"reflection {i}: {text(w)}")
    print("==== END SUMMARY ====")
