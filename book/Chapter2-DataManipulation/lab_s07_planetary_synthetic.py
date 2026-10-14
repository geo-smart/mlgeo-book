"""Session 7, planetary science track: a synthetic twin of a TESS light curve and of a planet table.

The twin copies the structure of the real files in snapshots/ (the sector 56 cadences of HD 209458, the
2-minute sampling, the documented Earth-point gaps and momentum dumps, the flux level and the white-noise
level of the real light curve (not its slower variations), the columns and units of the NASA Exoplanet
Archive table) and puts a planet with known properties in it: the planet "SYN-209458 b", whose radius ratio, orbit and star are copied from the real
row of HD 209458 b. Then it plants problems that real instruments, pipelines and catalogs produce. Which
problems, and where, depends on a seed derived from your UW NetID; an empty NetID gives the class seed used
for the worked solution and on the book build. Every twin holds a second row for its host star and at least one
light-curve problem that moves the depth by 0.4% or more after removing only duplicated cadences and zeros.

Main functions
  make_twin(netid)                  build your twin, write its two files, return a Twin object
  transit_depth(time, flux, ...)    the depth recipe used in Part 5 (mean in transit vs mean around it)
  planet_radius_earth(depth, rstar) planet radius in Earth radii from a depth and a stellar radius
  depth_per_transit(time, flux, g)  the same recipe, one transit at a time
  score_findings(findings, twin)    how many planted problems your findings locate (categories on request)
  check_depth(my_clean, twin, rs)   the Part 5c self-check: depth, what still moves it, the stellar radius
  transit_model(...)                limb-darkened transit light curve (numerical, after Mandel & Agol 2002)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import lab_s07_planetary_checks as ck

NOT_ANSWERED = ck.NOT_ANSWERED

CATEGORY_WORDS = {
    "outliers": "single-cadence outliers: data glitches of the kind SPOC flags as 'impulsive outlier' "
                "(quality bit 512; the real sector 56 file has one)",
    "flux_steps": "unflagged flux discontinuities: sudden steps in sensitivity or pointing, which SPOC (the Science "
                  "Processing Operations Center pipeline) marks with quality bit 256 when it corrects them",
    "scattered_light": "light from Earth or Moon scattered into the camera: the background rises and falls over "
                       "about a day, and the background estimate leaves part of it in the flux",
    "zero_gap": "a gap filled with zeros instead of NaN, with quality 0 (a processing error, not the star)",
    "time_offset": "one stretch with spacecraft time instead of barycentric time (the light-travel correction "
                   "was not applied)",
    "duplicates": "a block of duplicated cadences (the same rows written twice)",
    "forward_fill": "a gap filled by repeating the last good value (forward fill) with quality 0: a processing "
                    "fill, not the star",
    "flare": "a stellar flare (a real outburst of the star, not an instrument error): rare at this size on an old, "
             "quiet star like HD 209458, common on young stars and red dwarfs; in the twin it stands for real "
             "stellar activity that cleaning must not erase",
    "jupiter_radii": "catalog rows whose radius is in Jupiter radii although the column says Earth radii",
    "duplicate_planets": "the same planet listed twice under two spellings, with different values",
    "mislabelled_mass": "masses computed from the radius (Chen & Kipping relation) but labelled 'Mass'",
}
LC_CATS = ["outliers", "flux_steps", "scattered_light", "zero_gap", "time_offset", "duplicates", "forward_fill",
           "flare"]
CAT_CATS = ["jupiter_radii", "duplicate_planets", "mislabelled_mass"]
TARGET_SYN = "SYN-209458 b"


# --------------------------------------------------------------------------------------------------
# Transit physics
# --------------------------------------------------------------------------------------------------
def sky_separation(time, t0, period, a_rs, inc_deg):
    """Projected star-planet distance in stellar radii for a circular orbit; +inf when the planet is behind."""
    ph = 2 * np.pi * (np.asarray(time, float) - t0) / period
    inc = np.deg2rad(inc_deg)
    z = a_rs * np.sqrt(np.sin(ph) ** 2 + (np.cos(inc) * np.cos(ph)) ** 2)
    return np.where(np.cos(ph) > 0, z, np.inf)


def transit_model(time, t0, period, rp_rs, a_rs, inc_deg, u1, u2, n_annuli=400):
    """Relative flux of a star with quadratic limb darkening while a dark disk crosses it.

    The stellar disk is cut into thin rings; for each ring the code computes which fraction of its
    circumference the planet covers, and sums the blocked light. This is the geometry of Mandel & Agol
    (2002, doi:10.1086/345520) done numerically instead of with their closed-form expressions.
    """
    z = sky_separation(time, t0, period, a_rs, inc_deg)
    out = np.ones_like(z, dtype=float)
    hit = z < 1 + rp_rs
    if not hit.any():
        return out
    edges = np.linspace(0, 1, n_annuli + 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    mu = np.sqrt(1 - r ** 2)
    w = (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) * 2 * np.pi * r * np.diff(edges)
    zz = z[hit][:, None]
    p = rp_rs
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = np.clip((r ** 2 + zz ** 2 - p ** 2) / (2 * r * zz), -1, 1)
        frac = np.arccos(cosang) / np.pi
    frac = np.where(p >= r + zz, 1.0, frac)
    frac = np.where((r >= zz + p) | (r <= zz - p), 0.0, frac)
    out[hit] = 1 - (frac * w).sum(axis=1) / w.sum()
    return out


def transit_duration(period, rp_rs, a_rs, inc_deg):
    """First to fourth contact, in days, for a circular orbit."""
    b = a_rs * np.cos(np.deg2rad(inc_deg))
    return period / np.pi * np.arcsin(np.sqrt((1 + rp_rs) ** 2 - b ** 2) / (a_rs * np.sin(np.deg2rad(inc_deg))))


def fit_limb_darkening(time, flux, t0, period, rp_rs, a_rs, inc_deg):
    """Fit the two limb-darkening coefficients (and a flux scale) to a real light curve, geometry fixed."""
    from scipy.optimize import least_squares
    ok = np.isfinite(time) & np.isfinite(flux)
    ph = ck.phase_days(time[ok], period, t0)
    sel = np.abs(ph) < 0.2
    t, f = time[ok][sel], flux[ok][sel] / np.nanmedian(flux[ok])

    def resid(x):
        return x[2] * transit_model(t, t0, period, rp_rs, a_rs, inc_deg, x[0], x[1]) - f
    sol = least_squares(resid, [0.4, 0.2, 1.0], bounds=([0, -0.5, 0.9], [1, 1, 1.1]))
    return {"u1": float(sol.x[0]), "u2": float(sol.x[1]), "scale": float(sol.x[2]),
            "rms_ppm": float(np.std(resid(sol.x)) * 1e6), "n_points": int(sel.sum())}


def transit_depth(time, flux, period, t0, duration, stat="mean"):
    """Depth recipe used in Part 5, in parts per million (ppm).

    Flux is divided by its median over the whole file. 'In transit' is the central half of each transit
    (|time from mid-transit| < duration/4); 'around' is from 0.75 to 1.5 durations on either side.
    depth = stat(around) - stat(in transit). stat is 'mean' (default) or 'median'.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux)
    f = flux / np.median(flux[ok])
    ph = np.abs(ck.phase_days(time, period, t0))
    inn = ok & (ph < 0.25 * duration)
    out = ok & (ph > 0.75 * duration) & (ph < 1.5 * duration)
    agg = np.mean if stat == "mean" else np.median
    depth = float((agg(f[out]) - agg(f[inn])) * 1e6) if inn.any() and out.any() else float("nan")
    return {"depth_ppm": depth, "n_in": int(inn.sum()), "n_around": int(out.sum())}


def depth_per_transit(time, flux, target):
    """The Part 5 recipe applied to each transit on its own: one row per transit, depth in ppm.

    The flux is divided by its median over the whole file first, as in transit_depth. A transit whose depth
    disagrees with the others points to a problem in or around it.
    """
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux)
    f = flux / np.median(flux[ok])
    g = target
    n = np.round((time - g["t0_btjd"]) / g["period_days"])
    rows = []
    for k in np.unique(n[ok]):
        w = ok & (n == k)
        r = transit_depth(time[w], f[w], g["period_days"], g["t0_btjd"], g["duration_days"])
        if r["n_in"] and r["n_around"]:
            rows.append({"mid_btjd": round(g["t0_btjd"] + k * g["period_days"], 3), "depth_ppm": round(r["depth_ppm"]),
                         "n_in": r["n_in"], "n_around": r["n_around"]})
    return pd.DataFrame(rows)


def planet_radius_earth(depth_ppm, st_rad_solar):
    """Planet radius in Earth radii, assuming depth = (Rp/R*)^2 (no limb darkening). NaN if depth <= 0."""
    if not depth_ppm > 0:
        return float("nan")
    return float(np.sqrt(depth_ppm * 1e-6) * st_rad_solar * ck.R_SUN_IN_R_EARTH)


# --------------------------------------------------------------------------------------------------
# The twin
# --------------------------------------------------------------------------------------------------
@dataclass
class Twin:
    netid: str
    seed: int
    lc: pd.DataFrame
    catalog: pd.DataFrame
    lc_path: Path
    catalog_path: Path
    target: dict
    _truth_lc: pd.DataFrame = field(repr=False)
    _truth_catalog: pd.DataFrame = field(repr=False)
    _planted: list = field(repr=False)

    def __repr__(self):
        return (f"Twin(netid={self.netid!r}, light curve {len(self.lc)} rows -> {self.lc_path.name}, "
                f"catalog {len(self.catalog)} rows -> {self.catalog_path.name}, "
                f"{len(self._planted)} planted problems)")

    def truth_depth(self):
        """The Part 5 recipe applied to the twin before any problem was planted."""
        t = self._truth_lc
        g = self.target
        return transit_depth(t["time_btjd"], t["flux"], g["period_days"], g["t0_btjd"], g["duration_days"])

    def truth_radius(self):
        """Planet radius from the true radius ratio and the true stellar radius, in Earth radii."""
        g = self.target
        return float(g["rp_rs_true"] * g["st_rad_true"] * ck.R_SUN_IN_R_EARTH)

    def _depth(self, df):
        g = self.target
        return transit_depth(df["time_btjd"], df["flux"], g["period_days"], g["t0_btjd"], g["duration_days"])["depth_ppm"]

    def bias_alone(self):
        """Depth change (percent of the truth) that each planted light-curve problem causes on its own."""
        t = self._truth_lc
        base = self._depth(t)
        out = []
        for p in self._planted:
            if p["kind"] != "lc":
                continue
            fp = p["footprint"]
            h = t.copy()
            if p["category"] == "duplicates":
                h = pd.concat([h, t[t["cadenceno"].isin(fp["cadenceno"])]], ignore_index=True)
            else:
                i = h.index[h["cadenceno"].isin(fp["cadenceno"])]
                pos = pd.Series(np.arange(len(fp["cadenceno"])), index=fp["cadenceno"])
                k = pos[h.loc[i, "cadenceno"]].to_numpy()
                h.loc[i, "time_btjd"] += fp["d_time"][k]
                h.loc[i, "flux"] += fp["d_flux"][k]
            out.append((p["category"], (self._depth(h) - base) / base * 100))
        return out

    def bias_left(self, cleaned):
        """For a cleaned light curve: depth change (percent of the truth) that each planted problem still causes.

        Each problem's rows are put back to the truth (same normalization as yours, duplicated cadences
        dropped for the duplicates); the difference with your depth is what that problem still does to it.
        """
        t = self._truth_lc.set_index("cadenceno")
        c = cleaned[np.isfinite(cleaned["time_btjd"]) & np.isfinite(cleaned["flux"])]
        mine = self._depth(c)
        truth = self._depth(self._truth_lc)
        common = c["cadenceno"].isin(t.index)
        scale = np.median(c.loc[common, "flux"]) / np.median(t.loc[c.loc[common, "cadenceno"], "flux"])
        out = []
        for p in self._planted:
            if p["kind"] != "lc":
                continue
            if p["category"] == "duplicates":
                fixed = c.drop_duplicates("cadenceno")
            else:
                fixed = c.copy()
                sel = fixed["cadenceno"].isin(p["footprint"]["cadenceno"])
                cads = fixed.loc[sel, "cadenceno"]
                fixed.loc[sel, "time_btjd"] = t.loc[cads, "time_btjd"].to_numpy()
                fixed.loc[sel, "flux"] = t.loc[cads, "flux"].to_numpy() * scale
            out.append((p["category"], (mine - self._depth(fixed)) / truth * 100))
        return out


def _real_inputs():
    p = ck.read_planets().set_index("pl_name").loc[ck.TARGET]
    lc, h = ck.read_lightcurve()
    ld = ck.provenance()["synthetic_twin_inputs"]["limb_darkening_fit"]
    return p, lc, h, ld


def _segments(nan_time):
    """Index ranges [i0, i1) of the stretches with valid times, split by the Earth-point blocks."""
    good = ~nan_time
    edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


MUST_ADDRESS_PERCENT = 0.3    # Part 5c: a problem that still moves your depth by more than this must be dealt with
TWIN_NEEDS_PERCENT = 0.4      # every twin holds a problem that moves the depth this much after removing only
                              # duplicated cadences and zeros, plus a second row for its host star


def minimal_cleaning(lc):
    """Only the obvious: drop duplicated cadences and zero flux. Used to choose twins, never enough to pass 5c."""
    c = lc.drop_duplicates("cadenceno")
    return c[~(c["flux"] <= 0)]


def _twin_is_usable(tw):
    star_twice = any(p["category"] == "duplicate_planets" for p in tw._planted)
    left = [abs(b) for _, b in tw.bias_left(minimal_cleaning(tw.lc))]
    return star_twice and max(left, default=0) >= TWIN_NEEDS_PERCENT


def make_twin(netid=None, outdir=None):
    """Build the synthetic twin for a NetID (empty: the class seed), write its two CSV files, return it.

    The seed comes from the NetID. If its twin holds no problem that matters for the depth beyond duplicated
    cadences and zeros, or no second row for the host star, the next seed is tried (the class seed passes as is).
    """
    seed = ck.seed_from_netid(netid)
    for k in range(200):
        tw = _build_twin(netid, seed + k, outdir)
        if _twin_is_usable(tw):
            return tw
    return tw


def _build_twin(netid, seed, outdir):
    rng = np.random.default_rng(seed)
    p, lc, h, ld = _real_inputs()
    who = re.sub(r"[^a-z0-9]", "", (netid or "").lower()) or "class"

    # ---- truth: the real cadences, a constant star, the planet, white noise at the real level ----------
    cad = lc["CADENCENO"].to_numpy()
    t_real = lc["TIME"].to_numpy()
    nan_time = ~np.isfinite(t_real)
    t = np.interp(cad, cad[~nan_time], t_real[~nan_time])
    q_real = lc["QUALITY"].to_numpy()
    period, t0 = float(p["pl_orbper"]), float(p["pl_tranmid"] - 2457000.0)
    t0 = t0 + period * np.ceil((np.nanmin(t_real) - t0) / period)     # first transit of the sector
    rp, a_rs, inc = float(p["pl_ratror"]), float(p["pl_ratdor"]), float(p["pl_orbincl"])
    dur = float(transit_duration(period, rp, a_rs, inc))
    f0 = float(np.nanmedian(lc["SAP_FLUX"]))
    pdc = lc["PDCSAP_FLUX"].to_numpy(float)
    oot = np.isfinite(pdc) & (np.abs(ck.phase_days(t_real, period, t0)) > dur)
    rel_sigma = float(1.4826 * np.nanmedian(np.abs(pdc[oot] / np.nanmedian(pdc[oot]) - 1)))
    model = transit_model(t, t0, period, rp, a_rs, inc, ld["u1"], ld["u2"])
    flux = f0 * model + rng.normal(0, rel_sigma * f0, len(t))
    bkg = 2700.0 + rng.normal(0, 15.0, len(t))
    # uncertainty of each value: photon noise scales with the square root of the light collected
    ferr = rel_sigma * f0 * np.sqrt(model + bkg / f0) / np.sqrt(1 + 2700.0 / f0)
    quality = np.zeros(len(t), dtype=int)
    documented = nan_time | ((q_real & (8 | 32 | 128)) > 0)
    quality[nan_time] = 8
    quality[(q_real & 32) > 0] |= 32
    quality[(q_real & 128) > 0] |= 128
    flux[documented] = np.nan
    bkg[documented] = np.nan
    truth = pd.DataFrame({"time_btjd": t, "cadenceno": cad, "flux": flux,
                          "flux_err": np.where(documented, np.nan, ferr), "background": bkg,
                          "quality": quality})
    syn = truth.copy()
    segs = _segments(nan_time)
    transits = t0 + period * np.arange(np.ceil((t[0] - t0) / period), np.floor((t[-1] - t0) / period) + 1)
    usable = ~documented

    # ---- which problems, and where -------------------------------------------------------------------
    n_lc = int(rng.integers(4, 6))
    lc_cats = [str(c) for c in rng.choice(LC_CATS, size=n_lc, replace=False)]
    cat_cats = [str(c) for c in rng.choice(CAT_CATS, size=2, replace=False)]
    planted, busy = [], []

    def free(t_a, t_b, margin=0.15):
        return all(t_b + margin < a or t_a - margin > b for a, b in busy)

    def pick_block(n, near_transit=False, tries=400):
        for _ in range(tries):
            if near_transit:
                tc = rng.choice(transits)
                i0 = int(np.searchsorted(t, tc + rng.choice([-1, 1]) * rng.uniform(0.8, 1.3) * dur))
            else:
                i0 = int(rng.integers(0, len(t) - n))
            i1 = i0 + n
            if i1 >= len(t) or not usable[i0:i1].all():
                continue
            if free(t[i0], t[i1 - 1]):
                return i0, i1
        raise RuntimeError("could not place a planted problem")

    def plant(cat, t_a, t_b, **extra):
        busy.append((t_a, t_b))
        planted.append({"category": cat, "kind": "lc", "t0": float(t_a), "t1": float(t_b), **extra})

    def footprint(before):
        """Rows and changes of time and flux made by the problem just planted (no random draws here)."""
        after = syn[["time_btjd", "flux"]].to_numpy(float)
        changed = ~np.isclose(after, before, rtol=0, atol=1e-9, equal_nan=True)
        rows = np.flatnonzero(changed.any(axis=1))
        delta = np.nan_to_num(after[rows] - before[rows])
        planted[-1]["footprint"] = {"cadenceno": cad[rows], "d_time": delta[:, 0], "d_flux": delta[:, 1]}

    for cat in sorted(lc_cats, key=LC_CATS.index):
        before = syn[["time_btjd", "flux"]].to_numpy(float).copy()
        if cat == "time_offset":
            i0, i1 = segs[int(rng.integers(1, len(segs)))]
            corr = lc["TIMECORR"].to_numpy(float)[i0:i1]
            corr = np.where(np.isfinite(corr), corr, np.nanmedian(corr))
            syn.loc[i0:i1 - 1, "time_btjd"] = t[i0:i1] - corr
            plant(cat, t[i0], t[i1 - 1], shift_minutes=float(np.median(corr) * 1440))
        elif cat == "scattered_light":
            # Sector 56: the real background rises from about 3,700 to 13,300 e-/s (6-hour means) between BTJD
            # 2834.5 and a peak at 2835.32, and is back near 4,200 by 2836.0, in the middle of a stretch.
            i0, i1 = segs[int(rng.integers(0, len(segs)))]
            tc = rng.uniform(t[i0] + 1.0, t[i1 - 1] - 1.0)
            w = rng.uniform(0.2, 0.4)
            bump = np.exp(-0.5 * ((t - tc) / w) ** 2)
            excess = rng.uniform(6000, 15000)              # e-/s; the real peak is 11,858 above the median
            syn["background"] += excess * bump
            # imperfect background estimate: a few times the real coupling, so it shows above the white noise
            syn["flux"] += rng.choice([-1, 1]) * rng.uniform(0.01, 0.03) * excess * bump
            plant(cat, tc - 2 * w, tc + 2 * w)
        elif cat == "zero_gap":
            i0, i1 = pick_block(int(rng.integers(40, 120)), near_transit=rng.random() < 0.8)
            syn.loc[i0:i1 - 1, ["flux", "flux_err", "background"]] = 0.0
            syn.loc[i0:i1 - 1, "quality"] = 0
            plant(cat, t[i0], t[i1 - 1])
        elif cat == "forward_fill":
            # a stretch of lost values filled by repeating the last good row: the processing, not the star
            i0, i1 = pick_block(int(rng.integers(30, 90)), near_transit=rng.random() < 0.5)
            for col in ("flux", "flux_err", "background"):
                syn.loc[i0:i1 - 1, col] = float(syn.loc[i0, col])
            plant(cat, t[i0], t[i1 - 1])
        elif cat == "flare":
            if rng.random() < 0.6:
                tc = rng.choice(transits)
                tf = tc + rng.uniform(-0.25, 0.25) * dur
            else:
                tf = rng.uniform(t[0] + 0.5, t[-1] - 0.5)
            for _ in range(200):
                if free(tf - 0.01, tf + 0.2) and usable[np.searchsorted(t, tf)]:
                    break
                tf = rng.uniform(t[0] + 0.5, t[-1] - 0.5)
            amp, tau = rng.uniform(0.002, 0.008), rng.uniform(8, 25) / 1440
            x = t - tf
            prof = np.where(x < 0, np.clip(1 + x / (3 / 1440), 0, None), np.exp(-np.clip(x, 0, None) / tau))
            syn["flux"] += amp * f0 * prof
            plant(cat, tf - 3 / 1440, tf + 5 * tau, amplitude_percent=float(amp * 100))
        elif cat == "outliers":
            # single-value data glitches (a cosmic ray cannot reach this size on a star this bright)
            times = []
            for _ in range(int(rng.integers(3, 7))):
                i0, _i1 = pick_block(1)
                syn.loc[i0, "flux"] += rng.uniform(0.006, 0.03) * f0
                times.append(float(t[i0]))
                busy.append((t[i0], t[i0]))
            planted.append({"category": cat, "kind": "lc", "times": times,
                            "t0": min(times), "t1": max(times)})
        elif cat == "flux_steps":
            # a sudden change of sensitivity or of the star's share in the aperture: multiplicative, 0.1-0.3%
            # (the one real step in sector 56 is about 820 ppm, across the downlink at BTJD 2838.6)
            times = []
            for _ in range(int(rng.integers(2, 4))):
                i0, _i1 = pick_block(1, near_transit=rng.random() < 0.4)
                s_end = next(b for a, b in segs if a <= i0 < b)
                syn.loc[i0:s_end - 1, "flux"] *= 1 + rng.choice([-1, 1]) * rng.uniform(0.001, 0.003)
                times.append(float(t[i0]))
                busy.append((t[i0], t[i0]))
            planted.append({"category": cat, "kind": "lc", "times": times,
                            "t0": min(times), "t1": max(times)})
        footprint(before)
    syn["flux"] = syn["flux"].where(~documented, np.nan)
    if "duplicates" in lc_cats:
        i0, i1 = pick_block(int(rng.integers(20, 80)))
        block = syn.iloc[i0:i1]
        syn = pd.concat([syn.iloc[:i1], block, syn.iloc[i1:]], ignore_index=True)
        plant("duplicates", t[i0], t[i1 - 1])
        planted[-1]["footprint"] = {"cadenceno": cad[i0:i1], "d_time": np.zeros(i1 - i0), "d_flux": np.zeros(i1 - i0)}

    # ---- the catalog ---------------------------------------------------------------------------------
    real = ck.read_planets()
    m = ck.measured_mass_and_radius(real)
    m = m[(m["pl_bmasseerr1"] / m["pl_bmasse"] < 0.3) & (m["pl_radeerr1"] / m["pl_rade"] < 0.3)
          & m["st_rad"].notna() & (m["pl_name"] != ck.TARGET)]
    n_cat = int(rng.integers(160, 200))
    base = m.sample(n=n_cat, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    names = [f"SYN-{i:04d} b" for i in rng.choice(np.arange(1, 10000), size=n_cat, replace=False)]
    rad = np.clip(base["pl_rade"] * (1 + rng.normal(0, 0.05, n_cat)), 0.3, None)
    mass = np.clip(base["pl_bmasse"] * (1 + rng.normal(0, 0.10, n_cat)), 0.05, None)
    prov = np.array(["Mass"] * n_cat, dtype=object)
    mr = rng.random(n_cat) < 0.3
    mr &= rad < 11.1                      # the archive computes masses only below 11.1 Earth radii
    mass = np.where(mr, ck.chen_kipping_mass(rad), mass)
    prov[mr] = "M-R relationship"
    cat = pd.DataFrame({"pl_name": names, "hostname": [n[:-2] for n in names],
                        "discoverymethod": "Transit", "disc_year": base["disc_year"].astype(int),
                        "pl_orbper": base["pl_orbper"].round(5), "pl_rade": np.round(rad, 3),
                        "pl_bmasse": np.round(mass, 3), "pl_bmassprov": prov,
                        "st_rad": base["st_rad"].round(3), "st_teff": base["st_teff"].round(0),
                        "sy_dist": base["sy_dist"].round(2)})
    tgt = pd.DataFrame([{"pl_name": TARGET_SYN, "hostname": "SYN-209458", "discoverymethod": "Transit",
                         "disc_year": 2022, "pl_orbper": round(period, 8),
                         "pl_rade": round(rp * float(p["st_rad"]) * ck.R_SUN_IN_R_EARTH, 3),
                         "pl_bmasse": round(float(p["pl_bmasse"]), 3), "pl_bmassprov": "Mass",
                         "st_rad": round(float(p["st_rad"]), 3), "st_teff": float(p["st_teff"]),
                         "sy_dist": round(float(p["sy_dist"]), 2)}])
    cat = pd.concat([cat, tgt], ignore_index=True)
    true_cat = cat.copy()
    for c in sorted(cat_cats, key=CAT_CATS.index):
        if c == "jupiter_radii":
            giants = cat.index[(cat["pl_rade"] > 6) & (cat["pl_name"] != TARGET_SYN)]
            rows = rng.choice(giants, size=min(len(giants), int(rng.integers(4, 8))), replace=False)
            cat.loc[rows, "pl_rade"] = (cat.loc[rows, "pl_rade"] / ck.R_JUP_IN_R_EARTH).round(4)
            planted.append({"category": c, "kind": "catalog", "names": list(cat.loc[rows, "pl_name"])})
        elif c == "mislabelled_mass":
            cand = cat.index[cat["pl_bmassprov"] == "M-R relationship"]
            rows = rng.choice(cand, size=min(len(cand), int(rng.integers(5, 10))), replace=False)
            cat.loc[rows, "pl_bmassprov"] = "Mass"
            planted.append({"category": c, "kind": "catalog", "names": list(cat.loc[rows, "pl_name"])})
        elif c == "duplicate_planets":
            others = rng.choice(cat.index[cat["pl_name"] != TARGET_SYN], size=int(rng.integers(2, 5)),
                                replace=False)
            rows = [int(cat.index[cat["pl_name"] == TARGET_SYN][0])] + [int(i) for i in others]
            dup = cat.loc[rows].copy()
            dup["pl_name"] = [n.replace("SYN-", "SYN ") if k % 2 else n.replace(" b", "b")
                              for k, n in enumerate(dup["pl_name"])]
            dup["hostname"] = dup["pl_name"].str.replace(r"\s?b$", "", regex=True)
            dup["pl_rade"] = (dup["pl_rade"] * (1 + rng.normal(0, 0.04, len(dup)))).round(3)
            dup["pl_bmasse"] = (dup["pl_bmasse"] * (1 + rng.normal(0, 0.08, len(dup)))).round(3)
            dup["disc_year"] = np.minimum(dup["disc_year"] + rng.integers(1, 4, len(dup)), 2026)
            k = rng.choice([rng.uniform(0.85, 0.92), rng.uniform(1.08, 1.15)])
            dup.loc[dup.index[0], "st_rad"] = round(float(p["st_rad"]) * k, 3)
            cat = pd.concat([cat, dup], ignore_index=True)
            planted.append({"category": c, "kind": "catalog",
                            "names": list(cat.loc[rows, "pl_name"]) + list(dup["pl_name"])})
    cat = cat.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

    outdir = Path(outdir) if outdir else ck.output_dir()
    lc_path, cat_path = outdir / f"synthetic_tess_lightcurve_{who}.csv", outdir / f"synthetic_planets_{who}.csv"
    syn.to_csv(lc_path, index=False, float_format="%.7f")
    cat.to_csv(cat_path, index=False)
    target = {"name": TARGET_SYN, "period_days": period, "t0_btjd": float(t0), "duration_days": dur,
              "rp_rs_true": rp, "st_rad_true": float(p["st_rad"]), "a_rs": a_rs, "inc_deg": inc,
              "flux_level_e_per_s": f0, "noise_ppm": rel_sigma * 1e6}
    return Twin(netid or "", seed, pd.read_csv(lc_path), pd.read_csv(cat_path), lc_path, cat_path, target,
                truth, true_cat, planted)


def observed_minus_computed(time, flux, target, search_minutes=30):
    """For each predicted transit, the time shift (minutes) that best lines the data up with the model.

    A physical check of the clock: a planet on a circular orbit transits on a strict schedule, so the shift
    should be near zero for every transit, with scatter of about a minute for a planet this size.
    """
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    g = target
    ld = ck.provenance()["synthetic_twin_inputs"]["limb_darkening_fit"]
    shifts = np.arange(-search_minutes, search_minutes + 0.25, 0.25) / 1440
    tcs = g["t0_btjd"] + g["period_days"] * np.arange(0, int((np.nanmax(time) - g["t0_btjd"]) / g["period_days"]) + 1)
    rows = []
    for tc in tcs:
        w = ok & (np.abs(time - tc) < 1.5 * g["duration_days"])
        if w.sum() < 100:
            continue
        t, f = time[w], flux[w] / np.median(flux[w][np.abs(time[w] - tc) > 0.6 * g["duration_days"]])
        chi = [np.sum((f - transit_model(t - s, g["t0_btjd"], g["period_days"], g["rp_rs_true"], g["a_rs"],
                                          g["inc_deg"], ld["u1"], ld["u2"])) ** 2) for s in shifts]
        rows.append({"predicted_mid_btjd": round(float(tc), 4), "shift_minutes": float(shifts[int(np.argmin(chi))] * 1440),
                     "n_points": int(w.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Scoring the hunt
# --------------------------------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _matches(f, p):
    if not str(f.get("what", "")).strip():
        return False
    if p["kind"] == "catalog":
        names = f.get("planets") or f.get("planet") or []
        names = [names] if isinstance(names, str) else list(names)
        want = {_norm(n) for n in p["names"]}
        return any(_norm(n) in want for n in names)
    try:
        a = float(f.get("start"))
        b = float(f.get("end", f.get("start")))
    except (TypeError, ValueError):
        return False
    a, b = min(a, b), max(a, b)
    span_ok = 3.0 if "times" in p else max(3.0, 3 * (p["t1"] - p["t0"]))
    if b - a > span_ok:
        return False                      # a finding that spans most of the sector locates nothing
    tol = 0.05
    if p["category"] == "time_offset" and not re.search(r"time|clock|offset|shift|early|late|barycent|timing",
                                                         str(f.get("what", "")).lower()):
        return False
    if p["category"] == "flux_steps":     # a step persists: either end of the window may mark it
        return any(abs(a - x) <= 0.1 or abs(b - x) <= 0.1 for x in p["times"])
    if "times" in p:
        return any(a - tol <= x <= b + tol for x in p["times"])
    return b + tol >= p["t0"] and a - tol <= p["t1"]


def score_findings(findings, twin, reveal=False, who="you"):
    """Count the planted problems that at least one finding locates. reveal=True lists the missed kinds."""
    if not findings or all(not str(f.get("what", "")).strip() for f in findings):
        print(NOT_ANSWERED)
        return None
    found = [any(_matches(f, p) for f in findings) for p in twin._planted]
    unmatched = [i for i, f in enumerate(findings) if not any(_matches(f, p) for p in twin._planted)]
    print(f"{who[0].upper() + who[1:]} located {sum(found)} of the {len(found)} planted problems.")
    if unmatched:
        listed = "; ".join(f"{i + 1}: {str(findings[i].get('what', ''))[:40]}" for i in unmatched)
        print(f"{len(unmatched)} finding(s) match no planted problem ({listed}): either something real "
              "in the twin (transits, documented gaps and momentum dumps, honest 'M-R relationship' masses), "
              "or a problem placed at the wrong time or planet.")
    if sum(found) < len(found) and not reveal:
        print("To see which kinds you missed (not where they are), run the same call with reveal=True.")
    if reveal:
        missed = [CATEGORY_WORDS[p["category"]] for p, ok in zip(twin._planted, found) if not ok]
        print("Missed kinds:" if missed else "Nothing missed.")
        for m_ in missed:
            print("  -", m_)
    return sum(found)


STILL_WRONG = {
    "outliers": "single 2-minute values that no star produces still sit in or around a transit",
    "flux_steps": "around at least one transit the flux level is not the level inside it: something changed "
                  "the level and it stayed",
    "scattered_light": "light that does not come from the star still raises or lowers the flux near a transit; "
                       "compare flux and background there",
    "zero_gap": "values that are not light at all still sit near a transit",
    "time_offset": "in one stretch the transits do not arrive when the orbit says, so the windows cut the dip",
    "duplicates": "some cadences still count twice",
    "forward_fill": "a stretch near a transit still repeats one value",
    "flare": "light from the star itself still fills part of a transit",
}


def _host_rows(twin):
    host = twin.catalog["hostname"].astype(str).str.replace(" ", "-")
    return twin.catalog[host.str.replace("-b$", "", regex=True).str.startswith("SYN-209458")]


def assess_cleaning(cleaned, twin, st_rad=None):
    """What 5c checks, as numbers: your depth, how far it is from the truth, what still moves it, your star."""
    truth = twin.truth_depth()["depth_ppm"]
    given = twin._depth(twin.lc)
    out = {"truth_ppm": truth, "given_ppm": given, "cleaned_ppm": None, "passed": None}
    if cleaned is None or not isinstance(cleaned, pd.DataFrame):
        return out
    if len(cleaned) == len(twin.lc) and cleaned.reset_index(drop=True).equals(twin.lc):
        return out
    mine = twin._depth(cleaned)
    out["cleaned_ppm"] = mine
    out["off_percent"] = (mine - truth) / truth * 100 if np.isfinite(mine) else float("nan")
    out["left"] = ([(c, b) for c, b in twin.bias_left(cleaned) if abs(b) > MUST_ADDRESS_PERCENT]
                   if "cadenceno" in cleaned else None)
    star_values = sorted(_host_rows(twin)["st_rad"].unique())
    out["star_rows"] = len(star_values)
    out["star_ok"] = st_rad is not None and abs(float(st_rad) - twin.target["st_rad_true"]) < 0.0015
    out["passed"] = bool(np.isfinite(out["off_percent"]) and abs(out["off_percent"]) <= 1.5
                         and out["left"] == [] and out["star_ok"])
    return out


def check_depth(cleaned, twin, st_rad=None):
    """Self-check for Part 5c. Passes when (1) your depth is within 1.5% of the recipe on the truth, (2) no planted
    problem still moves it by more than 0.25%, and (3) my_st_rad is the radius of the star the twin was copied from.
    Hints say what is still wrong, never where."""
    a = assess_cleaning(cleaned, twin, st_rad)
    if a["cleaned_ppm"] is None:
        print(NOT_ANSWERED + ": my_clean is still the file as given")
        return None
    if a["passed"]:
        print(f"correct: your depth is {a['off_percent']:+.1f}% from the recipe on the truth, no planted problem "
              f"still moves it by more than {MUST_ADDRESS_PERCENT}%, and your stellar radius is the star's")
        return True
    msg = []
    if not np.isfinite(a["off_percent"]):
        msg.append("no depth: keep the transits and the stretches around them in my_clean")
    elif abs(a["off_percent"]) > 1.5:
        msg.append(f"your depth is {a['off_percent']:+.1f}% from the recipe on the truth (aim for 1.5%)")
    if a["left"] is None:
        msg.append("keep the column cadenceno in my_clean, so your rows can be compared with the truth")
    for c, b in a["left"] or []:
        msg.append(f"{STILL_WRONG[c]} (it moves your depth by {b:+.2f}%)")
    if st_rad is None:
        msg.append("set my_st_rad: the planet radius is the quantity this track asks for")
    elif not a["star_ok"]:
        msg.append("your stellar radius: cat5 has " + ("more than one row for this star; which radius belongs "
                   "to the real star the twin was copied from (Part 0)?" if a["star_rows"] > 1 else
                   "one row for this star; use its st_rad"))
    text = "; ".join(msg)
    print("not yet. Hint: " + text + ("" if text.endswith("?") else "."))
    return False


def check_ai_depth(depth_ppm, twin):
    """Part 8: how far the depth reported by the AI tool is from the recipe applied to the truth."""
    if depth_ppm is None:
        print(NOT_ANSWERED)
        return None
    truth = twin.truth_depth()["depth_ppm"]
    off = (float(depth_ppm) - truth) / truth * 100
    print(f"its depth is {off:+.1f}% from the truth "
          + ("(within 1.5%)" if abs(off) <= 1.5 else
             "(more than 1.5%: look at what it removed in transit, or left around the transits)"))
    return abs(off) <= 1.5


AI_INSTRUCTIONS = """I attach two CSV files from a synthetic copy of a NASA TESS light curve and of a planet table.

File 1, {lc}: one row per 2-minute cadence of a Sun-like star with a transiting giant planet.
Columns: time_btjd (Barycentric TESS Julian Date, BJD - 2457000, days), cadenceno (cadence counter),
flux and flux_err (electrons per second, simple aperture photometry), background (electrons per second,
already subtracted from flux), quality (bit flags: 8 = spacecraft pointed at Earth for data downlink,
32 = momentum dump, 128 = manual exclude; rows with these flags have no flux on purpose).

File 2, {cat}: one row per planet. Columns as in the NASA Exoplanet Archive composite table:
pl_rade (planet radius, Earth radii), pl_bmasse (planet mass, Earth masses), pl_bmassprov
(how the mass was obtained), st_rad (stellar radius, solar radii), pl_orbper (days), and others.

Tasks:
1. List every problem in these files that a real instrument, processing pipeline or catalog could produce.
   For each one give: what it is, where it is (a time range in BTJD, or the planet names), and the physical
   reason it cannot be a property of the star or the planet. Say which values you would remove, which you
   would flag, and which you would keep because they are real.
2. For planet SYN-209458 b (period {P:.8f} days, mid-transit time {T0:.5f} BTJD, transit duration
   {D:.4f} days), compute the transit depth in parts per million with this recipe: divide flux by its median;
   'in transit' = |time from mid-transit| < duration/4; 'around' = between 0.75 and 1.5 durations from
   mid-transit; depth = mean(around) - mean(in transit). Do it on the file as given, and again after your
   cleaning.
3. Compute the planet radius in Earth radii as sqrt(depth) x stellar radius x 109.08, with the stellar
   radius of SYN-209458 in file 2. Say which row you used and why.
Show your code."""


def ai_instructions(twin):
    g = twin.target
    return AI_INSTRUCTIONS.format(lc=twin.lc_path.name, cat=twin.catalog_path.name, P=g["period_days"],
                               T0=g["t0_btjd"], D=g["duration_days"])
