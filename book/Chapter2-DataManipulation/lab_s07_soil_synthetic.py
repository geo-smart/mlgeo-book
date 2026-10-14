"""Synthetic twin of the USCRN Spokane 17 SSW daily soil record, with planted problems.

Used in Part 5 and Part 8 of lab_s07_soil_moisture.ipynb (Session 7, soil moisture track).

What it builds
--------------
1. A *true* daily record, 2015-01-01 to 2024-12-31, at the same station, with the
   same columns and units as the U.S. Climate Reference Network (USCRN) daily file:
   air temperature (degrees C), precipitation (mm), volumetric soil moisture
   (m3/m3) and soil temperature (degrees C) at 5, 10, 20, 50 and 100 cm.
   - Weather comes from a stochastic generator whose month-by-month statistics
     (chance of rain after a dry or a wet day, rain amounts, temperature cycle,
     day-to-day persistence and daily range) are measured on the real
     course copy, snapshots/uscrn_WA_Spokane_17_SSW_2015-2024.txt.
   - Soil temperature follows air temperature with a lag and damping that grow
     with depth, and stays near 0 degrees C under snow. At 20, 50 and 100 cm it
     is also pulled toward the annual mean, as heat from the ground below keeps
     deep soil from freezing; and it never falls below the lowest daily value
     the real probe recorded at that depth (0.1, 0.3 and 0.5 degrees C).
   - Soil water follows a five-layer bucket model: snow accumulates below 0 C
     and melts with a degree-day factor, water enters the top layer, drains
     downward above field capacity, and leaves by evaporation (top layer) and
     by root uptake (all layers) at a rate set by a temperature-based
     potential evaporation (Hargreaves and Samani, 1985).
   - Residual and wet-end water contents of each layer are taken from the real
     record (1st and 99th percentiles), so the twin has the real site's range.
   - The constants were tuned by hand so that the 5 cm monthly means and the dry
     floors resemble the real record. Two layers do not: the 10 cm layer freezes
     more often than the real one, and the 100 cm layer wets later in autumn.
     Compare the twin only with its own truth.
   - Days when the daily mean soil temperature at a probe depth is below 0 C
     are left blank, as USCRN does for frozen soil, because the probe cannot
     measure water in frozen soil. (USCRN blanks a day when hourly readings were
     flagged, so a real day with a mean of +0.5 C can still be blank.)
   The truth uses one fixed seed, so every student has the same truth.

2. An *observed* copy: the truth plus small measurement noise and rounding,
   then six problems planted from a list of ten, chosen and placed with a seed
   derived from the student's UW NetID, so classmates get different sets.
   If the problems drawn move the 5 cm response to rain by less than MIN_BIAS
   (15%) from the truth, they are drawn again with the next seed (seed + 1,
   + 2, ...), so every twin has a damage to measure in Part 5.
   CLASS_SEED is used for the worked solution and for the book build.

The planted list stays inside the Twin object. check_findings() scores a list of
findings against it and only reveals which categories were missed when asked.

Nothing here downloads anything. Running time is under two seconds.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REAL_FILE = HERE / "snapshots" / "uscrn_WA_Spokane_17_SSW_2015-2024.txt"
LATITUDE_DEG = 47.42            # station latitude from the data file (field 5)
CLASS_SEED = 629                # worked solution and book build; chosen so its six problems include some that
                                # move the 5 cm rain response and some that do not
TRUTH_SEED = 2015               # one truth for the whole class
N_PLANTED = 6                   # problems per student, out of the ten types below
MIN_BIAS = 0.15                 # the twin as delivered moves the 5 cm response to rain at least this far from the truth
DEPTHS_CM = [5, 10, 20, 50, 100]
SM = [f"sm_{d}cm" for d in DEPTHS_CM]
ST = [f"st_{d}cm" for d in DEPTHS_CM]
COLUMNS = ["date", "t_max_c", "t_min_c", "t_mean_c", "precip_mm"] + SM + ST

USCRN_NAMES = ("WBANNO LST_DATE CRX_VN LONGITUDE LATITUDE T_DAILY_MAX T_DAILY_MIN T_DAILY_MEAN T_DAILY_AVG "
               "P_DAILY_CALC SOLARAD_DAILY SUR_TEMP_DAILY_TYPE SUR_TEMP_DAILY_MAX SUR_TEMP_DAILY_MIN "
               "SUR_TEMP_DAILY_AVG RH_DAILY_MAX RH_DAILY_MIN RH_DAILY_AVG SOIL_MOISTURE_5_DAILY "
               "SOIL_MOISTURE_10_DAILY SOIL_MOISTURE_20_DAILY SOIL_MOISTURE_50_DAILY SOIL_MOISTURE_100_DAILY "
               "SOIL_TEMP_5_DAILY SOIL_TEMP_10_DAILY SOIL_TEMP_20_DAILY SOIL_TEMP_50_DAILY "
               "SOIL_TEMP_100_DAILY").split()

# The ten kinds of problem that can be planted. Keys are the names to use in your findings.
PROBLEM_TYPES = {
    "spike": "isolated one-day jumps in a soil moisture column",
    "drift": "a slow, steady change in one probe that is not in the soil",
    "stuck": "a probe that repeats the same value day after day",
    "frozen_read_as_dry": "frozen-soil days reported as very dry instead of left blank",
    "missing_code_left_in": "a missing-value code written as a number",
    "gauge_outage_as_zero": "days when the precipitation gauge failed, written as 0 mm",
    "rain_one_day_late": "precipitation written one day after it fell",
    "probes_swapped": "two probe depths exchanged after a maintenance visit",
    "duplicated_days": "days that appear twice in the table",
    "units_percent": "soil moisture written in percent instead of m3/m3",
}


# ----------------------------------------------------------------------------
# Reading the real course copy (calibration only)
# ----------------------------------------------------------------------------
def read_uscrn_daily(path=REAL_FILE):
    """Read a USCRN daily01 file; return a DataFrame indexed by LST date, missing codes as NaN."""
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=USCRN_NAMES,
                      dtype={"CRX_VN": str, "LST_DATE": str})
    df = raw.replace([-9999.0, -99.0], np.nan)
    df.index = pd.to_datetime(df["LST_DATE"], format="%Y%m%d")
    df.index.name = "date"
    return df


def seed_from_netid(netid):
    """A reproducible integer seed from a UW NetID (or the class seed for the placeholder)."""
    netid = (netid or "").strip().lower()
    if netid in ("", "your_netid", "class"):
        return CLASS_SEED
    return int(hashlib.sha256(netid.encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------
# 1. The truth
# ----------------------------------------------------------------------------
def _calibration(real):
    """Month-by-month weather statistics and soil water range measured on the real record."""
    p = real["P_DAILY_CALC"]
    wet = p >= 0.3                                        # 0.3 mm: three times the 0.1 mm reporting resolution
    prev = wet.shift(1)
    month = real.index.month
    cal = {"p_wd": {}, "p_ww": {}, "gamma_k": {}, "gamma_theta": {}, "dtr": {}, "dtr_sd": {}}
    for m in range(1, 13):
        sel = month == m
        dry_prev = sel & (prev == False)                  # noqa: E712  (NaN-safe)
        wet_prev = sel & (prev == True)                   # noqa: E712
        cal["p_wd"][m] = float(wet[dry_prev].mean())
        cal["p_ww"][m] = float(wet[wet_prev].mean())
        amt = p[sel & wet]
        mu, var = float(amt.mean()), float(amt.var())
        cal["gamma_k"][m] = mu * mu / var
        cal["gamma_theta"][m] = var / mu
        dtr = (real["T_DAILY_MAX"] - real["T_DAILY_MIN"])[sel]
        cal["dtr"][m], cal["dtr_sd"][m] = float(dtr.mean()), float(dtr.std())
    # seasonal cycle of daily mean temperature: two harmonics, least squares
    t = real["T_DAILY_MEAN"].dropna()
    w = 2 * np.pi * t.index.dayofyear.values / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    coef, *_ = np.linalg.lstsq(X, t.values, rcond=None)
    anom = pd.Series(t.values - X @ coef, index=t.index).asfreq("D")
    cal["t_coef"] = coef
    cal["t_phi"] = float(anom.autocorr(1))
    cal["t_sd"] = {m: float(anom[anom.index.month == m].std()) for m in range(1, 13)}
    cal["theta_r"] = [float(real[f"SOIL_MOISTURE_{d}_DAILY"].quantile(0.01)) for d in DEPTHS_CM]
    cal["theta_w"] = [float(real[f"SOIL_MOISTURE_{d}_DAILY"].quantile(0.99)) for d in DEPTHS_CM]
    # lowest daily soil temperature the real probes recorded at 20, 50 and 100 cm: a floor for the twin
    cal["st_floor"] = [float(real[f"SOIL_TEMP_{d}_DAILY"].min()) for d in DEPTHS_CM[2:]]
    return cal


def _weather(cal, dates, rng):
    n = len(dates)
    month = dates.month.values
    doy = dates.dayofyear.values
    p = np.zeros(n)
    is_wet = False
    for i in range(n):
        m = month[i]
        prob = cal["p_ww"][m] if is_wet else cal["p_wd"][m]
        is_wet = rng.random() < prob
        if is_wet:
            p[i] = max(0.3, rng.gamma(cal["gamma_k"][m], cal["gamma_theta"][m]))
    w = 2 * np.pi * doy / 365.25
    c = cal["t_coef"]
    clim = c[0] + c[1] * np.cos(w) + c[2] * np.sin(w) + c[3] * np.cos(2 * w) + c[4] * np.sin(2 * w)
    a = np.zeros(n)
    phi = cal["t_phi"]
    for i in range(1, n):
        sd = cal["t_sd"][month[i]] * np.sqrt(1 - phi ** 2)
        a[i] = phi * a[i - 1] + rng.normal(0, sd)
    tmean = clim + a
    dtr = np.array([cal["dtr"][m] for m in month]) + rng.normal(0, 1, n) * np.array([cal["dtr_sd"][m] for m in month])
    dtr = np.clip(dtr - 3.0 * (p > 2), 2.0, None)        # cloudy rain days have a smaller daily range
    return pd.DataFrame({"t_max_c": tmean + dtr / 2, "t_min_c": tmean - dtr / 2, "t_mean_c": tmean,
                         "precip_mm": p}, index=dates)


def _extraterrestrial_mm(doy, lat_deg=LATITUDE_DEG):
    """Top-of-atmosphere radiation in mm/day of evaporated water (FAO-56 equations 21 to 25)."""
    lat = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(ws))
    return 0.408 * ra                                     # MJ m-2 day-1 to mm/day


# Model constants, tuned by hand (not fitted) so that the 5 cm monthly means and the dry floors
# resemble the real record. The 10 cm layer freezes far more often than the real one, and the
# 100 cm layer wets later in autumn than the real one.
SOIL = {
    "dz_mm": [80.0, 50.0, 100.0, 150.0, 250.0],     # water-holding thickness represented by each probe
    "fc_frac": 0.92,                                # field capacity as a fraction of the 99th percentile
    "kd": [0.45, 0.405, 0.3, 0.2, 0.03],            # share of the water above field capacity drained per day
    "root": [0.10, 0.15, 0.25, 0.25, 0.25],         # share of root uptake by layer
    "evap": 0.14,                                   # soil evaporation as a share of potential evaporation
    "transp": 0.5,                                  # root uptake as a share of potential evaporation
    "green_end": 300,                               # last day of year with active roots
    "bypass": 0.2,                                  # share of infiltration that skips to 50 and 100 cm
    "interception": 3.0,                            # mm of each rain day caught by plants, litter and the top 2 cm
    "lag": [0.55, 0.40, 0.22, 0.08, 0.035],         # soil temperature relaxation per day
    "cold_cap": [2.5, 0.9, 0.0, 0.0, 0.0],          # how far below 0 C each depth can go (latent heat holds it)
    "deep": [0.0, 0.0, 0.02, 0.03, 0.04],           # relaxation per day toward the annual mean (heat from below)
}


def _soil(weather, cal, prm=None):
    """Soil temperature and a five-layer water balance. Returns (theta, soil_temp), both (n, 5)."""
    prm = {**SOIL, **(prm or {})}
    n = len(weather)
    tmean, tmax, tmin = (weather[c].values for c in ("t_mean_c", "t_max_c", "t_min_c"))
    p = weather["precip_mm"].values
    doy = weather.index.dayofyear.values
    pet = 0.0023 * _extraterrestrial_mm(doy) * np.clip(tmean + 17.8, 0, None) * np.sqrt(np.clip(tmax - tmin, 0, None))

    dz = np.array(prm["dz_mm"])
    th_r = np.array(cal["theta_r"]) * 0.95
    th_s = np.array(cal["theta_w"]) * 1.02
    th_fc = np.array(cal["theta_w"]) * prm["fc_frac"]
    kd, root, lag = (np.array(prm[k]) for k in ("kd", "root", "lag"))
    cap = np.array(prm["cold_cap"])
    deep = np.array(prm["deep"])
    t_annual = float(np.mean(tmean)) + 1.5                # mean soil surface temperature
    floor = np.array(cal["st_floor"])

    theta = np.zeros((n, 5))
    temp = np.zeros((n, 5))
    th = th_fc * 0.9
    ts = np.full(5, 2.0)
    snow = 0.0
    for i in range(n):
        # snow and melt (degree-day factor 3 mm per degree C per day)
        if tmean[i] < 0.5:
            snow += p[i]
            water_in = 0.0
        else:
            water_in = max(p[i] - prm["interception"], 0.0)
        melt = min(snow, 3.0 * max(tmean[i], 0.0))
        snow -= melt
        water_in += melt
        # soil temperature: surface forcing insulated by snow, lagged and damped with depth;
        # below 0 C the latent heat of freezing water holds each depth close to 0 C
        surf = tmean[i] + 1.5
        if snow > 5.0:
            surf = 0.15 * min(surf, 0.0) + 0.2
        free = ts + lag * (surf - ts) + deep * (t_annual - ts)
        with np.errstate(divide="ignore", invalid="ignore"):
            held = np.where(cap > 0, -cap * (1 - np.exp(free / np.where(cap > 0, cap, 1))), 0.05)
        ts = np.where(free < 0, held, free)
        ts[2:] = np.maximum(ts[2:], floor)                # backstop: never colder than the real probe ever was
        frozen = ts < 0
        # infiltration: a frozen top layer refuses water, which waits on the surface
        if frozen[0]:
            snow += water_in
            water_in = 0.0
        # part of the water bypasses the upper soil through cracks and root channels (preferential flow)
        bypass = prm["bypass"] * water_in
        water_in -= bypass
        th[3] += 0.5 * bypass / dz[3]
        th[4] += 0.5 * bypass / dz[4]
        for k in range(5):
            if frozen[k]:
                break
            th[k] += water_in / dz[k]
            water_in = 0.0
            if th[k] > th_s[k]:
                water_in = (th[k] - th_s[k]) * dz[k]
                th[k] = th_s[k]
            q = kd[k] * max(th[k] - th_fc[k], 0.0) * dz[k]
            th[k] -= q / dz[k]
            if k < 4 and not frozen[k + 1]:
                th[k + 1] += q / dz[k + 1]
            elif k < 4:
                th[k] += q / dz[k]
        # evaporation from the top layer and root uptake, limited by the water above the dry floor
        beta = np.clip((th - th_r) / (th_fc - th_r), 0, 1)
        green = 1.0 if 4.0 < tmean[i] and 80 <= doy[i] <= prm["green_end"] else 0.1
        evap = 0.0 if frozen[0] else prm["evap"] * pet[i] * beta[0]
        uptake = prm["transp"] * green * pet[i] * root * np.sqrt(beta) * (~frozen)
        loss = uptake.copy()
        loss[0] += evap
        th = np.maximum(th - loss / dz, th_r)
        theta[i] = th
        temp[i] = ts
    return theta, temp


def make_truth(seed=TRUTH_SEED, real_path=REAL_FILE, prm=None):
    """The true daily record (no noise, no problems). Frozen probe days are NaN, as USCRN reports them.

    Returns (truth table indexed by date, {sm column: water content the model holds on every day,
    frozen or not}).
    """
    real = read_uscrn_daily(real_path)
    cal = _calibration(real)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="D", name="date")
    wx = _weather(cal, dates, rng)
    theta_end, temp = _soil(wx, cal, prm)
    # The file reports daily means of hourly values, not end-of-day states. On a rain day the storm
    # arrives at a random time of day, so only part of its jump shows in that day's mean and the
    # rest in the next day's; on a dry day the mean sits halfway between the two end-of-day states.
    before = np.vstack([theta_end[:1], theta_end[:-1]])
    f = np.where(wx["precip_mm"].values > 0, rng.uniform(0, 1, len(wx)), 0.5)[:, None]
    theta = f * before + (1 - f) * theta_end
    truth = wx.copy()
    for k, d in enumerate(DEPTHS_CM):
        truth[f"sm_{d}cm"] = np.where(temp[:, k] < 0, np.nan, theta[:, k])
    for k, d in enumerate(DEPTHS_CM):
        truth[f"st_{d}cm"] = temp[:, k]
    water = {f"sm_{d}cm": theta[:, k] for k, d in enumerate(DEPTHS_CM)}
    return truth, water


# ----------------------------------------------------------------------------
# 2. Observation and planted problems
# ----------------------------------------------------------------------------
@dataclass
class Planted:
    kind: str
    columns: list
    ranges: list                       # list of (start, end) Timestamps
    note: str = ""


@dataclass
class Twin:
    netid: str
    seed: int
    truth: pd.DataFrame
    observed: pd.DataFrame
    _planted: list = field(repr=False, default_factory=list)

    @property
    def n_planted(self):
        return len(self._planted)

    def write_csv(self, path):
        """Write the observed table as a colleague would hand it over: one row per day, blank = missing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.observed.to_csv(path, index=False)
        return path


def _observe(truth, rng):
    obs = truth.copy()
    for c in SM:
        obs[c] = (obs[c] + rng.normal(0, 0.002, len(obs))).round(3)
    for c in ST:
        obs[c] = (obs[c] + rng.normal(0, 0.1, len(obs))).round(1)
    for c in ("t_max_c", "t_min_c", "t_mean_c"):
        obs[c] = obs[c].round(1)
    obs["t_mean_c"] = ((obs["t_max_c"] + obs["t_min_c"]) / 2).round(1)   # USCRN T_DAILY_MEAN definition
    obs["precip_mm"] = obs["precip_mm"].round(1)
    return obs


def _warm(idx):
    return np.isin(idx.month, range(4, 11))


def _free(occupied, a, b, pad=10):
    return not any(a - pad <= y and x - pad <= b for x, y in occupied)


def _pick_window(rng, n, length, ok, occupied, tries=400):
    """Start index of a window of `length` days where ok(start) is true and nothing else is planted."""
    for _ in range(tries):
        s = int(rng.integers(30, n - length - 30))
        if ok(s) and _free(occupied, s, s + length - 1):
            occupied.append((s, s + length - 1))
            return s
    raise RuntimeError("could not place a planted problem; try another seed")


def plant_problems(obs, truth, water, seed, n_planted=N_PLANTED, kinds=None):
    """Return (observed table with problems, list of Planted). Row order is by date; duplicates are inserted last."""
    rng = np.random.default_rng(seed)
    drawn = [str(k) for k in rng.choice(list(PROBLEM_TYPES), size=n_planted, replace=False)]
    kinds = drawn if kinds is None else list(kinds)
    df = obs.copy()
    idx = df.index
    n = len(df)
    p_true = truth["precip_mm"].values
    occupied = []
    planted = []
    day = lambda i: idx[int(i)]                                      # noqa: E731
    # rain_one_day_late is planted only in a year where the shift leaves a trace: shifted rain on day t must
    # correlate with the true 5 cm change on day t - 1, on days when the 5 cm soil is above 0 C. In a year whose
    # few large same-day wettings fall next to frozen days, the shift is close to invisible.
    dsm = pd.Series(truth["sm_5cm"].values, index=idx).diff().shift(1)
    rain = pd.Series(p_true, index=idx).shift(1)
    thawed = pd.Series(truth["st_5cm"].values > 0, index=idx)
    same_day_r = {y: float(rain[(idx.year == y) & thawed].corr(dsm[(idx.year == y) & thawed]))
                  for y in range(2016, 2024)}

    for kind in kinds:
        if kind == "spike":
            # electrical interference: half of the spikes fall on thunderstorm days, when they mimic wetting
            col = str(rng.choice(["sm_5cm", "sm_10cm"]))
            days = []
            while len(days) < 4:
                stormy = len(days) % 2 == 0

                def ok(s, stormy=stormy):
                    rain = p_true[s] >= 2 if stormy else p_true[s - 1:s + 2].sum() == 0
                    return (_warm(idx[[s]])[0] and rain and np.isfinite(df[col].values[s - 1:s + 2]).all())
                s = _pick_window(rng, n, 1, ok, occupied)
                df.iloc[s, df.columns.get_loc(col)] = round(df[col].iat[s] + rng.uniform(0.08, 0.15), 3)
                days.append(s)
            planted.append(Planted(kind, [col], [(day(s), day(s)) for s in sorted(days)]))

        elif kind == "drift":
            col = str(rng.choice(["sm_20cm", "sm_50cm", "sm_100cm"]))
            s = int(rng.integers(idx.get_loc(pd.Timestamp("2017-01-01")), idx.get_loc(pd.Timestamp("2021-12-31"))))
            rate = rng.uniform(0.02, 0.035) / 365.25
            ramp = np.clip(np.arange(n) - s, 0, None) * rate
            df[col] = (df[col] + ramp).round(3)
            planted.append(Planted(kind, [col], [(day(s), day(n - 1))], f"{rate * 365.25:.3f} m3/m3 per year"))

        elif kind == "stuck":
            col = "sm_5cm"
            length = int(rng.integers(20, 36))

            def ok(s, length=length):
                w = slice(s, s + length)
                return (_warm(idx[[s, s + length]]).all() and (p_true[w] >= 5).sum() >= 1
                        and np.isfinite(df[col].values[w]).all())
            s = _pick_window(rng, n, length, ok, occupied)
            df.iloc[s:s + length, df.columns.get_loc(col)] = df[col].iat[s]
            planted.append(Planted(kind, [col], [(day(s), day(s + length - 1))]))

        elif kind == "frozen_read_as_dry":
            liquid = water
            ranges = []
            winters = rng.choice(np.arange(2016, 2025), size=3, replace=False)
            for y in sorted(winters):
                w = (idx >= pd.Timestamp(f"{y - 1}-11-01")) & (idx <= pd.Timestamp(f"{y}-03-31"))
                hit = False
                for col, tcol in (("sm_5cm", "st_5cm"), ("sm_10cm", "st_10cm")):
                    frozen = w & (truth[tcol].values < 0)
                    if frozen.sum() == 0:
                        continue
                    frac = rng.uniform(0.12, 0.25)          # unfrozen water left in the pores
                    vals = liquid[col][frozen] * frac + rng.normal(0, 0.004, frozen.sum())
                    df.loc[frozen, col] = np.clip(vals, 0.005, None).round(3)
                    hit = True
                if hit:
                    fz = np.where(w & (truth["st_5cm"].values < 0))[0]
                    if len(fz):
                        ranges.append((day(fz.min()), day(fz.max())))
            planted.append(Planted(kind, ["sm_5cm", "sm_10cm"], ranges))

        elif kind == "missing_code_left_in":
            col = "sm_5cm" if rng.random() < 0.5 else str(rng.choice(SM[1:]))
            length = int(rng.integers(6, 15))
            s = _pick_window(rng, n, length, lambda s, length=length: _warm(idx[[s]])[0]
                             and (p_true[s:s + length] >= 2).any(), occupied)
            df.iloc[s:s + length, df.columns.get_loc(col)] = -99.0
            planted.append(Planted(kind, [col], [(day(s), day(s + length - 1))], "-99.000 is the USCRN code"))

        elif kind == "gauge_outage_as_zero":
            length = int(rng.integers(25, 46))

            def ok(s, length=length):
                w = slice(s, s + length)
                # at least three real storms, and the 5 cm soil visibly wets up at least twice,
                # so the outage leaves a physical trace a student can find
                return ((p_true[w] >= 5).sum() >= 3 and (truth["st_5cm"].values[w] > 0.5).all()
                        and (np.diff(truth["sm_5cm"].values[w]) > 0.03).sum() >= 2)
            s = _pick_window(rng, n, length, ok, occupied)
            df.iloc[s:s + length, df.columns.get_loc("precip_mm")] = 0.0
            planted.append(Planted(kind, ["precip_mm"], [(day(s), day(s + length - 1))]))

        elif kind == "rain_one_day_late":
            y = int(rng.integers(2016, 2024))
            if same_day_r[y] <= 0.3:                       # draw again among the years where it shows
                y = int(rng.choice([k for k, r in same_day_r.items() if r > 0.3]))
            w = (idx.year == y)
            i0, i1 = np.where(w)[0][[0, -1]]
            shifted = df["precip_mm"].shift(1)
            df.loc[w, "precip_mm"] = shifted[w]
            occupied.append((i0, i1))
            planted.append(Planted(kind, ["precip_mm"], [(day(i0), day(i1))], f"calendar year {y}"))

        elif kind == "probes_swapped":
            s = int(rng.integers(idx.get_loc(pd.Timestamp("2018-01-01")), idx.get_loc(pd.Timestamp("2022-12-31"))))
            s = int(np.where((idx.month[s:] >= 5) & (idx.month[s:] <= 9))[0][0] + s)   # visits happen in summer
            for a, b in (("sm_10cm", "sm_20cm"), ("st_10cm", "st_20cm")):
                va, vb = df[a].values.copy(), df[b].values.copy()
                df.iloc[s:, df.columns.get_loc(a)] = vb[s:]
                df.iloc[s:, df.columns.get_loc(b)] = va[s:]
            # problems planted earlier in 10 or 20 cm now sit, from the swap on, in the other column
            other = {"sm_10cm": "sm_20cm", "sm_20cm": "sm_10cm"}
            for q in planted:
                if any(y >= day(s) for _, y in q.ranges):
                    q.columns = sorted(set(q.columns) | {other.get(c, c) for c in q.columns})
            planted.append(Planted(kind, ["sm_10cm", "sm_20cm", "st_10cm", "st_20cm"], [(day(s), day(n - 1))]))

        elif kind == "units_percent":
            length = int(rng.integers(45, 120))
            s = _pick_window(rng, n, length, lambda s: True, occupied)
            for c in SM:
                v = df[c].values[s:s + length]
                df.iloc[s:s + length, df.columns.get_loc(c)] = np.where(v > -50, (v * 100).round(1), v)
            planted.append(Planted(kind, list(SM), [(day(s), day(s + length - 1))]))

        elif kind == "duplicated_days":
            pass                                               # applied last, after all values are set

    df = df.reset_index()
    if "duplicated_days" in kinds:
        length = int(rng.integers(4, 9))
        s = _pick_window(rng, n, length, lambda s: True, occupied)
        block = df.iloc[s:s + length]
        df = pd.concat([df.iloc[:s + length], block, df.iloc[s + length:]], ignore_index=True)
        planted.append(Planted("duplicated_days", ["date"], [(day(s), day(s + length - 1))]))
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[COLUMNS], planted


def make_twin(netid="", real_path=REAL_FILE, n_planted=N_PLANTED):
    """Truth (same for all) and an observed table with problems placed from your NetID.

    The seed comes from the NetID. When the problems it draws move the 5 cm response to rain by less than
    MIN_BIAS from the truth, the next seed is tried, so that Part 5 always has a damage to measure.
    Twin.seed is the seed actually used.
    """
    seed0 = seed_from_netid(netid)
    truth, water = make_truth(TRUTH_SEED, real_path)
    obs = _observe(truth, np.random.default_rng(TRUTH_SEED + 1))
    truth_out = truth.reset_index()
    truth_out["date"] = truth_out["date"].dt.strftime("%Y-%m-%d")
    truth_out = truth_out[COLUMNS]
    r_true = rain_response(truth_out)[0]
    for k in range(50):
        try:
            observed, planted = plant_problems(obs, truth, water, seed0 + k, n_planted)
        except RuntimeError:
            if k == 0:
                raise
            continue
        r_obs = rain_response(observed)[0]
        if np.isfinite(r_obs) and abs(r_obs / r_true - 1) >= MIN_BIAS:
            break
    return Twin(netid=netid, seed=seed0 + k, truth=truth_out, observed=observed, _planted=planted)


# ----------------------------------------------------------------------------
# 3. Derived quantities
# ----------------------------------------------------------------------------
def rain_response(table, col="sm_5cm", rain_col="precip_mm", min_rain=2.0, temp_col="st_5cm"):
    """Soil-moisture response to rain: the least-squares slope of the day-to-day change at `col`
    against the day's rain, in m3/m3 per 10 mm.

    The table is used in the order its rows come, as most quick analyses do: the change on a row
    is that row minus the row before. Days enter when they have at least `min_rain` mm and the soil
    at 5 cm is above 0 C on that row and on the row before (a frozen probe does not see water).

    Returns (slope per 10 mm, table of the days used).
    """
    t = table.reset_index(drop=True)
    s = pd.to_numeric(t[col], errors="coerce")
    p = pd.to_numeric(t[rain_col], errors="coerce")
    warm_soil = (t[temp_col] > 0) & (t[temp_col].shift(1) > 0)
    change = s - s.shift(1)
    use = (p >= min_rain) & warm_soil & change.notna()
    days = pd.DataFrame({"date": t["date"][use], "rain_mm": p[use], "change": change[use]})
    if use.sum() < 3:
        return np.nan, days
    slope = np.polyfit(days["rain_mm"].values, days["change"].values, 1)[0]
    return float(slope * 10), days


def drydown_timescale(table, col="sm_5cm", rain_col="precip_mm", rain_start=5.0, dry_day=1.0,
                      min_days=5, max_days=20, months=range(4, 11)):
    """Median e-folding time (days) of soil drying after warm-season storms.

    A storm is a day with at least `rain_start` mm in April to October. The drydown starts
    on the wettest day among the next `max_days` days that have less than `dry_day` mm of rain,
    and lasts until the next day with `dry_day` mm or more. Over the drydown, the excess
    over the record's dry floor (2nd percentile of warm-season values) is fitted with an
    exponential, ln(theta - floor) = a - t / tau. Drydowns with a blank day, or shorter than
    `min_days`, are skipped.

    Returns (median tau in days, table of drydowns).
    """
    t = table.copy()
    t["date"] = pd.to_datetime(t["date"])
    s = t[col].astype(float).values
    p = t[rain_col].astype(float).values
    month = t["date"].dt.month.values
    warm = np.isin(month, list(months))
    floor = np.nanquantile(s[warm], 0.02)
    rows = []
    for i in np.where((p >= rain_start) & warm)[0]:
        seg = []
        j = i + 1
        while j < len(s) and j <= i + max_days and p[j] < dry_day:
            seg.append(j)
            j += 1
        if len(seg) < min_days:
            continue
        y = s[seg]
        if np.isnan(y).any():
            continue
        k = int(np.argmax(y))
        y = y[k:]
        z = y - floor
        ok = z > 0.005
        if len(y) < min_days or ok.sum() < min_days:
            continue
        slope = np.polyfit(np.arange(len(y))[ok], np.log(z[ok]), 1)[0]
        tau = -1.0 / slope if slope < 0 else np.inf
        rows.append({"storm_day": t["date"].iat[i].date(), "start": t["date"].iat[seg[k]].date(),
                     "n_days": len(y), "theta_start": round(float(y[0]), 3), "tau_days": tau})
    events = pd.DataFrame(rows, columns=["storm_day", "start", "n_days", "theta_start", "tau_days"])
    med = float(np.median(events["tau_days"])) if len(events) else np.nan
    return med, events


# ----------------------------------------------------------------------------
# 4. Scoring findings
# ----------------------------------------------------------------------------
def _parse_finding(f):
    kind = str(f.get("problem", "")).strip()
    col = str(f.get("column", "any")).strip()
    try:
        a = pd.Timestamp(f.get("start"))
        b = pd.Timestamp(f.get("end", f.get("start")))
    except (ValueError, TypeError):
        return kind, col, None, None
    return kind, col, min(a, b), max(a, b)


def _matches(f, p, tol_days=3):
    kind, col, a, b = _parse_finding(f)
    if kind != p.kind or a is None or pd.isna(a):
        return False
    if col not in ("any", "all", "") and col not in p.columns and p.kind != "duplicated_days":
        return False
    tol = pd.Timedelta(days=tol_days)
    # a finding must also be about as short as the problem: a span over the whole record matches nothing
    if (b - a) > 3 * max(y - x for x, y in p.ranges) + pd.Timedelta(days=30):
        return False
    return any(a - tol <= y and x - tol <= b for x, y in p.ranges)


def score_findings(findings, twin):
    """(number of planted problems matched, list of unmatched findings, list of missed kinds)."""
    findings = [f for f in (findings or []) if isinstance(f, dict)]
    matched = [p for p in twin._planted if any(_matches(f, p) for f in findings)]
    extra = [f for f in findings if not any(_matches(f, p) for p in twin._planted)]
    missed = [p.kind for p in twin._planted if p not in matched]
    return len(matched), extra, missed


def check_findings(findings, twin, show_missed=False, who="You", offer_missed=True):
    """Print how many planted problems the findings match; with show_missed, which kinds were missed."""
    findings = [f for f in (findings or []) if isinstance(f, dict)]
    if not findings:
        print("not answered yet: add at least one finding to the list")
        return None
    bad = [f for f in findings if str(f.get("problem", "")).strip() not in PROBLEM_TYPES]
    if bad:
        print("These findings use a problem name that is not on the list; use one of:")
        print("   " + ", ".join(PROBLEM_TYPES))
        for f in bad:
            print("   ", f)
    no_why = [f for f in findings if len(str(f.get("why", "")).strip()) < 15]
    n_found, extra, missed = score_findings(findings, twin)
    print(f"{who} recorded {len(findings)} findings. They match {n_found} of the {twin.n_planted} planted problems.")
    if extra:
        print(f"{len(extra)} finding(s) match no planted problem: a wrong name, column or date, a span much "
              "longer than the problem, or something the soil really does: check rain, temperature and the "
              "other depths on those dates.")
    if no_why:
        print(f"{len(no_why)} finding(s) have no physical reason in 'why'. Say why the value is impossible for soil.")
    if n_found < twin.n_planted and not show_missed and offer_missed:
        print("Set show_missed=True to see which kinds you missed (not where).")
    if show_missed and missed:
        print("Kinds you have not found yet: " + ", ".join(sorted(missed)))


if __name__ == "__main__":
    import time
    t0 = time.time()
    tw = make_twin("")
    print(f"built in {time.time() - t0:.2f} s; planted: {[p.kind for p in tw._planted]}")
    for name, tab in (("truth", tw.truth), ("observed", tw.observed)):
        r, used = rain_response(tab)
        tau, ev = drydown_timescale(tab)
        print(f"{name}: response {r:.4f} m3/m3 per 10 mm on {len(used)} days; drydown {tau:.1f} days ({len(ev)})")
