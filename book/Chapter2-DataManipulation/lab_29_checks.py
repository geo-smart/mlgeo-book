"""Self-checks and small helpers for book section 2.9, filtering and repairing records (UW.RATT, Chignik 2021).

Each "Your turn" cell ends with `your_turn(key, globals())`. It prints `correct`, a hint that says where to
look (never the answer), or `not answered yet`. Unless the answer is correct, it puts the reference result in
place of each name so that the cells after it still run, keeps a wrong value as `<name>_yours`, and says so in
one line.
The reference results are computed here from the notebook's own record, never typed in.
Nothing here downloads anything, except `real_gap_at_ratt`, which the notebook calls only when online.
Run from book/Chapter2-DataManipulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

SNAP_MAP = "snapshots/naturalearth_pnw_map.csv"
SNAP_COAST = "snapshots/naturalearth_npacific_coast.csv"   # Alaska Peninsula to Washington, 1:50m
BAND = (1.0, 10.0)        # Hz, the band of the lecture figures
POLES = 2                 # Butterworth order passed to butter (a band-pass doubles it)
GAP = (850.0, 870.0)      # s after 06:15:00 UTC, the planted gap
SHIFT_S = 0.5             # s, the planted clock error
P_WINDOW = (355.0, 415.0)  # s, the correlation window on the P wave train

HINTS = {
    "1.step": "np.diff(raw) is largest where the surface waves are steepest, not at a step. A step is where the "
              "slope changes most abruptly: look at the second difference after 700 s (np.diff(raw, n=2) of "
              "samples i, i+1, i+2 belongs to sample i+1).",
    "1.time": "t_step should be a time in seconds after 06:15:00, between 700 and 7200, not a sample index: "
              "index the time axis t with the position you found.",
    "1.other": "look for the largest absolute second difference after 700 s, and take its time from t.",
    "2.sos": "butter takes the order, the two corners in Hz, the band type, fs=fs and output='sos'.",
    "2.zp": "the zero-phase result comes from signal.sosfiltfilt(sos, z), forward then backward.",
    "2.causal": "the causal result comes from signal.sosfilt(sos, z), forward only.",
    "2.lead": "both are times relative to t_step, so subtract t_step. lead_zp uses the window t_step - 1 to "
              "t_step + 0.5 s and the threshold on np.abs(zf); delay_c uses the 0.5 s after the step and np.abs(zc).",
    "2.zpeak": "delay_c is the peak of the zero-phase output; use the causal output zc.",
    "3.grade": "keep the samples 0 to 2 s outside each edge (before gap[0] and from gap[1] on), take the RMS of the "
               "intact filtered record zf there as the signal, and the RMS of the difference from zf for each method; "
               "divide each error by the signal (np.sqrt, np.mean).",
    "4.shift": "a later copy has each sample at a larger index; the shift in samples is an integer; filter the "
               "copy the same way as zf.",
    "3.seg": "start from an array of NaN of the same length, find the indices that are not NaN with np.isnan and "
             "np.flatnonzero, cut them where consecutive indices jump by more than one (np.split), and filter each "
             "piece of z_nan with signal.sosfiltfilt.",
    "3.nan": "the gap should stay NaN in zf_seg: fill only the indices of the valid segments.",
    "4.lag": "cc and lags must come from the same order of b and a and the same mode; lags in seconds, so divide "
             "by fs; lag_peak is the lag where cc is largest.",
    "4.sign": "the sign is reversed: put the late copy first in signal.correlate and in correlation_lags.",
    "4.sub": "check the sign and the denominator of the correction against the formula above it; the correction "
             "is in samples, so divide by fs, and add it to lag_peak.",
}

_cache: dict = {}


# ----------------------------------------------------------------------------- map
def draw_map(prov: dict, ev: dict):
    """Station and epicentre on longitude-latitude axes, with the great-circle path between them."""
    import matplotlib.pyplot as plt

    sta = prov["station"]
    lat1, lon1, lat2, lon2 = np.radians([sta["latitude"], sta["longitude"], ev["latitude"], ev["longitude"]])
    p1 = np.array([np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)])
    p2 = np.array([np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)])
    om = np.arccos(p1 @ p2)
    f = np.linspace(0, 1, 100)[:, None]
    p = (np.sin((1 - f) * om) * p1 + np.sin(f * om) * p2) / np.sin(om)
    gc_lat, gc_lon = np.degrees(np.arcsin(p[:, 2])), np.degrees(np.arctan2(p[:, 1], p[:, 0]))

    ne = pd.read_csv(SNAP_MAP)
    coast = pd.read_csv(SNAP_COAST)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.7, 1]})
    for _, g in coast.groupby("part"):
        a1.plot(g["lon"], g["lat"], color="0.3", lw=0.6)
    for _, g in ne[ne["kind"] == "border"].groupby("part"):
        a1.plot(g["lon"], g["lat"], color="0.65", lw=0.7)
    a1.plot(gc_lon, gc_lat, color="0.4", ls="--", lw=1)
    a1.annotate("Alaska Peninsula", (-156.5, 56.6), xytext=(-151, 57.2), fontsize=8, color="0.35", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    for (kind, part), g in ne[ne["kind"] != "place"].groupby(["kind", "part"]):
        a2.plot(g["lon"], g["lat"], color="0.3" if kind == "coastline" else "0.65", lw=0.7)
    for ax in (a1, a2):
        ax.plot(sta["longitude"], sta["latitude"], "^", color="C0", ms=10, mec="k")
    a1.plot(ev["longitude"], ev["latitude"], "*", color="C3", ms=16, mec="k")
    a1.annotate(f"Chignik {ev['magnitude_type'].capitalize()} {ev['magnitude']}\n{ev['depth_km']} km deep",
                (ev["longitude"], ev["latitude"]), xytext=(8, -26), textcoords="offset points", fontsize=9)
    a1.annotate("UW.RATT", (sta["longitude"], sta["latitude"]), xytext=(-60, 8), textcoords="offset points",
                fontsize=9)
    a1.set(xlim=(-162, -116), ylim=(44, 60), xlabel="longitude (°)", ylabel="latitude (°)",
           title="Great-circle path")
    a1.set_aspect(1 / np.cos(np.radians(52)))
    for _, pl in ne[ne["kind"] == "place"].iterrows():
        a2.plot(pl["lon"], pl["lat"], "s", color="0.2", ms=4)
        a2.annotate(pl["name"], (pl["lon"], pl["lat"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    a2.annotate("RATT", (sta["longitude"], sta["latitude"]), xytext=(6, -12), textcoords="offset points", fontsize=9)
    a2.set(xlim=(-125, -116.5), ylim=(45, 49.5), xlabel="longitude (°)", title="Washington")
    a2.set_aspect(1 / np.cos(np.radians(47.4)))
    fig.text(0.01, 0.0, "Coast and borders: Natural Earth 1:50m (course copies). "
             "Station: EarthScope StationXML. Epicentre: USGS ComCat.", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ----------------------------------------------------------------------------- figures of the notebook
def plot_record(t, raw, T_P, T_S):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, raw / 1e6, color="k", lw=0.5)
    for tt, name in [(T_P, "P"), (T_S, "S")]:
        ax.axvline(tt, color="C0", lw=1.5)
        ax.text(tt + 6, 1.5, name, color="C0", fontweight="bold")
    ax.set(xlim=(300, 1000), xlabel="seconds after 06:15:00 UTC", ylabel="10⁶ counts",
           title="UW.RATT HHZ, raw counts")
    return fig


def plot_step(t, raw, t_step, fs):
    import matplotlib.pyplot as plt
    k = int(round(t_step * fs))
    w = slice(k - 60, k + 60)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.2))
    ax[0].plot(t[w], raw[w] / 1e3, ".-", color="k", lw=0.8, ms=3)
    ax[0].set(ylabel="10³ raw counts", title="raw record: a steep swing")
    ax[1].plot(t[w][1:], np.diff(raw[w]), ".-", color="k", lw=0.8, ms=3)
    ax[1].set(ylabel="counts per sample", title="first difference: the step")
    for a in ax:
        a.axvline(t_step, color="C3", lw=1)
        a.set_xlabel("seconds after 06:15:00 UTC")
    fig.tight_layout()
    return fig


def plot_filters(t, zf, zc, t_step, T_P):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, (t0, x0, x1, name) in zip(ax, [(t_step, 0.9, 0.9, f"step at {t_step:.2f} s"),
                                          (T_P, 6, 10, f"iasp91 P, {T_P:.1f} s")]):
        a.plot(t, zf, color="C0", lw=1.5, label="zero-phase")
        a.plot(t, zc, color="C1", lw=1.5, alpha=0.9, label="causal")
        a.axvline(t0, color="k", lw=1)
        w = (t > t0 - x0) & (t < t0 + x1)
        top = 1.1 * max(np.abs(zf[w]).max(), np.abs(zc[w]).max())
        a.set(xlim=(t0 - x0, t0 + x1), ylim=(-top, top), xlabel="seconds after 06:15:00 UTC", ylabel="counts",
              title=name)
    ax[0].legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_gap(t, zf, zf_gap, zf_seg, gap):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t, zf, color="0.5", lw=1.2, label="intact record, filtered")
    ax[0].plot(t, zf_gap, color="C3", lw=1, label="filtered across the zero-filled gap")
    ax[1].plot(t, zf, color="0.5", lw=1.2, label="intact record, filtered")
    ax[1].plot(t, zf_seg, color="C0", lw=1, label="each segment filtered alone")
    ax[1].set(ylim=(-400, 400), xlabel="seconds after 06:15:00 UTC")
    for a in ax:
        a.axvspan(*gap, color="k", alpha=0.08)
        a.set(xlim=(gap[0] - 15, gap[1] + 15), ylabel="counts")
        a.legend(loc="upper left")
    return fig


def plot_correlation(lags, cc, lag_peak, window):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(lags, cc / cc.max(), color="C0")
    ax.axvline(lag_peak, color="C3", ls="--")
    ax.set(xlim=(-2, 2), xlabel="lag (s)", ylabel="normalized",
           title=f"Correlation of {window[0]:.0f}-{window[1]:.0f} s, late copy against reference")
    return fig


# ----------------------------------------------------------------------------- reference results
def _ref(ns: dict) -> dict:
    """Reference results from the notebook's record (raw, z, t, fs), computed once."""
    key = (id(ns.get("z")), len(ns.get("z")))
    if _cache.get("key") == key:
        return _cache["ref"]
    raw, z, t, fs = ns["raw"], ns["z"], ns["t"], ns["fs"]
    r = {}
    d2 = np.abs(np.diff(raw, 2))
    after = t[1:-1] > 700
    r["t_step"] = float(t[1:-1][after][np.argmax(d2[after])])
    r["sos"] = signal.butter(POLES, list(BAND), "bandpass", fs=fs, output="sos")
    r["zf"] = signal.sosfiltfilt(r["sos"], z)
    r["zc"] = signal.sosfilt(r["sos"], z)
    w = (t >= r["t_step"] - 1) & (t < r["t_step"] + 0.5)
    a_ = np.abs(r["zf"][w])
    r["lead_zp"] = float(t[w][np.argmax(a_ > 0.1 * a_.max())] - r["t_step"])
    r["delay_c"] = peak_after(t, r["zc"], r["t_step"])
    r["peak_zp"] = peak_after(t, r["zf"], r["t_step"])
    d1 = np.abs(np.diff(raw))
    r["t_d1"] = float(t[1:][t[1:] > 700][np.argmax(d1[t[1:] > 700])])
    i0, i1 = int(GAP[0] * fs), int(GAP[1] * fs)
    z_gap = z.copy()
    z_gap[i0:i1] = 0.0
    z_nan = z.copy()
    z_nan[i0:i1] = np.nan
    r["zf_seg"] = filter_segments(z_nan, r["sos"])
    e_n, sig = edge_error(signal.sosfiltfilt(r["sos"], z_gap), r["zf"], t, GAP)
    e_s, _ = edge_error(r["zf_seg"], r["zf"], t, GAP)
    r["ratio_naive"], r["ratio_seg"] = e_n / sig, e_s / sig
    r["z_late"] = np.roll(z, int(round(SHIFT_S * fs)))
    r["zf_late"] = late = signal.sosfiltfilt(r["sos"], r["z_late"])
    w0, w1 = int(P_WINDOW[0] * fs), int(P_WINDOW[1] * fs)
    a, b = r["zf"][w0:w1], late[w0:w1]
    r["cc"] = signal.correlate(b, a, mode="full")
    r["lags"] = signal.correlation_lags(len(b), len(a), mode="full") / fs
    j = int(np.argmax(r["cc"]))
    r["lag_peak"] = float(r["lags"][j])
    y0, y1, y2 = r["cc"][j - 1], r["cc"][j], r["cc"][j + 1]
    r["lag_sub"] = float(r["lag_peak"] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) / fs)
    _cache.update(key=key, ref=r)
    return r


def filter_segments(x, sos):
    """Zero-phase filter each run of valid (not NaN) samples on its own; the gap stays NaN."""
    out = np.full(len(x), np.nan)
    good = np.flatnonzero(~np.isnan(x))
    for seg in np.split(good, np.flatnonzero(np.diff(good) > 1) + 1):
        out[seg] = signal.sosfiltfilt(sos, x[seg])
    return out


# ----------------------------------------------------------------------------- measurements used by the notebook
def first_above(t, y, t0, before=1.0, after=0.5, frac=0.1):
    """Time, relative to t0, of the first sample in [t0 - before, t0 + after) above frac of the window's peak."""
    w = (t >= t0 - before) & (t < t0 + after)
    a = np.abs(y[w])
    return float(t[w][np.argmax(a > frac * a.max())] - t0)


def peak_after(t, y, t0, after=0.5):
    """Time, relative to t0, of the largest absolute value in [t0, t0 + after)."""
    w = (t >= t0) & (t < t0 + after)
    return float(t[w][np.argmax(np.abs(y[w]))] - t0)


def onset_above(t, y, noise=(300.0, 350.0), k=5.0, start=350.0):
    """First time after `start` where |y| exceeds k times the standard deviation of y over the `noise` window."""
    sd = float(np.std(y[(t >= noise[0]) & (t < noise[1])]))
    i = np.flatnonzero((t >= start) & (np.abs(y) > k * sd))
    return (float(t[i[0]]) if len(i) else float("nan")), sd


def filter_delays(sos, fs, freqs):
    """Phase delay and group delay (s) of a causal filter at the given frequencies; negative means earlier."""
    w, h = signal.sosfreqz(sos, worN=np.asarray(freqs, float), fs=fs)
    phase = -np.unwrap(np.angle(h)) / (2 * np.pi * w)
    b, a = signal.sos2tf(sos)
    _, gd = signal.group_delay((b, a), w=np.asarray(freqs, float), fs=fs)
    return phase, gd / fs


def edge_error(test, ref, t, gap, width=2.0):
    """RMS of (test - ref) over 0 to `width` s outside each gap edge, and the RMS of ref there (the local signal)."""
    near = ((t >= gap[0] - width) & (t < gap[0])) | ((t >= gap[1]) & (t < gap[1] + width))
    err = np.sqrt(np.mean((test[near] - ref[near]) ** 2))
    sig = np.sqrt(np.mean(ref[near] ** 2))
    return float(err), float(sig)


def guard_interval(test, ref, t, gap, frac=0.01):
    """Seconds before and after the gap over which |test - ref| exceeds frac times the background RMS
    (the RMS of ref 5 to 60 s from the gap). Returns (before, after, background)."""
    dist = np.where(t < gap[0], gap[0] - t, t - gap[1])
    out = ~np.isnan(test)
    local = np.sqrt(np.mean(ref[out & (dist > 5) & (dist < 60)] ** 2))
    bad = out & (np.abs(test - ref) > frac * local) & (dist < 200)
    before = float(dist[bad & (t < gap[0])].max()) if (bad & (t < gap[0])).any() else 0.0
    after = float(dist[bad & (t > gap[1])].max()) if (bad & (t > gap[1])).any() else 0.0
    return before, after, float(local)


# ----------------------------------------------------------------------------- the checks
def _arr(v):
    try:
        return np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None


def _close_arr(v, ref, rtol=1e-6):
    a = _arr(v)
    if a is None or a.shape != np.shape(ref):
        return False
    m = ~np.isnan(ref)
    return bool(np.array_equal(np.isnan(a), ~m)) and bool(np.allclose(a[m], ref[m], rtol=rtol,
                                                                      atol=1e-6 * np.nanmax(np.abs(ref))))


def _shape_ok(v, ref):
    if np.isscalar(ref):
        return isinstance(v, (int, float, np.integer, np.floating))
    a = _arr(v)
    return a is not None and a.shape == np.shape(ref)


TURNS = {
    "1.step": ["t_step"],
    "2.filter": ["sos", "zf", "zc"],
    "2.lead": ["lead_zp", "delay_c"],
    "3.segments": ["zf_seg"],
    "3.grade": ["ratio_naive", "ratio_seg"],
    "4.shift": ["z_late", "zf_late"],
    "4.lag": ["cc", "lags", "lag_peak"],
    "4.sub": ["lag_sub"],
}


def _verdict(key, ns, r):
    v = {n: ns.get(n) for n in TURNS[key]}
    if all(x is None for x in v.values()):
        return "not answered yet"
    if key == "1.step":
        ts = v["t_step"]
        if not _shape_ok(ts, r["t_step"]) or isinstance(ts, (int, np.integer)) or not 700 <= ts <= 7200:
            return "not yet. Hint: " + HINTS["1.time"]
        if abs(ts - r["t_step"]) <= 0.015:
            return "correct"
        return "not yet. Hint: " + HINTS["1.step" if abs(ts - r["t_d1"]) <= 0.015 else "1.other"]
    if key == "2.filter":
        if not _close_arr(v["sos"], r["sos"]):
            return "not yet. Hint: " + HINTS["2.sos"]
        if _close_arr(v["zf"], r["zc"]) and _close_arr(v["zc"], r["zf"]):
            return "not yet. Hint: zf and zc are swapped; zf is the zero-phase result, zc the causal one."
        if not _close_arr(v["zf"], r["zf"]):
            return "not yet. Hint: " + HINTS["2.zp"]
        if not _close_arr(v["zc"], r["zc"]):
            return "not yet. Hint: " + HINTS["2.causal"]
        return "correct"
    if key == "3.segments":
        a = _arr(v["zf_seg"])
        if a is None or a.shape != r["zf_seg"].shape:
            return "not yet. Hint: zf_seg should be one array as long as z_nan. " + HINTS["3.seg"]
        if not np.array_equal(np.isnan(a), np.isnan(r["zf_seg"])):
            return "not yet. Hint: " + HINTS["3.nan"]
        return "correct" if _close_arr(a, r["zf_seg"], rtol=1e-4) else "not yet. Hint: " + HINTS["3.seg"]
    if key == "2.lead":
        if not (_shape_ok(v["lead_zp"], 0.0) and _shape_ok(v["delay_c"], 0.0)):
            return "not yet. Hint: lead_zp and delay_c should be two numbers in seconds. " + HINTS["2.lead"]
        if abs(v["lead_zp"] + r["lead_zp"]) < 0.015 and abs(v["lead_zp"]) > 0.1:
            return "not yet. Hint: the sign of lead_zp: subtract t_step from the time you found."
        if abs(v["delay_c"] - r["peak_zp"]) < 0.005 and abs(v["delay_c"] - r["delay_c"]) >= 0.005:
            return "not yet. Hint: " + HINTS["2.zpeak"]
        ok = abs(v["lead_zp"] - r["lead_zp"]) < 0.015 and abs(v["delay_c"] - r["delay_c"]) < 0.005
        return "correct" if ok else "not yet. Hint: " + HINTS["2.lead"]
    if key == "3.grade":
        if not (_shape_ok(v["ratio_naive"], 0.0) and _shape_ok(v["ratio_seg"], 0.0)):
            return "not yet. Hint: ratio_naive and ratio_seg should be two numbers. " + HINTS["3.grade"]
        ok = abs(v["ratio_naive"] / r["ratio_naive"] - 1) < 0.02 and abs(v["ratio_seg"] / r["ratio_seg"] - 1) < 0.02
        return "correct" if ok else "not yet. Hint: " + HINTS["3.grade"]
    if key == "4.shift":
        if not _close_arr(v["z_late"], r["z_late"]):
            return "not yet. Hint: " + HINTS["4.shift"]
        return "correct" if _close_arr(v["zf_late"], r["zf_late"]) else "not yet. Hint: " + HINTS["4.shift"]
    if key == "4.lag":
        lp = v["lag_peak"]
        if not _shape_ok(lp, r["lag_peak"]):
            return "not yet. Hint: lag_peak should be one number in seconds. " + HINTS["4.lag"]
        if abs(lp + r["lag_peak"]) < 0.011:
            return "not yet. Hint: " + HINTS["4.sign"]
        return "correct" if abs(lp - r["lag_peak"]) < 0.011 else "not yet. Hint: " + HINTS["4.lag"]
    if key == "4.sub":
        ls = v["lag_sub"]
        if not _shape_ok(ls, r["lag_sub"]):
            return "not yet. Hint: lag_sub should be one number in seconds. " + HINTS["4.sub"]
        if abs(ls - r["lag_peak"]) < 1e-9:
            return "not yet. Hint: that is the peak sample itself; " + HINTS["4.sub"]
        # grade the parabola correction itself: with an integer shift it is tiny, so the value alone would pass
        # a flipped sign or a reversed denominator
        dc, dr = ls - r["lag_peak"], r["lag_sub"] - r["lag_peak"]
        return "correct" if abs(dc - dr) < 0.1 * abs(dr) + 1e-7 else "not yet. Hint: " + HINTS["4.sub"]
    raise KeyError(key)


def your_turn(key: str, ns: dict) -> None:
    """Check a Your-turn cell; unless it is correct, put the reference results in ns and keep the student's."""
    r = _ref(ns)
    verdict = _verdict(key, ns, r)
    print(f"[{key}] {verdict}")
    if verdict == "correct":
        return
    kept = []
    for n in TURNS[key]:
        if ns.get(n) is not None:
            ns[n + "_yours"] = ns[n]
            kept.append(n + "_yours")
        ns[n] = r[n]
    note = f"; yours kept as {', '.join(kept)}" if kept else ""
    print(f"[{', '.join(TURNS[key])}] using the reference result so the next cells run{note}.")


def check_text(qid: str, text) -> None:
    """Written answers are recorded, not graded."""
    t = " ".join(str(text or "").split())
    print(f"[{qid}] " + (f"recorded ({len(t.split())} words)" if t else "not answered yet"))


# ----------------------------------------------------------------------------- optional: a real gap
def real_gap_at_ratt(pad=600.0):
    """Fetch the 40 s telemetry gap at UW.RATT HHZ on 2021-01-21 17:48 UTC from EarthScope and grade
    zero-filling against segment-by-segment filtering, as in the lecture 8 figure script lec08_real_seismic_gap.py
    (4-pole zero-phase 1-10 Hz band-pass from obspy; segments demeaned and tapered over 1 s)."""
    import matplotlib.pyplot as plt
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client

    g_guess = UTCDateTime("2021-01-21T17:48:03")
    st = Client("EARTHSCOPE", timeout=60).get_waveforms("UW", "RATT", "--", "HHZ", g_guess - pad, g_guess + 40 + pad)
    g = max(st.get_gaps(), key=lambda x: x[6])
    g0, gdur = g[4], g[6]
    fs = st[0].stats.sampling_rate
    zero = st.copy().merge(fill_value=0)[0]
    tz = zero.times() + (zero.stats.starttime - g0)
    dist = np.where(tz < 0, -tz, tz - gdur)
    zero.detrend("demean")
    zero.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True)
    seg = np.full(zero.stats.npts, np.nan)
    for tr in st.copy():
        tr.detrend("demean"); tr.taper(max_percentage=None, max_length=1.0)
        tr.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True)
        i0 = int(round((tr.stats.starttime - zero.stats.starttime) * fs))
        seg[i0:i0 + tr.stats.npts] = tr.data
    out = ~np.isnan(seg)
    local = float(np.sqrt(np.mean(seg[out & (dist > 60) & (dist < 300)] ** 2)))
    diff = np.abs(zero.data - seg)
    peak = float(np.nanmax(diff[out & (dist < 5)]))
    bad = out & (diff > local) & (dist < 120)
    before = float(-tz[bad & (tz < 0)].min()) if (bad & (tz < 0)).any() else 0.0
    after = float(tz[bad & (tz > gdur)].max() - gdur) if (bad & (tz > gdur)).any() else 0.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(tz, zero.data, color="C1", lw=0.8, label="merged with zeros, then filtered")
    ax.plot(tz, seg, color="C0", lw=0.8, label="each segment filtered alone")
    ax.axvspan(0, gdur, color="k", alpha=0.08)
    ax.set(xlim=(-20, gdur + 20), xlabel=f"seconds after {g0.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]} UTC",
           ylabel="counts", title="UW.RATT HHZ, real telemetry gap, 1-10 Hz")
    ax.legend(loc="upper right", fontsize=8)
    print(f"gap: {g0.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]} UTC, {gdur:.2f} s; local 1-10 Hz signal {local:.1f} counts RMS (60 to 300 s from the gap)")
    print(f"peak edge error of the zero-filled record {peak:.0f} counts, {peak / local:.0f} times the local signal; "
          f"above the local signal for {before:.1f} s before and {after:.1f} s after the gap")
    return fig
