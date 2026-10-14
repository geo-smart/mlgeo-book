"""Synthetic twin for the Session 7 volcano lab (Mount St Helens, 2004). The self-checks are in lab_s07_volcano_checks.py.

The twin imitates three hours of the short-period vertical seismometer HSR
(South Ridge, 2.5 km south of the 2004 vent) during dome building: a 100-sample-
per-second record in digitizer counts with a constant offset, background noise
in the sensor's passband, regularly spaced "drumbeat" earthquakes and a few
larger earthquakes, plus the event list built from it (time, peak amplitude in
counts within 6 s, duration magnitude Md, magnitude type, id). The values are
anchored to the real HSR record and to Moran and others (2008,
https://doi.org/10.3133/pp17502): 100 Hz sampling, a recording range that
stops at -3584 and +3582 counts (every twin is cut there, as the real record is:
"Note that peak amplitudes clip at ~3,400 counts for station HSR", Moran and
others, 2008), an amplitude of about 300 counts for Md 1.0, and the sensor's
own response (S-13, natural frequency 1 Hz, damping 0.8, from the StationXML
poles) for the calibration pulse. The drumbeat spacing (32 to 45 s) and its
regularity (CV 0.15 to 0.28) are chosen for the twin, not measured.

The twin carries six problems drawn from eight types that real volcano
seismic records suffer from. The seed comes from the student's NetID, so
classmates hunt different sets; an empty NetID gives the class seed used for
the worked solution and the book build.

Nothing here downloads anything. Run from book/Chapter2-DataManipulation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

FS = 100.0                      # samples per second, as HSR in 2004
CLIP_LO, CLIP_HI = -3584, 3582  # where the real HSR record stops (snapshot, Part 3)
AMP_MD1 = 300.0                 # counts at HSR for Md about 1.0 (Moran and others, 2008, appendix 2)
CLASS_SEED_TEXT = "ess469-2026-class"
HOURS = 3.0

S13_F0, S13_H = 1.0, 0.8          # S-13 natural frequency (Hz) and damping, from the HSR StationXML poles

# The eight problem types that can be planted, in plain words, then four that never are. Clipping is
# never planted because it is not a fault: every twin stops at HSR's recording range, as the real record does.
POOL = ["calibration pulse", "zeros", "dropout", "gain change", "telemetry burst", "clock jump", "duplicate",
        "magnitude method change"]
KINDS = POOL + ["clipping", "spike", "unit change", "drift"]
N_PLANTED = 6

WHY_PHYSICAL = {
    "calibration pulse": "a smooth single swing with the shape of the sensor's own response (1 Hz, damped), then the same swing reversed a few seconds later; no P, no S, no coda, listed as an event with a small magnitude but a large amplitude",
    "zeros": "a working seismometer always records ground noise; minutes of exact zeros mean the logger filled missing data with 0",
    "dropout": "no samples at all for several minutes: the station or its telemetry stopped sending",
    "gain change": "the noise level and every earthquake amplitude change by the same factor at one instant, which the ground does not do",
    "telemetry burst": "a few seconds of boxy noise with no earthquake shape and no coda, listed as an event with a tiny magnitude but a huge amplitude",
    "clock jump": "after one instant the listed times stop lining up with the arrivals in the seismogram, by a constant offset",
    "duplicate": "two events less than a second apart with identical amplitude and magnitude; a drumbeat's coda lasts more than 10 s",
    "magnitude method change": "for a block of events the magnitudes drop by about half a unit while the amplitudes in the seismogram do not, and the label stays md: another method (for drumbeats, an amplitude magnitude comes out lower than a coda-duration Md)",
}
NOT_A_FAULT = {
    "clipping": "the analog telemetry's range: every twin stops there, as the real HSR record does, so the large earthquakes' amplitudes are lower bounds",
}

INSTRUCTIONS = """I attach two files from a short-period vertical seismometer 2.5 km from an erupting lava dome:
(1) {mseed} (MiniSEED, 100 samples per second, raw counts; if you cannot read MiniSEED use {csv1s}, which
gives for each second the minimum, maximum and mean count and the number of samples), and
(2) {events} (the list of events built from this record: time in UTC, peak amplitude in counts within 6 s
of the onset, duration magnitude, magnitude type, id).
The files contain problems introduced by the instrument, the telemetry, the clock, or the people and programs
that built the event list. List every problem you find, each with its kind, the time in minutes after the
start of the record, and the evidence. Then compute the coefficient of variation (standard deviation divided
by mean) of the time between consecutive events with magnitude below 2.0, before and after your cleaning.
Use only these files. Say which problems you could not undo, and say so if you could not run code or open a file."""


# ----------------------------------------------------------------------------- seeds
def seed_from_netid(netid: str | None) -> int:
    text = (netid or "").strip().lower() or CLASS_SEED_TEXT
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------- the twin
class Twin:
    """A synthetic three-hour HSR-like record and its event list, with known truth."""

    def __init__(self, netid: str | None = None):
        self.netid = (netid or "").strip().lower()
        self.tag = self.netid or "class"
        self.seed = seed_from_netid(netid)
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        n = int(HOURS * 3600 * FS)
        self.n = n
        day = int(rng.integers(0, 60))
        self.start = pd.Timestamp("2004-12-01T00:00:00Z") + pd.Timedelta(days=day)
        self.offset = float(rng.integers(140, 190))
        self.noise_sigma = float(rng.uniform(25, 40))
        self.mean_interval = float(rng.uniform(32, 45))
        self.cv_true_param = float(rng.uniform(0.15, 0.28))

        # 1. Background noise in the S-13 passband (about 1 to 20 Hz).
        sos = signal.butter(4, [1.0, 20.0], btype="bandpass", fs=FS, output="sos")
        noise = signal.sosfiltfilt(sos, rng.standard_normal(n + 2000))[1000:-1000]
        noise *= self.noise_sigma / noise.std()

        # 2. Event times: drumbeats, then three larger earthquakes (a separate process).
        t, times = 20.0, []
        while t < HOURS * 3600 - 30:
            times.append(t)
            t += max(8.0, self.mean_interval * (1 + self.cv_true_param * rng.standard_normal()))
        drum_t = np.array(times)
        self._drum_t = drum_t
        drum_peak = 10 ** (np.log10(rng.uniform(300, 450)) + 0.2 * rng.standard_normal(len(drum_t)))
        big_t = np.array([rng.uniform(10, 55), rng.uniform(70, 115), rng.uniform(125, 170)]) * 60
        big_md = rng.uniform(2.2, 2.8, 3)
        big_peak = AMP_MD1 * 10 ** (big_md - 1.0)

        # 3. Waveform templates: a family of similar drumbeats (low frequency), and larger events.
        def template(fmin, fmax, decay, length):
            k = int(length * FS)
            sos_t = signal.butter(3, [fmin, fmax], btype="bandpass", fs=FS, output="sos")
            w = signal.sosfiltfilt(sos_t, rng.standard_normal(k + 400))[200:-200]
            tt = np.arange(k) / FS
            env = np.minimum(tt / 0.4, 1.0) * np.exp(-np.maximum(tt - 0.4, 0) / decay)
            w = w * env
            return w / np.abs(w).max()

        drum_tpl = [template(1.0, 5.0, 2.5, 20) for _ in range(6)]
        big_tpl = [template(1.0, 8.0, 4.5, 45) for _ in range(3)]
        x = noise.copy()
        for ti, pk in zip(drum_t, drum_peak):
            i0 = int(ti * FS); w = drum_tpl[int(rng.integers(0, 6))] * pk
            x[i0:i0 + len(w)] += w[: n - i0]
        for j, (ti, pk) in enumerate(zip(big_t, big_peak)):
            i0 = int(ti * FS); w = big_tpl[j] * pk
            x[i0:i0 + len(w)] += w[: n - i0]
        self.truth_wave = np.round(x + self.offset)

        # 4. Truth event list.
        ev = pd.DataFrame({"t_s": np.r_[drum_t, big_t], "peak_true": np.r_[drum_peak, big_peak], "big": np.r_[np.zeros(len(drum_t), bool), np.ones(3, bool)]})
        ev = ev.sort_values("t_s").reset_index(drop=True)
        ev["mag"] = np.round(1.0 + np.log10(ev["peak_true"] / AMP_MD1) + 0.08 * rng.standard_normal(len(ev)), 1)
        ev["magType"] = "md"
        ev["id"] = self._ids(len(ev), salt=1)
        self._truth_events = ev
        self.big_minutes = list(np.round(big_t / 60, 2))
        self._big_peak = list(big_peak)

        # 5. Choose and place the problems.
        self.planted = sorted(rng.choice(POOL, size=N_PLANTED, replace=False).tolist(), key=POOL.index)
        self.where = {}
        self._place(rng)
        self._contaminate(rng)

    # ------------------------------------------------------------------ placement
    def _place(self, rng):
        busy = [(b - 1.5, b + 1.5) for b in self.big_minutes]

        def free(a, b, sep=4.0):
            return all(b + sep <= lo or a - sep >= hi for lo, hi in busy)

        def interval(lo, hi, length):
            for _ in range(500):
                a = float(rng.uniform(lo, hi - length))
                if free(a, a + length):
                    busy.append((a, a + length))
                    return round(a, 2), round(a + length, 2)
            raise RuntimeError("could not place a problem")

        if "zeros" in self.planted:
            self.where["zeros"] = interval(8, 172, float(rng.uniform(2, 6)))
        if "dropout" in self.planted:
            self.where["dropout"] = interval(8, 172, float(rng.uniform(8, 18)))
        if "gain change" in self.planted:
            lo = max(80.0, self.big_minutes[0] + 5)
            self.where["gain change"] = interval(lo, 166, 0.2)[0]
            self.gain = float(rng.choice([0.1, 0.25, 0.5]))
        if "clock jump" in self.planted:
            self.where["clock jump"] = interval(25, 170, 0.2)[0]
            self.clock_shift_s = float(rng.choice([-1, 1]) * rng.uniform(5, 12))
        if "magnitude method change" in self.planted:
            a, b = interval(10, 170, float(rng.uniform(10, 16)))
            self.where["magnitude method change"] = (a, b)
            self.ml_offset = float(rng.uniform(0.4, 0.6))
        if "telemetry burst" in self.planted:
            k = int(rng.integers(3, 6))
            self.where["telemetry burst"] = [self._between_drumbeats(interval(5, 175, 0.1)[0], rng, 4.0) for _ in range(k)]
        if "calibration pulse" in self.planted:
            k = int(rng.integers(1, 3))
            self.where["calibration pulse"] = [self._between_drumbeats(interval(5, 175, 0.2)[0], rng, 7.0) for _ in range(k)]
        if "duplicate" in self.planted:
            k = int(rng.integers(3, 6))
            self.where["duplicate"] = [interval(5, 175, 0.1)[0] for _ in range(k)]

    def _between_drumbeats(self, minute, rng, length_s):
        """Move a glitch to the quiet time between two drumbeats, so it never lands in a drumbeat's 6 s
        amplitude window (a real glitch can, but then the drumbeat's listed amplitude is the glitch's)."""
        d = self._drum_t
        j = max(int(np.searchsorted(d, minute * 60)) - 1, 0)
        while j + 1 < len(d):
            t = d[j] + float(rng.uniform(7.0, 9.0))
            if d[j + 1] - t >= length_s + 3.0:
                return round(t / 60, 4)
            j += 1
        return round(minute, 4)

    def _ids(self, n, salt):
        """Event ids as a network writes them: increasing in time order, with irregular steps."""
        g = np.random.default_rng(self.seed + salt)
        return [f"sy{v:06d}" for v in 100000 + np.cumsum(g.integers(3, 12, n))]

    # ------------------------------------------------------------------ contamination
    def _contaminate(self, rng):
        x = self.truth_wave.astype(float).copy()
        tmin = np.arange(self.n) / FS / 60
        if "gain change" in self.planted:
            k = tmin >= self.where["gain change"]
            x[k] = self.offset + self.gain * (x[k] - self.offset)
        # The recording range is part of the instrument, not a planted fault: every twin stops there.
        x = np.clip(x, CLIP_LO, CLIP_HI)
        gmin = self.where.get("gain change", np.inf)
        self.clipped_minutes = [m for m, pk in zip(self.big_minutes, self._big_peak)
                                if (pk if m < gmin else self.gain * pk) + self.offset > CLIP_HI]
        bursts = []
        if "telemetry burst" in self.planted:
            # Noise above 15 Hz that still passes the 20 Hz and 30 Hz analog filters of the HSR response.
            sos = signal.butter(4, [15.0, 20.0], btype="bandpass", fs=FS, output="sos")
            for m in self.where["telemetry burst"]:
                i0 = int(m * 60 * FS); L = int(rng.uniform(2.5, 4.0) * FS)
                b = signal.sosfilt(sos, rng.standard_normal(L + 200))[200:]
                b = b / b.std() * float(rng.uniform(800, 1600)) - 300
                x[i0:i0 + L] = np.clip(self.offset + b, CLIP_LO, CLIP_HI)
                bursts.append((m, L / FS))
        if "calibration pulse" in self.planted:
            # A step of force on the sensor's calibration coil, then its release: the output is the S-13's own
            # damped swing (1 Hz, damping 0.8), once each way.
            w0 = 2 * np.pi * S13_F0; wd = w0 * np.sqrt(1 - S13_H ** 2)
            tt = np.arange(int(3.0 * FS)) / FS
            swing = np.exp(-S13_H * w0 * tt) * np.sin(wd * tt)
            swing /= np.abs(swing).max()
            self.cal_amp = float(rng.uniform(1200, 1800))
            for m in self.where["calibration pulse"]:
                i0 = int(m * 60 * FS); i1 = i0 + int(4.0 * FS)
                x[i0:i0 + len(swing)] += self.cal_amp * swing
                x[i1:i1 + len(swing)] -= self.cal_amp * swing
                bursts.append((m, 7.0))
        if "zeros" in self.planted:
            a, b = self.where["zeros"]
            x[(tmin >= a) & (tmin < b)] = 0.0
        if "dropout" in self.planted:
            a, b = self.where["dropout"]
            x[(tmin >= a) & (tmin < b)] = np.nan
        self.wave = np.round(x)

        # Event list as the detector and the network would have written it.
        ev = self._truth_events.copy()
        no_data = [self.where[k] for k in ("zeros", "dropout") if k in self.where]
        keep = np.ones(len(ev), bool)
        for a, b in no_data:
            keep &= ~((ev["t_s"] / 60 >= a) & (ev["t_s"] / 60 < b))
        ev = ev[keep].copy()
        ev["peak_counts"] = [self._peak(t) for t in ev["t_s"]]
        extra = []
        for m, dur in bursts:
            t = m * 60
            extra.append({"t_s": t, "peak_true": np.nan, "big": False, "mag": round(float(rng.uniform(0.1, 0.5)), 1),
                          "magType": "md", "id": "", "peak_counts": self._peak(t), "planted": "glitch"})
        if "duplicate" in self.planted:
            locs = []
            for m in self.where["duplicate"]:
                cand = ev[(~ev["big"])]
                j = (cand["t_s"] - m * 60).abs().idxmin()
                row = ev.loc[j].to_dict()
                locs.append(round(row["t_s"] / 60, 2))
                row.update(t_s=row["t_s"] + float(rng.uniform(0.05, 0.4)), id="", planted="duplicate")
                extra.append(row)
            self.where["duplicate"] = locs
        ev["planted"] = ""
        ev = pd.concat([ev, pd.DataFrame(extra)], ignore_index=True) if extra else ev
        if "magnitude method change" in self.planted:
            a, b = self.where["magnitude method change"]
            k = (ev["t_s"] / 60 >= a) & (ev["t_s"] / 60 < b)
            ev.loc[k, "mag"] = np.round(ev.loc[k, "mag"] - self.ml_offset, 1)
        ev["t_listed"] = ev["t_s"]
        if "clock jump" in self.planted:
            k = ev["t_s"] / 60 >= self.where["clock jump"]
            ev.loc[k, "t_listed"] = ev.loc[k, "t_s"] + self.clock_shift_s
        ev = ev.sort_values("t_listed").reset_index(drop=True)
        # Ids follow the listed time, so an inserted row carries the next number like any other row.
        ev["id"] = self._ids(len(ev), salt=2)
        self._events_internal = ev
        self.events = pd.DataFrame({
            "time": (self.start + pd.to_timedelta(ev["t_listed"], unit="s")).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-4] + "Z",
            "peak_counts": ev["peak_counts"].round().astype("Int64"),
            "mag": ev["mag"], "magType": ev["magType"], "id": ev["id"]})
        tr = self._truth_events
        self.truth_events = pd.DataFrame({
            "time": (self.start + pd.to_timedelta(tr["t_s"], unit="s")).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-4] + "Z",
            "peak_counts": tr["peak_true"].round().astype("Int64"), "mag": tr["mag"], "magType": tr["magType"], "id": tr["id"]})

    def _peak(self, t_s):
        i0 = int(t_s * FS)
        seg = self.wave[i0:i0 + int(6 * FS)] - self.offset
        seg = seg[np.isfinite(seg) & (self.wave[i0:i0 + int(6 * FS)] != 0)]
        return float(np.abs(seg).max()) if len(seg) else np.nan

    # ------------------------------------------------------------------ views for students
    def minutes(self, times) -> np.ndarray:
        """Minutes after the start of the record for a column of ISO time strings."""
        return ((pd.to_datetime(pd.Series(times)) - self.start).dt.total_seconds() / 60).to_numpy()

    def times_s(self) -> np.ndarray:
        return np.arange(self.n) / FS

    def onset_lags(self, events: pd.DataFrame) -> pd.Series:
        """Listed time minus the nearest onset the waveform shows, in seconds (STA/LTA 1 s / 8 s, ratio 2.3,
        as Moran and others used at HSR). When the list and the record agree it is about -0.3 s: the detector
        triggers a few tenths of a second after the true onset, on the 0.4 s rise of each synthetic event.
        A positive jump means the listed times run late; subtract it to correct them."""
        from obspy.signal.trigger import classic_sta_lta, trigger_onset
        onsets = []
        x = self.wave
        good = np.isfinite(x) & (x != 0)
        edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < 1000:
                continue
            seg = x[a:b] - np.median(x[a:b])
            cft = classic_sta_lta(seg, int(FS), int(8 * FS))
            last = -1e9
            for on, _ in trigger_onset(cft, 2.3, 1.0):
                if on - last >= 6 * FS:
                    onsets.append((a + on) / FS); last = on
        onsets = np.array(onsets)
        t = self.minutes(events["time"]) * 60
        lag = []
        for ti in t:
            d = ti - onsets
            j = np.argmin(np.abs(d)) if len(d) else None
            lag.append(d[j] if j is not None and abs(d[j]) <= 25 else np.nan)
        return pd.Series(np.round(lag, 2), index=events.index, name="lag_s")

    def write_files(self, folder="downloads") -> dict:
        """Write the contaminated twin for an AI tool: MiniSEED, a per-second summary CSV and the event list."""
        from obspy import Stream, Trace, UTCDateTime
        folder = Path(folder); folder.mkdir(exist_ok=True)
        base = f"s07_volcano_HSR_{self.tag}"
        st = Stream()
        x = self.wave
        good = np.isfinite(x)
        edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            tr = Trace(data=x[a:b].astype(np.int32))
            tr.stats.update({"network": "SY", "station": "HSR", "channel": "EHZ", "sampling_rate": FS,
                             "starttime": UTCDateTime(self.start.to_pydatetime()) + a / FS})
            st.append(tr)
        paths = {"mseed": folder / f"{base}_seismogram.mseed", "csv1s": folder / f"{base}_seismogram_1s.csv",
                 "events": folder / f"{base}_events.csv"}
        st.write(str(paths["mseed"]), format="MSEED", encoding="STEIM2")
        sec = pd.DataFrame({"v": x}, index=self.start + pd.to_timedelta(np.arange(self.n) / FS, unit="s"))
        g = sec["v"].resample("1s")
        summ = pd.DataFrame({"min": g.min(), "max": g.max(), "mean": g.mean().round(1), "n_samples": g.count()})
        summ.index = summ.index.strftime("%Y-%m-%dT%H:%M:%SZ"); summ.index.name = "time"
        summ.to_csv(paths["csv1s"])
        self.events.to_csv(paths["events"], index=False)
        return {k: str(v) for k, v in paths.items()}

    def instructions(self, paths: dict) -> str:
        """The fixed instructions given to the AI tool in Part 8, with the file names filled in."""
        return INSTRUCTIONS.format(**{k: Path(v).name for k, v in paths.items()})

    agent_prompt = instructions     # older notebook copies call it by this name

    # ------------------------------------------------------------------ the derived quantity
    def clock(self, events: pd.DataFrame, exclude_minutes=(), label="") -> dict:
        """The drumbeat clock: events with magnitude below 2.0, sorted by time; intervals between consecutive
        ones, skipping any interval that overlaps an excluded window (minutes after the start).
        Returns the number of drumbeats, the rate per recorded hour, the median interval and its
        coefficient of variation (CV = standard deviation / mean; 0 is a perfect clock, about 1 is random)."""
        m = np.sort(self.minutes(events.loc[events["mag"] < 2.0, "time"]))
        dt = np.diff(m) * 60
        ok = np.ones(len(dt), bool)
        hours = HOURS
        for a, b in exclude_minutes:
            ok &= ~((m[1:] > a) & (m[:-1] < b))
            hours -= (b - a) / 60
        dt = dt[ok]
        out = {"label": label, "n_drumbeats": len(m), "rate_per_hour": round(len(m) / hours, 1),
               "median_interval_s": round(float(np.median(dt)), 1), "cv": round(float(dt.std() / dt.mean()), 3)}
        return out

    # ------------------------------------------------------------------ scoring
    def _match(self, kind, minute):
        w = self.where.get(kind)
        if w is None:
            return False
        if isinstance(w, tuple):
            return w[0] - 2 <= minute <= w[1] + 2
        if isinstance(w, list):
            return any(abs(minute - v) <= 2 for v in w)
        return abs(minute - w) <= 3

    def _sort(self, findings):
        """Split findings into planted kinds found, unmatched findings, real kinds, real earthquakes flagged, unreadable."""
        found, unmatched, bad, real_flagged, real_kind = set(), 0, [], [], set()
        for f in findings:
            kind = str(f.get("kind", "")).strip().lower()
            try:
                minute = float(f.get("minute"))
            except (TypeError, ValueError):
                bad.append(f); continue
            if kind not in KINDS:
                bad.append(f); continue
            if kind in NOT_A_FAULT:
                real_kind.add(kind); continue
            if any(abs(minute - b) <= 1.0 for b in self.big_minutes) and not self._match(kind, minute):
                real_flagged.append(minute)
            if kind in self.planted and self._match(kind, minute):
                found.add(kind)
            else:
                unmatched += 1
        return found, unmatched, bad, real_flagged, real_kind

    def match_findings(self, findings) -> set:
        """The planted kinds that the findings locate; prints nothing."""
        return self._sort(findings or [])[0]

    def score(self, findings, who="you", show_missed=False) -> dict:
        """Compare findings ([{'kind', 'minute', 'why'}, ...]) with the planted problems. It never prints
        locations, and never names the kinds that are not planted (that would give the planted set away)."""
        if not findings:
            print("not answered yet")
            return {}
        found, unmatched, bad, real_flagged, real_kind = self._sort(findings)
        n = len(self.planted)
        print(f"{who}: {len(found)} of {n} planted problems found.")
        if unmatched:
            print(f"  {unmatched} finding(s) do not match a planted problem. Check the minute, and whether you flagged a real feature.")
        for k in sorted(real_kind):
            print(f"  '{k}' is real, not planted: {NOT_A_FAULT[k]}. Keep it in your reflection on amplitudes.")
        if real_flagged:
            print(f"  {len(real_flagged)} finding(s) sit on a large earthquake, which is real: a real extreme is not an error.")
        if bad:
            print(f"  {len(bad)} finding(s) could not be read: use a kind from {KINDS} and a number for minute.")
        if show_missed:
            missed = [k for k in self.planted if k not in found]
            print("  kinds you missed:", missed if missed else "none")
        elif len(found) < n:
            print("  When you have looked again, set SHOW_MISSED = True to see which kinds you missed.")
        return {"found": sorted(found), "n_planted": n}


# ----------------------------------------------------------------------------- self-checks
# The self-checks moved to lab_s07_volcano_checks.py. These names stay importable from here so that
# notebook copies made before the move still run.
from lab_s07_volcano_checks import check, check_expectations, check_pipeline, check_text  # noqa: E402,F401
