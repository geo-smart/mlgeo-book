"""Slide figure: the same physical signal at its native sampling and as distributed.

(a) River discharge, Nooksack River at Ferndale, WA (USGS 12213100), during the
    November 2021 atmospheric-river flood: USGS instantaneous values (15-minute)
    against the USGS daily mean values for the same days.
(b) Ground velocity at IU.COR (Corvallis, OR) from the M6.4 Ferndale, CA
    earthquake of 2022-12-20 (USGS event nc73821036), 460 km away: the same
    sensor's 40 Hz channel (BHZ) against its 1 Hz channel (LHZ), both corrected
    for instrument response.
Every value is fetched from its provider at run time (USGS Water Services,
USGS ComCat, EarthScope/IRIS FDSN web services).

Regenerate: pixi run python book/slides/2026/figscripts/native_vs_distributed_series.py
"""
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from obspy import UTCDateTime  # noqa: E402
from obspy.clients.fdsn import Client  # noqa: E402
from obspy.geodetics import gps2dist_azimuth  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold"})
INK, MUTED = "#26241f", "#6e675c"
NATIVE_C, DIST_C = "#2a78d6", "#eb6834"        # dataviz reference slots 1 and 2
CFS_TO_M3S = 0.0283168

# ---------------------------------------------------------------- (a) river
SITE, START, END = "12213100", "2021-11-13", "2021-11-20"


def nwis_rdb(service, extra):
    url = (f"https://nwis.waterservices.usgs.gov/nwis/{service}/?format=rdb&sites={SITE}"
           f"&startDT={START}&endDT={END}&parameterCd=00060{extra}")
    lines = [ln for ln in requests.get(url, timeout=120).text.splitlines() if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t")
    return df


iv = nwis_rdb("iv", "")
qcol = [c for c in iv.columns if c.endswith("_00060")][0]
iv["t"] = pd.to_datetime(iv["datetime"])                      # local standard time (tz_cd = PST)
iv["q"] = iv[qcol].astype(float) * CFS_TO_M3S
dv = nwis_rdb("dv", "&statCd=00003")
dcol = [c for c in dv.columns if c.endswith("_00060_00003")][0]
dv["t"] = pd.to_datetime(dv["datetime"])
dv["q"] = dv[dcol].astype(float) * CFS_TO_M3S
peak_iv = iv.loc[iv["q"].idxmax()]
peak_dv = dv.loc[dv["q"].idxmax()]

# ---------------------------------------------------------------- (b) seismic
usgs, iris = Client("USGS"), Client("IRIS")
ev = usgs.get_events(eventid="nc73821036")[0]
org, mag = ev.preferred_origin(), ev.preferred_magnitude()
inv = iris.get_stations(network="IU", station="COR", location="00", channel="BHZ,LHZ", level="response",
                        starttime=org.time)
sta = inv[0][0]
dist_km = gps2dist_azimuth(org.latitude, org.longitude, sta.latitude, sta.longitude)[0] / 1e3
t0 = org.time
st = iris.get_waveforms("IU", "COR", "00", "BHZ,LHZ", t0 - 120, t0 + 420)
st.detrend("demean")
st.remove_response(inventory=inv, output="VEL", pre_filt=None,
                   water_level=60)
st.trim(t0 + 135, t0 + 180)
bh, lh = st.select(channel="BHZ")[0], st.select(channel="LHZ")[0]

# ---------------------------------------------------------------- figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.4))
ax1.plot(iv["t"], iv["q"], color=NATIVE_C, lw=1.6, label="instantaneous, every 15 min")
dv_t = list(dv["t"]) + [dv["t"].iloc[-1] + pd.Timedelta(days=1)]
dv_q = list(dv["q"]) + [dv["q"].iloc[-1]]
ax1.step(dv_t, dv_q, where="post", color=DIST_C, lw=2.6, label="daily mean")
ax1.set_ylim(0, 2000)
ax1.annotate(f"peak {peak_iv['q']:,.0f} m³/s\n{peak_iv['t']:%d %b %H:%M} PST", (peak_iv["t"], peak_iv["q"]),
             xytext=(40, 20), textcoords="offset points", fontsize=15, color=NATIVE_C, va="bottom", ha="left")
ax1.annotate(f"daily mean {peak_dv['q']:,.0f} m³/s", (peak_dv["t"] + pd.Timedelta(hours=12), peak_dv["q"]),
             xytext=(30, -42), textcoords="offset points", fontsize=15, color=DIST_C,
             arrowprops=dict(arrowstyle="-", color=DIST_C, lw=1))
ax1.set_ylabel("discharge (m³/s)")
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
ax1.set_xlabel("November 2021, local standard time (PST)")
ax1.set_xlim(pd.Timestamp("2021-11-13"), pd.Timestamp("2021-11-21"))
ax1.set_title("(a) Nooksack River at Ferndale, WA, Nov 2021", loc="left")
ax1.legend(frameon=False, loc="center right", fontsize=14)

tl = lh.times() + (lh.stats.starttime - t0)
tb = bh.times() + (bh.stats.starttime - t0)
ax2.plot(tb, bh.data * 1e3, color=NATIVE_C, lw=1.3, label=f"BHZ, {bh.stats.sampling_rate:g} samples/s")
ax2.plot(tl, lh.data * 1e3, color=DIST_C, lw=2.4, marker="o", ms=5, label=f"LHZ, {lh.stats.sampling_rate:g} sample/s")
ax2.set_xlabel(f"seconds after origin time ({t0.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
ax2.set_ylabel("vertical ground velocity (mm/s)")
ax2.set_title(f"(b) IU.COR, Corvallis, OR: M{mag.mag:.1f} at {dist_km:.0f} km (strongest 45 s)", loc="left")
ax2.legend(frameon=False, loc="lower left", fontsize=14)
for ax in (ax1, ax2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK)
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "figs" / "2.6_resampling" / "native_vs_distributed_series.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")
print(f"IV peak {peak_iv[qcol]:.0f} cfs = {peak_iv['q']:.0f} m3/s at {peak_iv['t']} PST; "
      f"DV max {peak_dv[dcol]:.0f} cfs = {peak_dv['q']:.0f} m3/s on {peak_dv['t']:%Y-%m-%d}; "
      f"ratio {peak_iv['q'] / peak_dv['q']:.2f}; IV approval codes {sorted(iv[qcol + '_cd'].unique())}")
print(f"event {ev.resource_id.id.split('eventid=')[-1].split('&')[0]} M{mag.mag} {mag.magnitude_type} "
      f"{org.time} dist {dist_km:.1f} km; BHZ peak {np.abs(bh.data).max() * 1e3:.3f} mm/s, "
      f"LHZ peak {np.abs(lh.data).max() * 1e3:.3f} mm/s, ratio {np.abs(bh.data).max() / np.abs(lh.data).max():.2f}")
