"""Session 7 deck (lec07_reading_data.qmd): file excerpts, the station map and the mini-lecture figures.

Everything is read from the snapshot files in book/Chapter2-DataManipulation/snapshots,
the same bytes the lab notebook and lab_s07_tools.py read:
  excerpts      exact lines or bytes of each file, written as Markdown code blocks the deck includes
  station map   coordinates from each provenance record (USGS site file, ghcnd-stations.txt,
                ComCat query box, StationXML, USGS event GeoJSON, NetCDF grid)
  mini-lecture  SeaTac June-August TMAX, Aki b-values for the Puget catalog, the NCEP spectrum at
                the SeaTac cell, the Welch spectrum of the RATT noise hour; every number is printed
  land          Natural Earth 10m

Regenerate: pixi run python book/slides/2026/figscripts/lec07_reading_data.py
"""
import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import obspy  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import xarray as xr  # noqa: E402
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees  # noqa: E402
from scipy import signal, stats  # noqa: E402

plt.rcParams.update({"font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold"})
HERE = Path(__file__).resolve().parent
SNAP = HERE.parents[2] / "Chapter2-DataManipulation" / "snapshots"
OUT = HERE.parent / "figs" / "lec07_reading_data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = pooch.os_cache("mlgeo-figscripts")
INK, MUTED = "#26241f", "#6e675c"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # dataviz reference slots 1 to 3

F0 = SNAP / "usgs_12213100_2021-11.csv"
FA = SNAP / "ghcnd_USW00024233_1948-2024.csv"
FB = SNAP / "comcat_puget_2000-2025.csv"
RATT = "ratt_chignik_2021-07-29_"
FD = SNAP / (RATT + "UW.RATT.HHZ_0615-0815.mseed")
FD_NOISE = SNAP / (RATT + "UW.RATT.HHZ_0415-0615.mseed")
FD_XML = SNAP / (RATT + "UW.RATT.stationxml.xml")
FD_JSON = SNAP / (RATT + "usgs_ak0219neiszm.geojson")
FE = SNAP / "ncep_air_2013-2014.nc"


def prov(name):
    return json.loads((SNAP / name).read_text())


def write_block(name, text, lang=""):
    """Write a fenced code block that the deck includes verbatim (.qmd, like refs/, so the book build skips it)."""
    (OUT / f"{name}.qmd").write_text(f"```{lang}\n{text.rstrip(chr(10))}\n```\n")
    print(f"wrote {name}.qmd ({len(text.splitlines())} lines)")


def lines(path, numbers):
    """Exact lines of a text file, by 1-based line number."""
    all_lines = path.read_text().splitlines()
    return [all_lines[n - 1] for n in numbers]


def find(path, needle):
    """1-based numbers of the lines containing needle."""
    return [i for i, ln in enumerate(path.read_text().splitlines(), 1) if needle in ln]


# ---------------------------------------------------------------------------
# 1. Excerpts: exact file content
# ---------------------------------------------------------------------------
# Snippet 1, GHCN-Daily SeaTac: header, the first trace day, three ordinary winter days with rain.
# No row here is an answer on a station A card (not an extreme, not flagged).
nA = [1, find(FA, ",0,T,")[0]] + find(FA, "1975-01-12") + find(FA, "1975-01-13") + find(FA, "1975-01-14")
write_block("snippet_1", "\n".join(lines(FA, nA)))
print("  snippet 1 lines", nA)

# Snippet 2, ComCat Puget: header, an ordinary md event, a small negative-depth ml event at Mount St Helens.
# Neither row is an answer on a station B card (not the largest, not the shallowest, not a duplicate).
cat = pd.read_csv(FB)
nB = [1, find(FB, "uw10581508")[0], find(FB, "uw61384256")[0]]
assert cat.loc[cat["id"] == "uw61384256", "depth"].item() > cat["depth"].min()
write_block("snippet_2", "\n".join(lines(FB, nB)))
print("  snippet 2 lines", nB)

# Snippet 3, MiniSEED: the first 64 bytes as xxd prints them, then obspy's summary
hexdump = subprocess.run(["xxd", "-l", "64", str(FD)], capture_output=True, text=True, check=True).stdout
st = obspy.read(str(FD))
write_block("snippet_3", hexdump + "\n>>> print(st)\n" + str(st))

# Snippet 4, NetCDF header (ncdump -h), the lines about the variable air and the grid
hdr = subprocess.run(["ncdump", "-h", str(FE)], capture_output=True, text=True, check=True).stdout.splitlines()
keep = [ln for ln in hdr if ln.startswith(("dimensions", "variables")) or ln.strip().startswith(("lat = ", "lon = ", "time = "))
        or ln.strip().startswith(("short air(", "air:units", "air:scale_factor", "float lat(", "float lon("))]
write_block("snippet_4", "\n".join(keep))
print("  snippet 4: ncdump -h lines kept:", len(keep), "of", len(hdr))

# Snippet 5, USGS GeoJSON event: the first and the last bytes of a one-line file
raw = FD_JSON.read_text()
head_end = raw.index('"updated"')
tail_start = raw.rindex(',"geometry"') + 1
write_block("snippet_5", raw[:head_end] + "\n" + raw[tail_start:])
print(f"  snippet 5: bytes 0-{head_end} and {tail_start}-{len(raw)} of {len(raw)}, lines in file:",
      raw.count("\n"))

# Station 0, Nooksack: header and first row, the repeated hour, the off-grid peak, the first daily mean
n0 = [1, 2] + find(F0, "2021-11-07 01:00") + find(F0, "2021-11-16 07:45") + find(F0, "2021-11-16 07:52") \
    + find(F0, "2021-11-16 08:00") + [find(F0, "daily_mean")[0]]
write_block("nooksack", "\n".join(f"{n:>4}: {s}" for n, s in zip(n0, lines(F0, n0))))
u = pd.read_csv(F0)
iv = u[u["product"] == "instantaneous"]
t_iv = pd.to_datetime(iv["time_local"])
print(f"Nooksack: shape {u.shape}; instantaneous {len(iv)}, daily {int((u['product'] == 'daily_mean').sum())}; "
      f"30 days x 96 = {30 * 96}; repeated local stamps {int(iv['time_local'].duplicated().sum())}; "
      f"off the 15-min grid {int((t_iv.dt.minute % 15 != 0).sum())}; tz {iv['tz'].value_counts().to_dict()}")
print(f"  peak {iv['discharge_cfs'].max():,.0f} cfs = {iv['discharge_cfs'].max() * 0.028316846592:,.1f} m3/s")

# ---------------------------------------------------------------------------
# 2. Station map, coordinates from the provenance records
# ---------------------------------------------------------------------------
p0, pA, pB, pD = prov("usgs_12213100_2021-11.provenance.json"), prov("ghcnd_USW00024233_1948-2024.provenance.json"), \
    prov("comcat_puget_2000-2025.provenance.json"), prov("ratt_chignik_2021-07-29.provenance.json")
q = pB["query"]
ratt = (pD["station"]["latitude"], pD["station"]["longitude"])
ev = pD["event"]
with xr.open_dataset(FE) as ds:
    lat_e, lon_e = ds["lat"].values, ds["lon"].values
dist_m, _, _ = gps2dist_azimuth(ev["latitude"], ev["longitude"], *ratt)
print(f"Chignik to RATT: {dist_m / 1000:,.0f} km, {kilometers2degrees(dist_m / 1000):.1f} deg")

fetch = lambda url, name: pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)  # noqa: E731
land = gpd.read_file(fetch("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip", "ne_10m_land.zip"))

fig = plt.figure(figsize=(20, 7.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.08)
ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (ax1, ax2):
    ax.set_facecolor("#e8eef3")
    land.plot(ax=ax, color="#f4f1ea", edgecolor="#b9b2a5", linewidth=0.6)
# (a) Pacific Northwest
ax1.add_patch(plt.Rectangle((q["minlongitude"], q["minlatitude"]), q["maxlongitude"] - q["minlongitude"],
                            q["maxlatitude"] - q["minlatitude"], fill=False, ec=C2, lw=3, ls="--"))
pts = [("0  Nooksack gauge", p0["latitude"], p0["longitude"], C3, "o", (12, 4)),
       ("A  SeaTac airport", pA["latitude"], pA["longitude"], C1, "s", (-14, 0)),
       ("D  RATT", ratt[0], ratt[1], INK, "^", (12, -4))]
for lab, la, lo, c, m, off in pts:
    ax1.plot(lo, la, marker=m, color=c, markersize=15, mec="white", mew=1.5, zorder=5)
    ax1.annotate(lab, (lo, la), xytext=off, textcoords="offset points", fontsize=18, color=INK,
                 ha="right" if off[0] < 0 else "left", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax1.text(q["minlongitude"] + 0.08, q["minlatitude"] + 0.1, "B  ComCat box", color=C2, fontsize=18, weight="bold")
ax1.set_xlim(-125.2, -120.3); ax1.set_ylim(45.2, 49.4)
ax1.set_aspect(1 / math.cos(math.radians(47.3)))
ax1.set_title("Pacific Northwest", loc="left")
ax1.set_xlabel("longitude (°)"); ax1.set_ylabel("latitude (°)")
# (b) The NCEP grid and the Chignik earthquake
lo0, lo1 = lon_e.min() - 360, lon_e.max() - 360
ax2.add_patch(plt.Rectangle((lo0, lat_e.min()), lo1 - lo0, lat_e.max() - lat_e.min(), fill=False, ec=C1, lw=3))
ax2.text(lo0 + 1.5, lat_e.min() + 1.5, f"E  NCEP grid, {lat_e.size} × {lon_e.size} cells of 2.5°", color=C1,
         fontsize=18, weight="bold")
ax2.plot([ev["longitude"], ratt[1]], [ev["latitude"], ratt[0]], color=MUTED, lw=2, ls=":")
ax2.plot(ev["longitude"], ev["latitude"], marker="*", color=C2, markersize=26, mec="white", mew=1.2, zorder=5)
ax2.annotate(f"M{ev['magnitude']} Chignik\n{ev['origin_time_utc'][:10]}", (ev["longitude"], ev["latitude"]),
             xytext=(0, -46), textcoords="offset points", ha="center", fontsize=17, color=INK,
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax2.plot(ratt[1], ratt[0], marker="^", color=INK, markersize=15, mec="white", mew=1.5, zorder=5)
ax2.annotate(f"RATT, {dist_m / 1000:,.0f} km", (ratt[1], ratt[0]), xytext=(12, 12), textcoords="offset points",
             fontsize=17, color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax2.set_xlim(-170, -25); ax2.set_ylim(10, 80)
ax2.set_aspect(1 / math.cos(math.radians(45)))
ax2.set_title("North America and the Alaska Peninsula", loc="left")
ax2.set_xlabel("longitude (°)"); ax2.set_ylabel("")
fig.savefig(OUT / "station_map.png", dpi=110, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote station_map.png")

# ---------------------------------------------------------------------------
# 3. Mini-lecture numbers and figures
# ---------------------------------------------------------------------------
g = pd.read_csv(FA, parse_dates=["date"])
jja = g.loc[g["date"].dt.month.isin([6, 7, 8]), "TMAX"].dropna() / 10
hot = g.loc[g["TMAX"].idxmax()]
tmax = hot["TMAX"] / 10
mu, sd = jja.mean(), jja.std()
z = (tmax - mu) / sd
p_tail = stats.norm.sf(z)
annual = g.groupby(g["date"].dt.year)["TMAX"].max() / 10
prev = annual[annual.index < 2021]
print(f"SeaTac JJA TMAX: n {jja.size}, mean {mu:.2f} C, sd {sd:.2f} C; hottest {hot['date']:%Y-%m-%d} {tmax:.1f} C; "
      f"z {z:.2f}; normal P(>z) {p_tail:.2e}; expected days {p_tail * jja.size:.2f}")
print(f"  previous annual maximum record {prev.max():.1f} C in {prev.idxmax()}")
print(f"  JJA TMAX skewness {stats.skew(jja):.2f}; kurtosis plain {stats.kurtosis(jja, fisher=False):.2f}, "
      f"excess {stats.kurtosis(jja):.2f}")
n_hot = int((jja >= 38.0).sum())
print(f"  observed JJA days >= 38.0 C: {n_hot}; normal expects {stats.norm.sf((38.0 - mu) / sd) * jja.size:.2f}")

mag = cat["mag"]
brows = []
for mc in (1.0, 1.5, 2.0):
    m = mag[mag >= mc - 1e-9]
    row = {"Mc": mc, "n": int(m.size)}
    for dm in (0.1, 0.01):
        row[f"b_{dm}"] = math.log10(math.e) / (m.mean() - (mc - dm / 2))     # Aki 1965 with Utsu's half bin
    brows.append(row)
bt = pd.DataFrame(brows)
print("Aki b, all magTypes, whole box 2000-2025, raw magnitudes >= Mc, Utsu half-bin correction:")
print(bt.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
mt = cat["magType"].value_counts().to_dict()
on01 = {k: float(np.isclose(cat.loc[cat["magType"] == k, "mag"] * 10,
                            np.round(cat.loc[cat["magType"] == k, "mag"] * 10), atol=1e-6).mean()) for k in ("md", "ml")}
print(f"  magType counts {mt}; share on the 0.1 grid {on01}")

with xr.open_dataset(FE) as ds:
    cell = ds["air"].sel(lat=pA["latitude"], lon=360 + pA["longitude"], method="nearest").load()
    time = ds["time"].values
dt_days = float(np.diff(time[:2])[0] / np.timedelta64(1, "D"))
x = cell.values - cell.values.mean()
f_e = np.fft.rfftfreq(x.size, d=dt_days)
p_e = np.abs(np.fft.rfft(x)) ** 2
top = (np.argsort(p_e[1:])[::-1][:3] + 1)
print(f"NCEP cell nearest SeaTac: {float(cell.lat)} N, {float(cell.lon)} E; mean {float(cell.mean()) - 273.15:.2f} C; "
      f"step {dt_days * 24:.0f} h; n {x.size}; top peaks (cpd) {[round(float(f_e[k]), 5) for k in top]} "
      f"= periods (days) {[round(1 / float(f_e[k]), 2) for k in top]}; Nyquist {f_e.max():.1f} cpd "
      f"= period {24 / f_e.max():.0f} h")

noise = obspy.read(str(FD_NOISE))[0]
inv = obspy.read_inventory(str(FD_XML))
sens = inv.select(channel="HHZ")[0][0][0].response.instrument_sensitivity.value
y = (noise.data.astype(float) - noise.data.mean()) / sens
f_d, p_d = signal.welch(y, fs=noise.stats.sampling_rate, nperseg=2 ** 15)
band = (f_d >= 0.03) & (f_d <= 1.0)
f_pk = float(f_d[band][np.argmax(p_d[band])])
print(f"RATT noise {noise.stats.starttime} to {noise.stats.endtime}: Welch 2^15 samples "
      f"({2 ** 15 / noise.stats.sampling_rate:.2f} s, df {f_d[1]:.4f} Hz); largest peak 0.03-1 Hz at {f_pk:.3f} Hz "
      f"= {1 / f_pk:.1f} s; sensitivity {sens:,.0f}")

# Figure: which moment (heat-dome histogram, b with Mc and bin width)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 7.0), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.28})
bins = np.arange(np.floor(jja.min()), 44, 1.0)
a1.hist(jja, bins=bins, color=C1, alpha=0.85, label=f"{jja.size:,} June–August days")
xx = np.linspace(bins[0], 44, 300)
a1.plot(xx, stats.norm.pdf(xx, mu, sd) * jja.size, color=INK, lw=2.5, label="normal fit")
a1.axvline(tmax, color=C2, lw=3)
a1.set_yscale("log"); a1.set_ylim(0.5, 3e4)
a1.annotate(f"{tmax:.1f} °C\n{hot['date']:%Y-%m-%d}\nz = {z:.2f}", (tmax, 150),
            xytext=(-14, 0), textcoords="offset points", ha="right", fontsize=18, color=C2, weight="bold")
a1.set_xlabel("daily maximum temperature (°C)"); a1.set_ylabel("days per 1 °C bin")
a1.set_title("SeaTac, June–August 1948–2024", loc="left")
a1.legend(fontsize=16, loc="upper left", frameon=False, ncol=2)
for col, c, lab in (("b_0.1", C1, "bin 0.1"), ("b_0.01", C2, "bin 0.01")):
    a2.plot(bt["Mc"], bt[col], marker="o", color=c, lw=3, markersize=12, label=lab)
    for mc, b in zip(bt["Mc"], bt[col]):
        a2.annotate(f"{b:.2f}", (mc, b), xytext=(-58, 10) if c == C2 else (10, -26), textcoords="offset points",
                    fontsize=17, color=c)
a2.set_xticks(bt["Mc"]); a2.set_xlim(0.75, 2.3); a2.set_ylim(0.5, 1.3)
a2.set_xlabel("magnitude of completeness Mc"); a2.set_ylabel("Aki b-value")
a2.set_title("Puget Sound catalog, 2000–2025", loc="left")
a2.legend(fontsize=16, loc="upper left", frameon=False)
fig.savefig(OUT / "which_moment.png", dpi=110, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote which_moment.png")

# Figure: which spectrum (NCEP periodogram at the SeaTac cell, RATT Welch spectrum of the noise hour)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 7.0), gridspec_kw={"wspace": 0.25})
a1.loglog(f_e[1:], p_e[1:], color=C1, lw=1.2)
for k, lab in zip(top[:2], ("annual", "daily")):
    a1.plot(f_e[k], p_e[k], "o", color=C2, markersize=11)
    a1.annotate(f"{lab}\n{f_e[k]:.4g} cycle/day", (f_e[k], p_e[k]), xytext=(12, -8) if lab == "annual" else (-14, 6),
                textcoords="offset points", fontsize=17, color=C2, va="top" if lab == "annual" else "bottom",
                ha="left" if lab == "annual" else "right")
a1.axvline(f_e.max(), color=MUTED, ls="--", lw=2)
a1.text(f_e.max() * 1.1, 1e3, f"Nyquist, {f_e.max():.0f} per day ({24 / f_e.max():.0f} h)", rotation=90,
        ha="left", va="center", fontsize=15, color=MUTED)
a1.set_xlim(None, f_e.max() * 1.9)
a1.set_xlabel("frequency (cycles per day)"); a1.set_ylabel("periodogram power (K², unscaled)")
a1.set_title(f"NCEP air, cell {float(cell.lat)} N {float(cell.lon)} E, 6-hourly", loc="left", fontsize=21)
a2.loglog(f_d[1:], p_d[1:], color=INK, lw=1.4)
a2.axvspan(0.03, 1.0, color=C3, alpha=0.12)
a2.plot(f_pk, p_d[f_d == f_pk][0], "o", color=C2, markersize=11)
a2.annotate(f"{f_pk:.3f} Hz ({1 / f_pk:.1f} s)", (f_pk, p_d[f_d == f_pk][0]), xytext=(14, 6),
            textcoords="offset points", fontsize=17, color=C2)
a2.set_xlim(0.01, 50); a2.set_ylim(None, 1e-12)
a2.set_xlabel("frequency (Hz)"); a2.set_ylabel("PSD ((m/s)²/Hz)")
a2.set_title("UW.RATT HHZ, 04:15–06:15 UTC, before the P wave", loc="left", fontsize=21)
fig.savefig(OUT / "which_spectrum.png", dpi=110, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote which_spectrum.png")

for k in (15, 12, 10, 9):
    f_k, p_k = signal.welch(y, fs=noise.stats.sampling_rate, nperseg=2 ** k)
    b_k = (f_k >= 0.03) & (f_k <= 1.0)
    print(f"  Welch 2^{k} samples: largest peak 0.03-1 Hz at {f_k[b_k][np.argmax(p_k[b_k])]:.3f} Hz (df {f_k[1]:.4f} Hz)")
