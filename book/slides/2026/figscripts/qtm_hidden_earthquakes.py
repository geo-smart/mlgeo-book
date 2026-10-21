"""Slide figures: ten times more earthquakes from the same waveforms (Ross et al., 2019).

Rebuilds the literature example of lecture 1 from the open catalogs instead of
copying the paper's figures:
  QTM catalog   Quake Template Matching catalog, 2008-2017, 9.5 MAD threshold,
                SCEDC https://service.scedc.caltech.edu/ftp/QTMcatalog/qtm_final_9.5dev.hypo
                (landing page https://scedc.caltech.edu/data/qtm-catalog.html)
  SCSN catalog  Southern California Seismic Network catalog, yearly ASCII files,
                SCEDC https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/YYYY.catalog
                (event type "eq" only; quarry blasts removed)
  context       Natural Earth 10m land, state and country lines, populated places;
                PB2002 plate boundaries (Bird, 2003) via github.com/fraxen/tectonicplates
Ross, Trugman, Hauksson & Shearer (2019), Science 364, 767-771, doi:10.1126/science.aaw6888.
The magnitude of completeness values marked on the plot (0.3 for QTM, 1.7 for
SCSN) are the paper's goodness-of-fit estimates (supplement, Fig. S5); the
maximum-curvature values computed here are printed for comparison.

Writes two PNGs to figs/lit_qtm_2019/ and prints the numbers and sample records
quoted on the slide.
Regenerate: pixi run python book/slides/2026/figscripts/qtm_hidden_earthquakes.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
CACHE = Path(pooch.os_cache("mlgeo-figscripts"))
INK, MUTED, QTM_C, SCSN_C = "#26241f", "#6e675c", "#c0392b", "#1f5f8b"
QTM_URL = "https://service.scedc.caltech.edu/ftp/QTMcatalog/qtm_final_9.5dev.hypo"
SCSN_URL = "https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/{y}.catalog"
YEARS = range(2008, 2018)
LON0, LON1, LAT0, LAT1 = -121.0, -114.0, 32.0, 37.0
MC_PAPER = {"QTM": 0.3, "SCSN": 1.7}        # Ross et al. (2019) supplement, goodness-of-fit method
OUT = Path(__file__).resolve().parent.parent / "figs" / "lit_qtm_2019"
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


# --- catalogs ---------------------------------------------------------------
qtm = pd.read_csv(fetch(QTM_URL, "qtm_final_9.5dev.hypo"), sep=r"\s+")
qtm["time"] = (pd.to_datetime(qtm[["YEAR", "MONTH", "DAY"]].rename(columns=str.lower))
               + pd.to_timedelta(qtm["HOUR"] * 3600 + qtm["MINUTE"] * 60 + qtm["SECOND"], unit="s"))
qtm["in_scsn"] = qtm["EVENTID"] == qtm["TEMPLATEID"]      # template events are the SCSN events

cols = ["date", "clock", "etype", "gtype", "mag", "magtype", "lat", "lon", "depth", "quality",
        "evid", "nph", "ngrm"]
scsn = pd.concat([pd.read_csv(fetch(SCSN_URL.format(y=y), f"scsn_{y}.catalog"), sep=r"\s+",
                              comment="#", header=None, names=cols) for y in YEARS], ignore_index=True)
n_blast = int((scsn["etype"] != "eq").sum())
scsn = scsn[scsn["etype"] == "eq"].copy()
scsn["time"] = pd.to_datetime(scsn["date"] + " " + scsn["clock"], format="%Y/%m/%d %H:%M:%S.%f")

n_q, n_s, n_t = len(qtm), len(scsn), int(qtm["in_scsn"].sum())

# --- map layers ---------------------------------------------------------------
land = gpd.read_file(fetch("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
                           "ne_10m_land.zip"))
states = gpd.read_file(fetch(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces_lines.zip",
    "ne_10m_admin_1_lines.zip"))
border = gpd.read_file(fetch(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_boundary_lines_land.zip",
    "ne_10m_admin_0_boundary_lines_land.zip"))
places = gpd.read_file(fetch(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip",
    "ne_10m_populated_places_simple.zip"))
plates = gpd.read_file(fetch(
    "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json",
    "PB2002_boundaries.json"))
cities = places[places["name"].isin(["Los Angeles", "San Diego", "Mexicali"])
                & places["adm0name"].isin(["United States of America", "Mexico"])]

bins = [np.arange(LON0, LON1 + 1e-9, 0.04), np.arange(LAT0, LAT1 + 1e-9, 0.04)]
h_q, _, _ = np.histogram2d(qtm["LONGITUDE"], qtm["LATITUDE"], bins=bins)
DM = 0.1
mb_q = np.round(qtm["MAGNITUDE"] / DM) * DM
mb_s = np.round(scsn["mag"] / DM) * DM
c_q, c_s = mb_q.value_counts().sort_index(), mb_s.value_counts().sort_index()
mc_maxc = {"QTM": float(c_q.idxmax()), "SCSN": float(c_s.idxmax())}
day_q = qtm.set_index("time").resample("D").size()
day_s = scsn.set_index("time").resample("D").size()

# records shown on the slide: one hour on 2008-01-01 at the cluster that SCSN event 10297553 belongs to
REC_ID = 10297553
rec = qtm.loc[qtm["EVENTID"] == REC_ID].iloc[0]
near = np.hypot((qtm["LATITUDE"] - rec["LATITUDE"]) * 111.2,
                (qtm["LONGITUDE"] - rec["LONGITUDE"]) * 111.2 * np.cos(np.deg2rad(rec["LATITUDE"]))) < 1.0
hour = qtm[near & (qtm["time"] >= pd.Timestamp("2008-01-01 04:50")) & (qtm["time"] < pd.Timestamp("2008-01-01 05:50"))]

# --- figure 1: the domain and the frequency-magnitude distribution -----------
fig = plt.figure(figsize=(15.5, 4.9))
gs = fig.add_gridspec(1, 2, width_ratios=[6.6, 8.6], wspace=0.04)
ax, a0 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
ax.set_facecolor("#eef3f8")
land.plot(ax=ax, color="#f4efe6", edgecolor=MUTED, lw=0.6)
states.plot(ax=ax, color=MUTED, lw=0.6)
border.plot(ax=ax, color=MUTED, lw=0.8)
plates.plot(ax=ax, color=INK, lw=1.2, ls="--")
im = ax.pcolormesh(bins[0], bins[1], np.where(h_q > 0, h_q, np.nan).T, cmap="inferno_r",
                   norm=LogNorm(vmin=1, vmax=h_q.max()), rasterized=True)
for _, c in cities.iterrows():
    ax.plot(c.geometry.x, c.geometry.y, "s", ms=6, color=INK)
    dx, dy, ha, va = {"Los Angeles": (-0.12, 0.1, "right", "bottom"), "Mexicali": (0.12, 0.1, "left", "bottom"),
                      "San Diego": (0.0, -0.15, "center", "top")}[c["name"]]
    ax.text(c.geometry.x + dx, c.geometry.y + dy, c["name"], fontsize=15, color=INK, ha=ha, va=va,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
ax.plot(rec["LONGITUDE"], rec["LATITUDE"], "*", ms=22, mfc="#2bb3a3", mec=INK, mew=1.2)
ax.text(rec["LONGITUDE"] - 0.3, rec["LATITUDE"] + 0.05, "records", fontsize=15, color=INK,
        ha="right", va="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
ax.text(LON1 - 0.1, LAT1 - 0.1, "dashed: plate\nboundary (PB2002)", fontsize=13, color=INK,
        ha="right", va="top")
ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
ax.set_aspect(1 / np.cos(np.deg2rad(34.5)))
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
ax.set_title(f"QTM, {n_q:,} events", loc="left", fontsize=21)
cax = ax.inset_axes([0.04, 0.16, 0.34, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("events per 0.04° cell", fontsize=14)
cb.ax.tick_params(labelsize=13)

a0.semilogy(c_s.index, c_s.to_numpy(), "o", ms=6, color=SCSN_C, label=f"SCSN, {n_s:,} events")
a0.semilogy(c_q.index, c_q.to_numpy(), "o", ms=6, color=QTM_C, label=f"QTM, {n_q:,} events")
for k, col in [("SCSN", SCSN_C), ("QTM", QTM_C)]:
    a0.axvline(MC_PAPER[k], color=col, ls="--", lw=2)
    a0.text(MC_PAPER[k] + 0.08, 2.0, f"Mc {MC_PAPER[k]:.1f}", color=col, fontsize=17)
a0.set_xlim(-2.0, 5.0); a0.set_ylim(1, c_q.max() * 8)
a0.set_xlabel("magnitude"); a0.set_ylabel("events per 0.1 unit")
a0.yaxis.set_label_position("right"); a0.yaxis.tick_right()
a0.legend(fontsize=16, loc="upper right", frameon=False)
a0.set_title("Frequency–magnitude", loc="left", x=0.03)
fig.savefig(OUT / "qtm_map_fmd.png", dpi=110, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# --- figure 2: events per day ------------------------------------------------
fig, a1 = plt.subplots(figsize=(15.5, 3.6))
a1.semilogy(day_q.index, day_q.clip(lower=0.8).to_numpy(), color=QTM_C, lw=0.8, label="QTM")
a1.semilogy(day_s.index, day_s.clip(lower=0.8).to_numpy(), color=SCSN_C, lw=0.8, label="SCSN")
a1.set_ylim(5, max(day_q.max(), day_s.max()) * 2.5)
a1.set_xlim(pd.Timestamp("2008-01-01"), pd.Timestamp("2018-01-01"))
a1.set_ylabel("events per day")
a1.set_title(f"Events per day: median {day_q.median():.0f} in QTM, {day_s.median():.0f} in SCSN",
             loc="left", x=0.01)
emc = qtm.loc[qtm["MAGNITUDE"].idxmax()]
a1.annotate(f"M{emc['MAGNITUDE']:.1f} El Mayor–Cucapah, {emc['time']:%Y-%m-%d}",
            xy=(emc["time"], day_q.max()), xytext=(pd.Timestamp("2011-01-01"), day_q.max() * 1.25),
            fontsize=16, color=INK, va="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
leg = a1.legend(fontsize=17, loc="lower right", ncol=2, framealpha=0.9)
for line in leg.get_lines():
    line.set_linewidth(4)
fig.text(0.125, -0.03, "SCEDC catalogs: QTM (9.5 MAD) and SCSN (earthquakes only), "
         f"{YEARS[0]}–{YEARS[-1]}, retrieved {pd.Timestamp.now(tz='UTC'):%Y-%m-%d} · "
         "Mc: Ross et al. (2019)", fontsize=13, color=MUTED)
fig.savefig(OUT / "qtm_daily.png", dpi=110, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# --- numbers and records for the slide --------------------------------------
print(f"QTM events {n_q:,}; of which SCSN template events {n_t:,}; new detections {n_q - n_t:,}")
print(f"SCSN earthquakes {YEARS[0]}-{YEARS[-1]}: {n_s:,} (quarry blasts removed: {n_blast:,}); ratio {n_q / n_s:.2f}")
print(f"median magnitude: QTM new {qtm.loc[~qtm.in_scsn, 'MAGNITUDE'].median():.2f}, SCSN {scsn['mag'].median():.2f}")
print(f"Mc maximum curvature: QTM {mc_maxc['QTM']:.1f}, SCSN {mc_maxc['SCSN']:.1f}; paper GFT: {MC_PAPER}")
print(f"median events per day: QTM {day_q.median():.0f}, SCSN {day_s.median():.0f}; "
      f"mean interevent time QTM {(qtm.time.max() - qtm.time.min()).total_seconds() / (n_q - 1):.0f} s")
print(f"busiest day: QTM {day_q.idxmax():%Y-%m-%d} {day_q.max():,}, SCSN {day_s.idxmax():%Y-%m-%d} {day_s.max():,}")
print(f"largest event: {emc['time']} M{emc['MAGNITUDE']} {emc['LATITUDE']:.3f} {emc['LONGITUDE']:.3f}")
show = ["time", "LATITUDE", "LONGITUDE", "DEPTH", "MAGNITUDE", "MAGP5", "MAGP95", "TEMPLATEID",
        "STACKCC", "MAD", "in_scsn"]
print("median stack CC and CC/MAD of new detections:", qtm.loc[~qtm.in_scsn, "STACKCC"].median(),
      (qtm.loc[~qtm.in_scsn, "STACKCC"] / qtm.loc[~qtm.in_scsn, "MAD"]).median())
print(f"records cluster: {len(hour)} QTM events within 1 km, 04:50-05:50 UTC, {int(hour.in_scsn.sum())} in SCSN")
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(hour[show].to_string(index=False))
print(f"wrote {OUT / 'qtm_map_fmd.png'} and {OUT / 'qtm_daily.png'}")
