"""Slide figures: the 1945 drop in global sea surface temperature (Thompson et al., 2008).

Rebuilds the literature example from open data instead of copying the paper's
figures:
  ship reports   ICOADS R3.1 IMMA1 monthly files, 1944-1947, NOAA NCEI
                 https://www.ncei.noaa.gov/data/international-comprehensive-ocean-atmosphere/v3/archive/final-untrim/
  raw grid       ICOADS 2-degree monthly mean SST, standard trimming, no bias
                 adjustment, NOAA PSL https://downloads.psl.noaa.gov/Datasets/icoads/2degree/std/sst.mean.nc
  adjusted       HadSST.4.2.0.0 monthly global anomaly, Met Office Hadley Centre
Deck, country and measurement-method codes follow the ICOADS code tables in
glamod/cdm_reader_mapper (ICOADS.C1.DCK, ICOADS.C0.C1, ICOADS.C0.SI).

Writes three PNGs to figs/lit_sst_1945/ and prints sample records for the slide table.
Regenerate: pixi run python book/slides/2026/figscripts/sst_1945_discontinuity.py
"""
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import xarray as xr  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
CACHE = Path(pooch.os_cache("mlgeo-figscripts"))
INK, MUTED, US_C, UK_C, OTHER_C = "#26241f", "#6e675c", "#1f5f8b", "#c0392b", "#b9b2a6"
IMMA_URL = ("https://www.ncei.noaa.gov/data/international-comprehensive-ocean-atmosphere/"
            "v3/archive/final-untrim/IMMA1_R3.1.0_{y}-{m:02d}.gz")
US_DECKS = {"110", "116", "117", "195", "281", "705"}     # US Navy and US merchant decks
UK_DECKS = {"184", "194", "204", "245"}                    # Great Britain marine and Royal Navy decks
PROFILE_DECKS = {"780"}                                    # World Ocean Database profiles, not ship-surface reports
OUT = Path(__file__).resolve().parent.parent / "figs" / "lit_sst_1945"
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


# IMMA1 core section and attachment 1, 0-based slices
FIELDS = {"YR": (0, 4), "MO": (4, 6), "DY": (6, 8), "HR": (8, 12), "LAT": (12, 17), "LON": (17, 23),
          "ID": (34, 43), "C1": (43, 45), "SI": (83, 85), "SST": (85, 89), "DCK": (118, 121)}


def read_month(y, m):
    path = fetch(IMMA_URL.format(y=y, m=m), f"icoads/IMMA1_R3.1.0_{y}-{m:02d}.gz")
    rows = []
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if not line.startswith(str(y)) or len(line) < 121 or not line[85:89].strip():
                continue
            rows.append([line[a:b] for a, b in FIELDS.values()])
    df = pd.DataFrame(rows, columns=list(FIELDS))
    for c in ["YR", "MO", "DY", "LAT", "LON", "SST"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["LAT"] /= 100; df["LON"] /= 100; df["SST"] /= 10
    df["HR"] = pd.to_numeric(df["HR"], errors="coerce") / 100
    for c in ["ID", "C1", "SI", "DCK"]:
        df[c] = df[c].str.strip()
    return df[~df["DCK"].isin(PROFILE_DECKS)]


obs = pd.concat([read_month(y, m) for y in range(1944, 1948) for m in range(1, 13)], ignore_index=True)
obs["group"] = np.select([obs["DCK"].isin(US_DECKS), obs["DCK"].isin(UK_DECKS)], ["US", "UK"], "other")
obs["date"] = pd.to_datetime(dict(year=obs["YR"], month=obs["MO"], day=1))
share = obs.groupby(["date", "group"]).size().unstack(fill_value=0)
share = share.div(share.sum(axis=1), axis=0)

# Raw global-mean anomaly from the ICOADS 2-degree means (no bias adjustment)
ds = xr.open_dataset(fetch("https://downloads.psl.noaa.gov/Datasets/icoads/2degree/std/sst.mean.nc",
                           "icoads_sst_mean_std.nc"))
sst = ds["sst"].sel(time=slice("1936-01-01", "1955-12-31"))
clim = sst.groupby("time.month").mean("time")
anom = sst.groupby("time.month") - clim
w = np.cos(np.deg2rad(anom["lat"]))
raw = anom.weighted(w.fillna(0)).mean(("lat", "lon")).to_series()
raw = raw - raw.mean()

had = pd.read_csv(fetch("https://www.metoffice.gov.uk/hadobs/hadsst4/data/data/HadSST.4.2.0.0_monthly_GLOBE.csv",
                        "HadSST.4.2.0.0_monthly_GLOBE.csv"))
had.index = pd.to_datetime(dict(year=had["year"], month=had["month"], day=1))
had = had.loc["1936":"1955", "anomaly"]
had = had - had.mean()

before = raw["1944-08":"1945-07"].mean(); after = raw["1945-09":"1946-08"].mean()
before_h = had["1944-08":"1945-07"].mean(); after_h = had["1945-09":"1946-08"].mean()

# Figure 1: the series where the error shows, with who was reporting
with plt.rc_context({"font.size": 22, "axes.titlesize": 25, "axes.labelsize": 21,
                     "xtick.labelsize": 19, "ytick.labelsize": 19}):
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(15, 4.7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.7, 1], hspace=0.2))
a0.plot(raw.index, raw.rolling(3, center=True).mean(), color=INK, lw=2.4,
        label="as recorded (ICOADS)")
a0.plot(had.index, had.rolling(3, center=True).mean(), color="#1f6f6b", lw=2.4,
        label="bias-adjusted (HadSST4)")
a0.axvline(pd.Timestamp("1945-08-15"), color=MUTED, ls="--", lw=1.5)
a0.set_ylabel("°C"); a0.set_yticks([-0.2, 0.0, 0.2])
a0.legend(fontsize=18, loc="lower left", bbox_to_anchor=(0.27, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
a0.set_title("Global-mean SST", loc="left", pad=8)
sh = share.reindex(columns=["US", "UK", "other"], fill_value=0)
a1.stackplot(sh.index, sh["US"], sh["UK"], sh["other"], colors=[US_C, UK_C, OTHER_C],
             labels=["US decks", "UK decks", "other"])
a1.axvline(pd.Timestamp("1945-08-15"), color=MUTED, ls="--", lw=1.5)
a1.set_ylim(0, 1); a1.set_yticks([0.5, 1])
a1.set_xlim(pd.Timestamp("1936-01-01"), pd.Timestamp("1955-12-31"))
a1.text(pd.Timestamp("1936-06-01"), 0.5, "share of reports by deck,\ncounted 1944–1947", fontsize=18, color=MUTED, va="center")
a1.legend(fontsize=17, loc="center right", ncol=1, framealpha=0.9)
fig.savefig(OUT / "sst_1945_series.png", dpi=110, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# Figure 2: where the 1945 ship reports are, colored by temperature
land = gpd.read_file(fetch("https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip", "ne_110m_land.zip"))
y45 = obs[obs["YR"] == 1945].dropna(subset=["LAT", "LON", "SST"])
lon = np.where(y45["LON"] > 180, y45["LON"] - 360, y45["LON"])
fig, ax = plt.subplots(figsize=(16, 5.2))
ax.set_facecolor("#eef3f8")
land.plot(ax=ax, color="#f4efe6", edgecolor=MUTED, lw=0.5)
sc = ax.scatter(lon, y45["LAT"], c=y45["SST"].clip(-2, 32), cmap="RdYlBu_r", s=1, lw=0, rasterized=True)
ax.set_xlim(-180, 180); ax.set_ylim(-45, 76); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01); cb.set_label("SST (°C)")
ax.set_title(f"{len(y45):,} ship SST reports in 1945 (45°S–76°N shown)", loc="left")
fig.savefig(OUT / "sst_1945_map.png", dpi=110, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

print(f"wrote {OUT}")
print(f"records 1944-1947 (excluding profiles): {len(obs):,}; 1945: {len(y45):,}")
print(f"raw ICOADS mean, Aug 1944-Jul 1945 vs Sep 1945-Aug 1946: {before:+.2f} -> {after:+.2f} (change {after - before:+.2f} °C)")
print(f"HadSST4 same windows: {before_h:+.2f} -> {after_h:+.2f} (change {after_h - before_h:+.2f} °C)")
for per, lab in [("1945-01", "Jan-Jul 1945"), ("1945-09", "Sep-Dec 1945")]:
    sub = share.loc[per:("1945-07" if per == "1945-01" else "1945-12")].mean()
    print(lab, {k: f"{v:.0%}" for k, v in sub.items()})
print("SI codes, 1945:", y45["SI"].replace("", "blank").value_counts().head(6).to_dict())
print("C1 codes, 1945:", y45["C1"].replace("", "blank").value_counts().head(6).to_dict())
cols = ["YR", "MO", "DY", "HR", "LAT", "LON", "SST", "SI", "C1", "DCK"]
print("\nsample US record, July 1945:")
print(obs[(obs["DCK"] == "195") & (obs["MO"] == 7) & (obs["YR"] == 1945)][cols].head(3).to_string(index=False))
print("\nsample UK record, December 1945:")
print(obs[(obs["DCK"] == "245") & (obs["MO"] == 12) & (obs["YR"] == 1945)][cols].head(3).to_string(index=False))
