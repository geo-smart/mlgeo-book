"""Slide figure: one catalog-cleaning decision, the magnitude cut, changes the b-value.

Pacific Northwest earthquakes from the USGS ComCat event service, fetched at
run time one year per request (the service caps a request at 20,000 events).
The Gutenberg-Richter b-value is estimated with the Aki-Utsu maximum-likelihood
formula twice: once on every event in the table, and once on events at or above
the magnitude of completeness Mc. Mc is the maximum-curvature estimate plus
0.2, following Woessner and Wiemer (2005, BSSA, doi:10.1785/0120040007).
  events     USGS ComCat, https://earthquake.usgs.gov/fdsnws/event/1/
  land       Natural Earth 50m land, states and provinces lines

Regenerate: pixi run python book/slides/2026/figscripts/pnw_catalog_completeness.py
"""
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402
import requests  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
})
CACHE = pooch.os_cache("mlgeo-figscripts")
INK, MUTED, ACCENT, GOOD = "#26241f", "#6e675c", "#c0392b", "#1f6f6b"
BOX = dict(minlatitude=42.0, maxlatitude=49.5, minlongitude=-125.5, maxlongitude=-116.5)
YEARS = range(2010, 2025)
DM = 0.1


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


def comcat_year(year):
    path = Path(CACHE) / f"comcat_pnw_{year}.csv"
    if not path.exists():
        r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=120, params=dict(
            format="csv", starttime=f"{year}-01-01", endtime=f"{year + 1}-01-01",
            minmagnitude=0, eventtype="earthquake", orderby="time-asc", **BOX))
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    return pd.read_csv(path)


def b_aki(m, mc):
    """Aki-Utsu maximum-likelihood b-value for magnitudes binned at DM."""
    m = m[m >= mc - 1e-9]
    return np.log10(np.e) / (m.mean() - (mc - DM / 2)), len(m)


cat = pd.concat([comcat_year(y) for y in YEARS], ignore_index=True)
cat = cat.dropna(subset=["mag"])
cat["mbin"] = np.round(cat["mag"] / DM) * DM

counts = cat["mbin"].value_counts().sort_index()
mc_maxc = counts.idxmax()
mc = round(mc_maxc + 0.2, 1)
m_all = cat["mbin"].min()
b_all, n_all = b_aki(cat["mbin"].to_numpy(), m_all)
b_mc, n_mc = b_aki(cat["mbin"].to_numpy(), mc)

land = gpd.read_file(fetch("https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip",
                           "ne_50m_land.zip"))
states = gpd.read_file(fetch(
    "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces_lines.zip",
    "ne_50m_admin_1_states_provinces_lines.zip"))

fig = plt.figure(figsize=(24.5, 7.0))
gs = fig.add_gridspec(1, 2, width_ratios=[5.6, 15.5], wspace=0.10)
ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

ax0.set_facecolor("#eef3f8")
land.plot(ax=ax0, color="#f4efe6", edgecolor=MUTED, lw=0.6)
states.plot(ax=ax0, color=MUTED, lw=0.6)
below = cat["mbin"] < mc
ax0.scatter(cat.loc[~below, "longitude"], cat.loc[~below, "latitude"], s=4, color=GOOD,
            lw=0, label=f"M ≥ {mc:.1f}")
ax0.scatter(cat.loc[below, "longitude"], cat.loc[below, "latitude"], s=3, color="#e0a458",
            lw=0, alpha=0.7, label=f"M < {mc:.1f}")
ax0.set_xlim(BOX["minlongitude"], BOX["maxlongitude"]); ax0.set_ylim(BOX["minlatitude"], BOX["maxlatitude"])
ax0.set_aspect(1 / np.cos(np.deg2rad(45.75)))
ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_xlabel(""); ax0.set_ylabel("")
for name, (x, y) in {"WA": (-118.2, 47.6), "OR": (-118.2, 43.4)}.items():
    ax0.text(x, y, name, fontsize=17, color=MUTED, ha="center")
ax0.legend(loc="lower left", fontsize=15, markerscale=4, framealpha=0.9)
ax0.set_title(f"{len(cat):,} events, {YEARS[0]}–{YEARS[-1]}", loc="left", fontsize=20)

mags = counts.index.to_numpy()
cum = counts[::-1].cumsum()[::-1]
ax1.semilogy(mags, counts.to_numpy(), "o", ms=5, color="#b9b2a6", label="per 0.1 magnitude bin")
ax1.semilogy(cum.index, cum.to_numpy(), "s", ms=5, color=INK, label="cumulative, M or larger")
for b, m0, n, col, lab in [(b_all, m_all, n_all, ACCENT, f"all events kept: b = {b_all:.2f}"),
                           (b_mc, mc, n_mc, GOOD, f"M ≥ Mc = {mc:.1f} only: b = {b_mc:.2f}")]:
    mm = np.linspace(m0, mags.max(), 50)
    ax1.semilogy(mm, n * 10 ** (-b * (mm - m0)), color=col, lw=3, label=lab)
ax1.axvline(mc, color=GOOD, ls="--", lw=1.5)
ax1.set_ylim(0.7, cum.max() * 2)
ax1.set_xlabel("magnitude")
ax1.set_ylabel("number of earthquakes")
ax1.legend(fontsize=15, loc="upper right")
ax1.set_title("Frequency–magnitude distribution", loc="left", x=0.02)

fig.text(0.125, -0.03, "USGS ComCat, 42–49.5°N, 125.5–116.5°W, M ≥ 0, retrieved "
         f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d} · b-value: Aki–Utsu maximum likelihood · "
         "Mc: maximum curvature + 0.2 (Woessner & Wiemer, 2005)", fontsize=12, color=MUTED)

out = Path(__file__).resolve().parent.parent / "figs" / "2.4_dataframes_prep" / "pnw_catalog_completeness_slide.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")
print(f"n={len(cat)}  min M={m_all:.1f}  Mc(maxc)={mc_maxc:.1f}  Mc={mc:.1f}  "
      f"b_all={b_all:.3f} (n={n_all})  b_mc={b_mc:.3f} (n={n_mc})  below Mc: {below.mean():.1%}")
print("magType counts:", cat["magType"].value_counts().to_dict())
print("network counts:", cat["net"].value_counts().head(5).to_dict())
