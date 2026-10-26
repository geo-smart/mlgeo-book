"""Slide figure: analyst labels for earthquakes and blasts in the Pacific Northwest catalog.

USGS ComCat event service, 42-49.5N, 125.5-116.5W, 2015-01-01 to 2025-01-01:
eventtype=earthquake (M >= 1) and eventtype=explosion (the Pacific Northwest
Seismic Network labels quarry and mine blasts "explosion"). Panel A: map;
panel B: local hour of day by label. The day/night contrast is the method of
Wiemer and Baer (2000, BSSA 90, 525-530, doi:10.1785/0119990104).

Regenerate: pixi run python book/slides/2026/figscripts/lec12_blasts_vs_quakes.py
"""
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
CACHE = Path(pooch.os_cache("mlgeo-figscripts"))
INK, MUTED, EQ, BL = "#26241f", "#6e675c", "#1f6f6b", "#d9822b"
BOX = dict(minlatitude=42.0, maxlatitude=49.5, minlongitude=-125.5, maxlongitude=-116.5)


def comcat(eventtype, extra, name):
    path = CACHE / name
    if not path.exists():
        frames = []
        for y in range(2015, 2025):
            r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=120, params=dict(
                format="csv", starttime=f"{y}-01-01", endtime=f"{y + 1}-01-01", eventtype=eventtype,
                orderby="time-asc", **BOX, **extra))
            r.raise_for_status()
            frames.append(pd.read_csv(pd.io.common.StringIO(r.text)))
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return pd.read_csv(path)


eq = comcat("earthquake", dict(minmagnitude=1), "comcat_pnw_eq_m1_2015_2024.csv")
bl = comcat("explosion", {}, "comcat_pnw_explosion_2015_2024.csv")
for d in (eq, bl):
    d["local"] = pd.to_datetime(d["time"], utc=True).dt.tz_convert("America/Los_Angeles")
    d["hour"] = d["local"].dt.hour
    d["weekday"] = d["local"].dt.dayofweek
work = lambda d: ((d["hour"] >= 8) & (d["hour"] < 17)).mean()  # noqa: E731

land = gpd.read_file(pooch.retrieve("https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip",
                                    known_hash=None, fname="ne_50m_land.zip", path=CACHE, progressbar=False))
states = gpd.read_file(pooch.retrieve(
    "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces_lines.zip",
    known_hash=None, fname="ne_50m_admin_1_states_provinces_lines.zip", path=CACHE, progressbar=False))

fig = plt.figure(figsize=(17.5, 8.6))
gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.25], wspace=0.12)
a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
a0.set_facecolor("#eef3f8")
land.plot(ax=a0, color="#f4efe6", edgecolor=MUTED, lw=0.6)
states.plot(ax=a0, color=MUTED, lw=0.6)
a0.scatter(eq["longitude"], eq["latitude"], s=3, color=EQ, lw=0, alpha=0.6, label=f"earthquake (M ≥ 1), {len(eq):,}")
a0.scatter(bl["longitude"], bl["latitude"], s=6, color=BL, lw=0, alpha=0.8, label=f"explosion, {len(bl):,}")
a0.set_xlim(BOX["minlongitude"], BOX["maxlongitude"]); a0.set_ylim(BOX["minlatitude"], BOX["maxlatitude"])
a0.set_aspect(1 / np.cos(np.deg2rad(45.75)))
a0.set_xticks([]); a0.set_yticks([]); a0.set_xlabel(""); a0.set_ylabel("")
for name, (x, y) in {"WA": (-118.2, 47.6), "OR": (-118.2, 43.4)}.items():
    a0.text(x, y, name, fontsize=17, color=MUTED, ha="center")
a0.legend(loc="lower left", fontsize=15, markerscale=4, framealpha=0.9)
a0.set_title("Analyst labels, 2015–2024", loc="left")

bins = np.arange(25) - 0.5
for d, col, lab in [(eq, EQ, "earthquake"), (bl, BL, "explosion")]:
    a1.hist(d["hour"], bins=bins, density=True, histtype="step", lw=3, color=col, label=f"{lab}: {work(d):.0%} 08:00–17:00")
a1.axvspan(7.5, 16.5, color="#f1e3cf", alpha=0.5, zorder=0)
a1.set_xlim(-0.5, 23.5); a1.set_xticks([0, 4, 8, 12, 16, 20])
a1.set_xlabel("local hour of day (Pacific time)"); a1.set_ylabel("fraction of events per hour")
a1.legend(fontsize=16, loc="upper left", frameon=False)
a1.set_title("When the events happen", loc="left")
fig.text(0.125, -0.03, "USGS ComCat, 42–49.5°N, 125.5–116.5°W, 2015–2024; labels assigned by the Pacific Northwest Seismic Network. "
         "Day/night method: Wiemer & Baer (2000).", fontsize=13, color=MUTED)

out = Path(__file__).resolve().parent.parent / "figs" / "lec12_supervision" / "blasts_vs_quakes.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")
print(f"earthquakes {len(eq)}, explosions {len(bl)}; share 08-17 local: eq {work(eq):.3f}, bl {work(bl):.3f}")
print(f"weekday share (Mon-Fri): eq {(eq['weekday'] < 5).mean():.3f}, bl {(bl['weekday'] < 5).mean():.3f}")
print(f"depth median km: eq {eq['depth'].median():.1f}, bl {bl['depth'].median():.1f}; mag median: eq {eq['mag'].median():.2f}, bl {bl['mag'].median():.2f}")
print(f"majority-class baseline accuracy: {max(len(eq), len(bl)) / (len(eq) + len(bl)):.3f}")
cols = ["local", "latitude", "longitude", "depth", "mag", "magType", "type"]
print(eq.sort_values("local").iloc[[100, 2000]][cols].to_string(index=False))
print(bl.sort_values("local").iloc[[100, 2000]][cols].to_string(index=False))
