"""Slide figure (lecture 4): daily east positions of GNSS stations P395 and P563.

Lecture-scale replot of the 1.7 notebook's first figure (figs/1.7_get_geodetic_gnss/fig_01.png,
which is extracted from the notebook at 10 pt). Values are fetched from NGL at run time:
  daily positions   NGL tenv3, IGS20 solution (Blewitt et al., 2018)
  velocities        NGL MIDAS, IGS20 frame (midas.IGS.txt) (Blewitt et al., 2016)

Regenerate: pixi run python book/slides/2026/figscripts/gnss_east_series.py
"""
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pooch  # noqa: E402

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 23, "axes.titleweight": "bold",
    "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
})
NGL = "https://geodesy.unr.edu"
CACHE = pooch.os_cache("mlgeo-figscripts")
INK, MUTED, GRID = "#26241f", "#6e675c", "#e4dfd6"
COLORS = {"P395": "#2a78d6", "P563": "#eb6834"}     # dataviz reference slots 1 and 2, as on the map slide
OUT = Path(__file__).resolve().parent.parent / "figs" / "lec04_gnss_series"
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, name):
    return pooch.retrieve(url, known_hash=None, fname=name, path=CACHE, progressbar=False)


def midas_igs():
    """MIDAS columns: 6 number of solutions, 9 east velocity (m/yr), 12 its uncertainty."""
    out = {}
    for line in open(fetch(f"{NGL}/gps_timeseries/IGS20/midas/midas.IGS.txt", "midas.IGS.txt")):
        p = line.split()
        if p and p[0] in COLORS:
            out[p[0]] = (int(p[5]), float(p[8]) * 1000, float(p[11]) * 1000)
    return out


vel = midas_igs()
fig, ax = plt.subplots(figsize=(16, 5.6))
for sta, c in COLORS.items():
    df = pd.read_csv(fetch(f"{NGL}/gps_timeseries/IGS20/tenv3/IGS20/{sta}.tenv3", f"ngl_tenv3_{sta}.tenv3"),
                     sep=r"\s+")
    east = (df["_e0(m)"] + df["__east(m)"]) * 1000
    east -= east.iloc[0]
    ax.plot(df["yyyy.yyyy"], east, color=c, lw=1.2)
    n, ve, se = vel[sta]
    ax.text(df["yyyy.yyyy"].iloc[-1] + 0.3, east.iloc[-1] + 15, f"{sta}\n{ve:+.1f} mm/yr".replace("-", "\u2212"),
            color=c, fontsize=19, fontweight="bold", va="bottom")
    print(f"{sta}: {len(df):,} daily positions, {df['yyyy.yyyy'].iloc[0]:.2f}-{df['yyyy.yyyy'].iloc[-1]:.2f}; "
          f"MIDAS IGS20 east {ve:+.1f} ± {se:.1f} mm/yr (MIDAS n={n})")
ax.axhline(0, color=MUTED, lw=0.8)
ax.grid(color=GRID, lw=0.8)
ax.set_xlim(2005.5, 2030.2); ax.set_ylim(-540, 30)
ax.set_xticks(range(2006, 2027, 4))
ax.set_ylabel("east position (mm)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Daily east position relative to the first day, NGL IGS20 solution", loc="left")
fig.text(0.01, -0.02, f"NGL tenv3 and MIDAS files, retrieved {dt.date.today().isoformat()}",
         fontsize=13, color=MUTED)
fig.savefig(OUT / "p395_p563_east_slide.png", dpi=110, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {OUT}")
