"""Slide figure: gridded geoscience datasets placed by grid spacing and extent.

Each point is one dataset from the lecture 6 gallery or its candidate list, with
the grid spacing and extent stated in the figure caption or text of its paper
(all CC BY 4.0; full citations in figs/lit_gridded_gallery/SOURCES.md). Color is
the data type, marker shape the dimensionality. Diagonal lines mark the number
of cells across the dataset (extent / spacing).

Regenerate: pixi run python book/slides/2026/figscripts/gridded_data_scales.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({"font.size": 17, "axes.titlesize": 22, "axes.titleweight": "bold"})
INK, MUTED, GRID = "#26241f", "#6e675c", "#d9d4ca"
TYPE_COLOR = {                       # dataviz reference categorical slots 1-4, fixed order
    "observation": "#2a78d6",
    "derived from observations": "#eb6834",
    "reanalysis": "#1baf7a",
    "model simulation": "#eda100",
}
DIM_MARKER = {"2D": "o", "2D + channels": "D", "3D space": "s", "space + time": "^", "ensemble": "*"}
GLOBAL = 4.0e7                       # Earth's circumference, m

# name, grid spacing (m), extent (m), type, dimensionality, label offset (points)
DATA = [
    ("micro-CT, sandstone and carbonate", 2.25e-6, 1.69e-2, "observation", "3D space", (8, -16)),
    ("micro-CT, sandstone under load", 7.91e-6, 2.44e-2, "observation", "space + time", (10, 4)),
    ("thin-section element maps", 4e-6, 2.5e-2, "observation", "2D + channels", (-10, 12)),
    ("hyperspectral core scan", 7e-5, 10.5, "observation", "2D + channels", (10, -4)),
    ("fiber-optic DAS on a glacier", 4.0, 1.0e3, "observation", "space + time", (-12, -2)),
    ("airborne lidar elevation", 3.0, 5.5e3, "derived from observations", "2D", (10, 2)),
    ("radar ice thickness", 10.0, 3.0e3, "derived from observations", "2D", (12, -14)),
    ("seismic tomography, Alps", 1.6e4, 7.0e5, "derived from observations", "3D space", (10, -14)),
    ("altimetry sea level", 2.5e4, 3.9e6, "derived from observations", "space + time", (12, 2)),
    ("InSAR velocity, Tibet", 1.0e3, 1.4e6, "derived from observations", "2D + channels", (-12, 0)),
]
# Global products (extent = Earth): numbered on the chart, named in a key.
# Stack factor separates markers that would overlap; true extent is GLOBAL for all.
GLOBAL_DATA = [
    ("km-scale ocean, ICON", 1.25e3, 1.0, "model simulation", "space + time"),
    ("sea surface temperature", 5.0e3, 1.6, "derived from observations", "space + time"),
    ("km-scale aerosol, nextGEMS", 5.0e3, 1 / 1.6, "model simulation", "2D + channels"),
    ("ERA5-Land", 9.0e3, 1.6, "reanalysis", "space + time"),
    ("ocean model, 0.1°", 1.0e4, 1 / 1.6, "model simulation", "space + time"),
    ("soil moisture", 2.5e4, 1.6, "derived from observations", "space + time"),
    ("ML weather ensemble", 2.8e4, 1.0, "model simulation", "ensemble"),
    ("ERA5", 3.1e4, 1 / 1.6, "reanalysis", "space + time"),
    ("GRACE water storage", 5.0e4, 1.0, "derived from observations", "space + time"),
    ("ocean model, 1°", 1.0e5, 1.0, "model simulation", "space + time"),
]

fig, ax = plt.subplots(figsize=(15, 8.6))
xs = np.logspace(-7, 7, 50)
for n, lab, x_lab in [(1e2, "100 cells across", 3e2), (1e4, "10,000 cells", 3e-5), (1e6, "1 million", 1e-4)]:
    ax.plot(xs, xs * n, color=GRID, lw=1.5, ls="--", zorder=0)
    ax.text(x_lab * 1.3, x_lab * n, lab, color=MUTED, fontsize=13, rotation=33, rotation_mode="anchor",
            ha="left", va="bottom")

for name, dx, ext, kind, dim, off in DATA:
    ax.scatter(dx, ext, s=210 if dim != "ensemble" else 460, marker=DIM_MARKER[dim], color=TYPE_COLOR[kind],
               edgecolor="white", linewidth=2, zorder=3)
    ha = "right" if off[0] < 0 else "left"
    ax.annotate(name, (dx, ext), xytext=off, textcoords="offset points", fontsize=14, color=INK, ha=ha,
                va="center", zorder=4)
key = []
for i, (name, dx, stack, kind, dim) in enumerate(GLOBAL_DATA, start=1):
    y = GLOBAL * stack
    ax.scatter(dx, y, s=260 if dim != "ensemble" else 520, marker=DIM_MARKER[dim], color=TYPE_COLOR[kind],
               edgecolor="white", linewidth=2, zorder=3)
    off = (14, 0) if name == "ML weather ensemble" else (0, 13 if stack >= 1 else -13)
    ax.annotate(str(i), (dx, y), xytext=off, textcoords="offset points", fontsize=13,
                fontweight="bold", color=INK, ha="center", va="center", zorder=4)
    spacing = f"{dx / 1e3:g} km"
    key.append(f"{i:>2}  {name} ({spacing})")
ax.text(2e-6, 2.5e8, "Global products, extent = Earth:\n" + "\n".join(key), fontsize=14, color=INK, va="top",
        ha="left", linespacing=1.35, family="DejaVu Sans")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-6, 3e6); ax.set_ylim(1e-3, 5e8)
ax.set_xlabel("grid spacing (m)"); ax.set_ylabel("extent of the dataset (m)")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(colors=INK)
ticks = {1e-6: "1 µm", 1e-3: "1 mm", 1: "1 m", 1e3: "1 km", 1e6: "1000 km"}
ax.set_xticks(list(ticks)); ax.set_xticklabels(list(ticks.values()))
yt = {1e-3: "1 mm", 1: "1 m", 1e3: "1 km", 1e6: "1000 km", GLOBAL: "Earth"}
ax.set_yticks(list(yt)); ax.set_yticklabels(list(yt.values()))
ax.grid(which="major", color="#efebe4", lw=0.8)
# (slide title carries the title)

type_handles = [Line2D([], [], marker="o", ls="", ms=12, color=c, label=k) for k, c in TYPE_COLOR.items()]
dim_handles = [Line2D([], [], marker=m, ls="", ms=12 if m != "*" else 16, color=MUTED, label=k)
               for k, m in DIM_MARKER.items()]
leg1 = ax.legend(handles=type_handles, title="data type", loc="lower right", bbox_to_anchor=(1.0, 0.0),
                 frameon=False, fontsize=14, title_fontsize=14)
ax.add_artist(leg1)
ax.legend(handles=dim_handles, title="dimensions", loc="lower right", bbox_to_anchor=(0.72, 0.0),
          frameon=False, fontsize=14, title_fontsize=14)
# Caveats (spacing sources, global stacking) are stated in the slide credit line and SOURCES.md.

out = Path(__file__).resolve().parent.parent / "figs" / "lit_gridded_gallery" / "gridded_data_scales.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.08)
print(f"wrote {out}")
