"""Build the committed North Pacific coastline used by the map in notebook 2.9.

Source: Natural Earth 1:50m physical coastline (ne_50m_coastline), public domain,
https://www.naturalearthdata.com. The script reads the local copy cartopy caches
(~/.local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp) or,
if that is missing, the zip on the Natural Earth CDN. It keeps the coast between
172° W and 114° W, 40° N and 64° N (the Alaska Peninsula to Washington), simplifies
it to 0.02° and writes snapshots/naturalearth_npacific_coast.csv with the columns of
naturalearth_pnw_map.csv (kind, part, lon, lat, name).

Regenerate: pixi run python book/Chapter2-DataManipulation/snapshots/make_naturalearth_npacific_coast.py
"""
from pathlib import Path

import pandas as pd
import pyogrio
from shapely.geometry import box

HERE = Path(__file__).resolve().parent
LOCAL = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp"
REMOTE = "/vsizip//vsicurl/https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"
BBOX = (-172.0, 40.0, -114.0, 64.0)

gdf = pyogrio.read_dataframe(str(LOCAL) if LOCAL.exists() else REMOTE)
clip = gdf.geometry.intersection(box(*BBOX)).simplify(0.02)
rows, part = [], 0
for geom in clip:
    if geom.is_empty:
        continue
    for line in getattr(geom, "geoms", [geom]):
        for lon, lat in line.coords:
            rows.append(("coastline", part, round(lon, 4), round(lat, 4), ""))
        part += 1
out = HERE / "naturalearth_npacific_coast.csv"
pd.DataFrame(rows, columns=["kind", "part", "lon", "lat", "name"]).to_csv(out, index=False)
print(f"wrote {out.name}: {part} lines, {len(rows)} points")
