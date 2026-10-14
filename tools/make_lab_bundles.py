"""Build one zip per Session 7 track so students can download a track without git.

Each zip unpacks to a folder laid out like book/Chapter2-DataManipulation/: the
track notebook, its two helper modules, the start page, the templates and the
course data the track reads, under snapshots/. The deploy workflow runs this
script and publishes the zips with the site, so they always match main.

Usage: python tools/make_lab_bundles.py --out book/_build/html/downloads
"""

import argparse
import zipfile
from pathlib import Path

CH2 = Path(__file__).resolve().parents[1] / "book" / "Chapter2-DataManipulation"

# Track -> (notebook, module prefix, snapshot name prefixes the track reads)
TRACKS = {
    "volcano": ("lab_s07_volcano.ipynb", "lab_s07_volcano_",
                ["comcat_msh_", "msh_", "make_msh_", "comcat_puget_", "make_comcat_puget_"]),
    "soil_moisture": ("lab_s07_soil_moisture.ipynb", "lab_s07_soil_",
                      ["uscrn_", "make_uscrn_"]),
    "land_surface_model": ("lab_s07_land_surface_model.ipynb", "lab_s07_lsm_",
                           ["cpc_", "make_cpc_", "uscrn_", "make_uscrn_"]),
    "planetary": ("lab_s07_planetary.ipynb", "lab_s07_planetary_",
                  ["nea_", "tess", "exoplanets_", "make_exoplanets_"]),
}
SHARED_SNAPSHOTS = ["naturalearth_pnw_map.csv"]


def bundle(track, out_dir):
    notebook, module_prefix, prefixes = TRACKS[track]
    root = f"lab_s07_{track}"
    files = [CH2 / notebook, CH2 / "lab_s07_start_here.md"]
    files += sorted(CH2.glob(f"{module_prefix}*.py"))
    files += sorted((CH2 / "lab_s07_templates").glob("*.md"))
    snaps = CH2 / "snapshots"
    files += sorted(p for p in snaps.iterdir()
                    if p.is_file() and (p.name in SHARED_SNAPSHOTS or p.name.startswith(tuple(prefixes))))
    out = out_dir / f"{root}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f"{root}/{f.relative_to(CH2).as_posix()}")
    return out, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        out, n = bundle(track, args.out)
        print(f"{out.name}: {n} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
