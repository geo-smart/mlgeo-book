#!/usr/bin/env python3
"""Extract output figures from executed book notebooks into slide asset dirs.

Usage: pixi run python tools/extract_figures.py <notebook.ipynb> [...]
Writes PNGs to book/slides/2026/figs/<notebook-stem>/fig_NN.png, numbered in
cell order, so slide decks reference stable paths and re-extraction after a
notebook re-run refreshes every figure deterministically.
"""
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "book" / "slides" / "2026" / "figs"
PAD = 8  # px kept around the trimmed content


def _trim(png_bytes: bytes) -> bytes:
    """Crop uniform border whitespace so the plot fills more of the slide."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    bg = Image.new("RGB", im.size, im.getpixel((0, 0)))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        left, top, right, bottom = bbox
        bbox = (max(0, left - PAD), max(0, top - PAD),
                min(im.width, right + PAD), min(im.height, bottom + PAD))
        im = im.crop(bbox)
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def extract(nb_path: Path) -> int:
    nb = json.loads(nb_path.read_text())
    out_dir = FIGS / nb_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                n += 1
                png = base64.b64decode("".join(data["image/png"]))
                (out_dir / f"fig_{n:02d}.png").write_bytes(_trim(png))
    print(f"{nb_path.name}: {n} figures (trimmed) -> {out_dir.relative_to(ROOT)}")
    return n


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        extract(Path(arg))
