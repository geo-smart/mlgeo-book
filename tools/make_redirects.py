"""Generate redirect stubs at the old Jupyter Book v1 URLs.

The 2024 edition published pages at nested paths like
Chapter1-GettingStarted/1.1_open_reproducible_science.html; the MyST site uses
flat slugs (/open-reproducible-science). External links — including the
published JOSE paper — point at the old paths, so after `myst build` this
script drops a meta-refresh stub at every old path inside book/_build/html.

Run from the repo root after a build: python tools/make_redirects.py
"""

import glob
import json
import os

HTML = "book/_build/html"
CONTENT = "book/_build/site/content"

# Old pages with no same-named source file in the 2026 edition -> best target.
LEGACY = {
    "about_this_book/about_this_book.html": "index",
    "about_this_book/1.10_MLGEO_final_project.html": "mlgeo-finalproject",
    "Chapter1-GettingStarted/1.20_MLGEO_Final_Project.html": "mlgeo-finalproject",
    "Chapter4-DeepLearning/mlgeo_4.9_LLM4Geo.html": "llms-to-agents",
    "Chapter4-DeepLearning/mlgeo_4.8_NAS.html": "mlgeo-4-7-modeltraining",
    # Chapter 4 renumbered in Nov 2026 (graph networks became 4.5): the training
    # lab 4.5 -> 4.7, its rubric 4.8 -> 4.7, the training clinic 4.9 -> 4.7 (in the
    # table of contents from be27650 to ed72530), PINNs 4.7 -> 4.8, forecasting 4.10 -> 4.9.
    # Old 2024 paths, then the old 2026 flat slugs (MyST serves each page as
    # <slug>/index.html). None of these paths is a page of the new numbering.
    "Chapter4-DeepLearning/mlgeo_4.5_ModelTraining.html": "mlgeo-4-7-modeltraining",
    "Chapter4-DeepLearning/mlgeo_4.7_PINN.html": "mlgeo-4-8-pinn",
    "Chapter4-DeepLearning/mlgeo_4.8_dl_rubric.html": "mlgeo-4-7-dl-rubric",
    "Chapter4-DeepLearning/mlgeo_4.10_timeseriesforecast.html": "mlgeo-4-9-timeseriesforecast",
    "mlgeo-4-5-modeltraining/index.html": "mlgeo-4-7-modeltraining",
    "mlgeo-4-7-pinn/index.html": "mlgeo-4-8-pinn",
    "mlgeo-4-8-dl-rubric/index.html": "mlgeo-4-7-dl-rubric",
    "mlgeo-4-9-training-clinic/index.html": "mlgeo-4-7-training-clinic",
    "mlgeo-4-10-timeseriesforecast/index.html": "mlgeo-4-9-timeseriesforecast",
}

STUB = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={target}">
<link rel="canonical" href="{target}">
<title>Redirecting</title>
</head>
<body>
<p>This page moved. <a href="{target}">Continue to the new location.</a></p>
</body>
</html>
"""


def main():
    mapping = {}
    for p in glob.glob(os.path.join(CONTENT, "*.json")):
        d = json.load(open(p))
        loc, slug = d.get("location", ""), d.get("slug", "")
        if loc and slug:
            old = loc.lstrip("/").rsplit(".", 1)[0] + ".html"
            mapping[old] = slug
    mapping.update(LEGACY)

    n = 0
    for old, slug in sorted(mapping.items()):
        depth = old.count("/")
        # MyST writes <slug>/index.html, so <slug>.html would 404: link the folder.
        target = "../" * depth + ("" if slug == "index" else slug + "/")
        if slug == "index":
            target = "../" * depth or "./"
        out = os.path.join(HTML, old)
        if os.path.exists(out):  # never overwrite a real page
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(STUB.format(target=target))
        n += 1
    print(f"wrote {n} redirect stubs into {HTML}")


if __name__ == "__main__":
    main()
