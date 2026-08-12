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
    "Chapter4-DeepLearning/mlgeo_4.8_NAS.html": "mlgeo-4-5-modeltraining",
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
        target = "../" * depth + ("" if slug == "index" else slug + ".html")
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
