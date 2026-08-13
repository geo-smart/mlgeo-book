"""Notebook translation helpers: move ONLY markdown cells, never code/outputs.

Translated notebooks must keep code cells and executed outputs byte-identical
to the English edition (translations/README.md ground rules). This tool makes
that guarantee structural: translators never touch the notebook JSON.

  extract  — dump markdown cells to a compact JSON file a translator can edit:
             python tools/nb_translate.py extract book/ChapterX/foo.ipynb /tmp/foo.md.json
  inject   — copy the English notebook and replace markdown cell sources with
             the translated ones, writing the translated notebook:
             python tools/nb_translate.py inject \
                 book/ChapterX/foo.ipynb /tmp/foo.md.json \
                 translations/fr/ChapterX/foo.ipynb

The JSON file is a list of {"i": <cell index>, "source": <markdown text>}.
Inject refuses to run if indices or cell counts do not match the source.

Data-localized notebooks (1.7 in both editions) intentionally carry different
code and outputs from the English source. Injecting those from the English
base would revert the localization, so inject refuses when it detects one.
Pass --base-dst to rebuild from the existing translation instead, which is
what you want when editing a localized notebook's prose.
"""

import json
import sys
from pathlib import Path


def extract(src: Path, out: Path) -> None:
    nb = json.loads(src.read_text())
    cells = [
        {"i": i, "source": "".join(c["source"])}
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "markdown"
    ]
    out.write_text(json.dumps(cells, ensure_ascii=False, indent=1))
    print(f"{src}: {len(cells)} markdown cells -> {out}")


def inject(src: Path, translated: Path, dst: Path, base_dst: bool = False) -> None:
    cells = json.loads(translated.read_text())
    # Guard against silently un-localizing a data-localized notebook (1.7 in
    # both editions): if dst already exists and its code differs from the
    # English source, dst is the localized truth and rebuilding from src would
    # revert the localization. check_translations.py cannot catch this — it
    # skips code comparison precisely for localized entries.
    if dst.exists() and not base_dst:
        old = json.loads(dst.read_text())
        old_code = [c for c in old.get("cells", []) if c["cell_type"] == "code"]
        src_code = [
            c for c in json.loads(src.read_text())["cells"] if c["cell_type"] == "code"
        ]
        if old_code != src_code:
            sys.exit(
                f"refusing to inject: {dst} has code/outputs that differ from "
                f"{src} — it looks data-localized, and rebuilding from the "
                f"English source would revert that. Re-run with --base-dst to "
                f"use the existing translation as the structural base."
            )
    nb = json.loads((dst if base_dst else src).read_text())
    md_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"]
    got_idx = [c["i"] for c in cells]
    if md_idx != got_idx:
        sys.exit(f"index mismatch: source md cells {md_idx} != translated {got_idx}")
    for c in cells:
        nb["cells"][c["i"]]["source"] = c["source"].splitlines(keepends=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"{dst}: wrote {len(cells)} translated markdown cells; code/outputs untouched")


def main() -> None:
    argv = [a for a in sys.argv[1:] if a != "--base-dst"]
    base_dst = "--base-dst" in sys.argv
    if len(argv) < 3 or argv[0] not in ("extract", "inject"):
        sys.exit(__doc__)
    if argv[0] == "extract":
        extract(Path(argv[1]), Path(argv[2]))
    else:
        if len(argv) != 4:
            sys.exit(__doc__)
        inject(Path(argv[1]), Path(argv[2]), Path(argv[3]), base_dst=base_dst)


if __name__ == "__main__":
    main()
