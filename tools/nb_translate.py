"""Notebook translation helpers: move ONLY markdown cells, never code/outputs.

Translated notebooks must keep code cells and executed outputs byte-identical
to the English edition (translations/README.md ground rules). This tool makes
that guarantee structural: translators never touch the notebook JSON.

  extract  — dump markdown cells to a compact JSON file a translator can edit:
             python tools/nb_translate.py extract book/ChapterX/foo.ipynb /tmp/foo.md.json
  inject   — copy the English notebook and replace markdown cell sources with
             the translated ones, writing the translated notebook:
             python tools/nb_translate.py inject book/ChapterX/foo.ipynb /tmp/foo.md.json translations/fr/ChapterX/foo.ipynb

The JSON file is a list of {"i": <cell index>, "source": <markdown text>}.
Inject refuses to run if indices or cell counts do not match the source.
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


def inject(src: Path, translated: Path, dst: Path) -> None:
    nb = json.loads(src.read_text())
    cells = json.loads(translated.read_text())
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
    if len(sys.argv) < 4 or sys.argv[1] not in ("extract", "inject"):
        sys.exit(__doc__)
    if sys.argv[1] == "extract":
        extract(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        if len(sys.argv) != 5:
            sys.exit(__doc__)
        inject(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))


if __name__ == "__main__":
    main()
