"""Spell-check the prose channel of every page, in every edition.

Running a spellchecker over raw `.ipynb` JSON is useless — base64 images,
executed model reports, station codes and identifiers all read as
misspellings. This extracts the prose channel first (see extract_channels.py),
writes it to a scratch directory, and runs codespell over that, so the hits it
reports are hits in text a reader actually sees.

codespell is used rather than a dictionary spellchecker because it only flags
known misspellings, which keeps false positives near zero on a book full of
domain vocabulary — but it is not zero: real author surnames and domain
acronyms (MAPE, Bodin) still appear, hence IGNORE below.

**English only.** codespell's dictionary is English, so pointing it at the
French or Spanish editions flags thousands of ordinary words (« fonction » →
"function", « blocs » → "blocks") and finds nothing real. Those editions need a
French/Spanish spellchecker — hunspell with `fr_FR`/`es_ES` dictionaries — which
is not part of this environment; `tools/lint_terminology.py` covers the
translation errors that matter most in the meantime.

Usage:
  python tools/check_prose_typos.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from extract_channels import extract  # noqa: E402

# Correct words codespell mistakes for typos. Keep this list short and
# justified — every entry is a check we have chosen to stop performing.
IGNORE = [
    "MAPE",   # mean absolute percentage error, a real metric
    "Bodin",  # Paul Bodin, co-author of the PNW-ML dataset citation
    "ND",     # dimension shorthand ("2D/ND array")
    "nd",
]


def prose_tree(roots: list, dest: Path) -> int:
    n = 0
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.suffix not in (".md", ".ipynb") or "_build" in p.parts:
                continue
            prose = "\n".join(extract(p)["prose"])
            if not prose.strip():
                continue
            flat = str(p.relative_to(ROOT)).replace("/", "__") + ".txt"
            (dest / flat).write_text(prose)
            n += 1
    return n


def main() -> None:
    if not shutil.which("codespell"):
        sys.exit("codespell not found. Install it: pipx install codespell")
    roots = [ROOT / "book"]
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        n = prose_tree(roots, dest)
        print(f"checking prose of {n} page(s) in {', '.join(r.name for r in roots)}")
        out = subprocess.run(
            ["codespell", "--ignore-words-list", ",".join(IGNORE), str(dest)],
            capture_output=True, text=True,
        )
        findings = [
            line.replace(str(dest) + "/", "").replace("__", "/")
            for line in out.stdout.splitlines() if line.strip()
        ]
    for line in findings:
        print("  " + line)
    if findings:
        print(f"\n{len(findings)} possible typo(s) in prose.")
        sys.exit(1)
    print("no prose typos found")


if __name__ == "__main__":
    main()
