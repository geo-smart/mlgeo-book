"""Integrity checker for the translated editions.

Verifies, for every entry in translations/*/MANIFEST.yml:
  - the translated file and its English source both exist;
  - .ipynb: code cells (source AND outputs) are identical to the English
    edition, and markdown cell counts match — unless the entry carries a
    `localization:` field (e.g. the GNSS notebook 1.7), in which case only
    structure is checked;
  - .md: the sequence of fenced code blocks is identical to the English
    source (code is never translated).
Also checks that every file in each translation's myst.yml toc has a
MANIFEST entry. Exit code 1 on any failure — suitable for CI.

Usage: python tools/check_translations.py [fr|es ...]
"""

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS = ROOT / "translations"

FENCE_RE = re.compile(r"^```[^\n]*\n.*?^```\s*$", re.M | re.S)


def code_fences(text: str) -> list:
    return FENCE_RE.findall(text)


def check_ipynb(src: Path, dst: Path, localized: bool, errors: list) -> None:
    a = json.loads(src.read_text())["cells"]
    b = json.loads(dst.read_text())["cells"]
    if [c["cell_type"] for c in a] != [c["cell_type"] for c in b]:
        errors.append(f"{dst}: cell structure differs from {src}")
        return
    if localized:
        return
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca["cell_type"] != "code":
            continue
        if ca["source"] != cb["source"]:
            errors.append(f"{dst}: code cell {i} source differs from English")
        if ca.get("outputs", []) != cb.get("outputs", []):
            errors.append(f"{dst}: code cell {i} outputs differ from English")


def plain_fences(text: str) -> list:
    # Directive fences ({note}, {warning}...) hold prose and are translated —
    # translations may even add one (e.g. the typographic-conventions note).
    # Only plain code fences must match the English edition byte for byte.
    return [f for f in code_fences(text) if not f.lstrip("`").startswith("{")]


def check_md(src: Path, dst: Path, errors: list) -> None:
    fa, fb = plain_fences(src.read_text()), plain_fences(dst.read_text())
    if len(fa) != len(fb):
        errors.append(f"{dst}: {len(fb)} code fences vs {len(fa)} in English source")
        return
    for i, (a, b) in enumerate(zip(fa, fb)):
        if a != b:
            errors.append(f"{dst}: code fence {i} differs from English source")


def toc_files(myst_yml: Path) -> list:
    cfg = yaml.safe_load(myst_yml.read_text())

    def walk(entries):
        for e in entries:
            if "file" in e:
                yield e["file"]
            yield from walk(e.get("children", []))

    return list(walk(cfg["project"]["toc"]))


def check_language(lang: str) -> list:
    errors: list = []
    manifest = yaml.safe_load((TRANSLATIONS / lang / "MANIFEST.yml").read_text())
    covered = set()
    for entry in manifest["files"]:
        # fr uses `translation: fr/<path>`; es uses `path: <path>` relative to
        # the language directory. Accept both.
        rel = entry["path"] if "path" in entry else entry["translation"].split("/", 1)[1]
        dst = TRANSLATIONS / lang / rel
        src = ROOT / entry["source"]
        covered.add(rel)
        if not dst.exists():
            errors.append(f"{dst}: translation listed in MANIFEST but missing")
            continue
        if not src.exists():
            errors.append(f"{src}: English source listed in MANIFEST but missing")
            continue
        if dst.suffix == ".ipynb":
            localized = "localization" in entry or "localized_data" in entry
            check_ipynb(src, dst, localized, errors)
        elif dst.suffix == ".md":
            check_md(src, dst, errors)
    myst_yml = TRANSLATIONS / lang / "myst.yml"
    if myst_yml.exists():
        for f in toc_files(myst_yml):
            if f not in covered:
                errors.append(f"{lang}/myst.yml toc file {f} has no MANIFEST entry")
    return errors


def main() -> None:
    langs = sys.argv[1:] or ["fr", "es"]
    failures = []
    for lang in langs:
        errs = check_language(lang)
        print(f"{lang}: {'OK' if not errs else f'{len(errs)} problem(s)'}")
        failures += errs
    for e in failures:
        print(f"  FAIL {e}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
