"""Regenerate a translation MANIFEST from the files present on disk.

Walks translations/<lang>/ for translated pages, pairs each with its English
source under book/, and records the source's last commit
(git log -1 --format=%H -- <source>) so CI can flag stale translations.

Preserves the language's existing header (everything above `files:`) and the
hand-written `localization:`/`localized_data:`/`note:` fields of entries that
already exist, so a regeneration never drops curatorial notes.

Schemas differ by language and are preserved:
  fr: `- translation: fr/<path>` ;  es: `- path: <path>`

Usage: python tools/gen_manifest.py [fr|es ...]
"""

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS = ROOT / "translations"
KEEP_FIELDS = ("localization", "localized_data", "note")
SUFFIXES = (".md", ".ipynb")


def source_commit(source: Path) -> str:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(source.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def existing_entries(manifest_path: Path, lang: str) -> dict:
    if not manifest_path.exists():
        return {}
    data = yaml.safe_load(manifest_path.read_text()) or {}
    out = {}
    for entry in data.get("files", []) or []:
        rel = entry["path"] if "path" in entry else entry["translation"].split("/", 1)[1]
        out[rel] = entry
    return out


def header_text(manifest_path: Path) -> str:
    """Everything above the `files:` key, verbatim."""
    if not manifest_path.exists():
        return ""
    lines = manifest_path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("files:"):
            return "".join(lines[:i])
    return "".join(lines)


def translated_files(lang_dir: Path) -> list:
    files = [
        p.relative_to(lang_dir)
        for p in sorted(lang_dir.rglob("*"))
        if p.suffix in SUFFIXES and p.is_file() and "_build" not in p.parts
    ]
    return [f for f in files if f.name != "MANIFEST.yml"]


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def generate(lang: str) -> None:
    lang_dir = TRANSLATIONS / lang
    manifest_path = lang_dir / "MANIFEST.yml"
    prior = existing_entries(manifest_path, lang)
    lines = [header_text(manifest_path), "files:\n"]
    missing_source = []
    for rel in translated_files(lang_dir):
        source = ROOT / "book" / rel
        if not source.exists():
            missing_source.append(str(rel))
            continue
        key = "translation" if lang == "fr" else "path"
        value = f"{lang}/{rel}" if lang == "fr" else str(rel)
        lines.append(f"  - {key}: {value}\n")
        lines.append(f"    source: book/{rel}\n")
        lines.append(f"    source_commit: {source_commit(source)}\n")
        for field in KEEP_FIELDS:
            if field in prior.get(str(rel), {}):
                lines.append(f"    {field}: {yaml_quote(prior[str(rel)][field])}\n")
    manifest_path.write_text("".join(lines))
    entries = sum(1 for line in lines if line.startswith(("  - path:", "  - translation:")))
    print(f"{lang}: wrote {entries} entries to {manifest_path.relative_to(ROOT)}")
    for m in missing_source:
        print(f"  WARN no English source for {m} — omitted")


def main() -> None:
    for lang in sys.argv[1:] or ["fr", "es"]:
        generate(lang)


if __name__ == "__main__":
    main()
