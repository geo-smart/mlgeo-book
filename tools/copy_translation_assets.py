"""Copy image assets referenced by translated pages into the translated trees.

Translated pages keep the English edition's relative image paths (figures are
not translated), but each language builds as its own MyST site rooted at
translations/<lang>/, so `![...](foo.png)` resolves inside that tree and 404s
unless the asset is there. This copies every referenced image from the English
edition to the matching path under each language.

Handles markdown images, HTML <img src>, and MyST {figure} directives, in .md
files and in notebook markdown cells. Paths that already resolve, remote URLs,
and non-image targets are skipped.

Usage: python tools/copy_translation_assets.py [--dry-run] [fr|es ...]
"""

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
TRANSLATIONS = ROOT / "translations"
IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

PATTERNS = (
    re.compile(r"!\[[^\]]*\]\(([^)\s]+)"),          # ![alt](path)
    re.compile(r"<img[^>]+src=[\"']([^\"']+)"),      # <img src="path">
    re.compile(r"^\s*```\{figure\}\s*(\S+)", re.M),  # ```{figure} path
    re.compile(r"^\s*:::\{figure\}\s*(\S+)", re.M),
)


def page_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        return "\n".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"
        )
    return path.read_text()


def referenced_images(path: Path) -> set:
    text = page_text(path)
    out = set()
    for pattern in PATTERNS:
        for m in pattern.findall(text):
            target = m.split("#")[0].split("?")[0]
            if target.startswith(("http://", "https://", "data:", "attachment:")):
                continue
            if Path(target).suffix.lower() in IMG_SUFFIXES:
                out.add(target)
    return out


def within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def copy_for_language(lang: str, dry_run: bool) -> int:
    lang_dir = (TRANSLATIONS / lang).resolve()
    book_dir = BOOK.resolve()
    copied = missing = 0
    for page in sorted(lang_dir.rglob("*")):
        if page.suffix not in (".md", ".ipynb") or "_build" in page.parts:
            continue
        rel_dir = page.parent.relative_to(lang_dir)
        for ref in referenced_images(page):
            if Path(ref).is_absolute():
                print(f"  SKIP absolute path {ref} (in {page.relative_to(ROOT)})")
                continue
            dst = (page.parent / ref).resolve()
            src = (book_dir / rel_dir / ref).resolve()
            # A page could reference ../../.. out of its tree; never read outside
            # the English book or write outside this language's directory.
            if not within(dst, lang_dir) or not within(src, book_dir):
                print(f"  SKIP out-of-tree {ref} (in {page.relative_to(ROOT)})")
                continue
            if dst.exists():
                continue
            if not src.exists():
                print(f"  MISSING {ref} (referenced by {page.relative_to(ROOT)})")
                missing += 1
                continue
            verb = "would copy" if dry_run else "copy"
            print(f"  {verb} {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied += 1
    print(f"{lang}: {copied} asset(s) {'to copy' if dry_run else 'copied'}, {missing} unresolved")
    return missing


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dry_run = "--dry-run" in sys.argv
    missing = sum(copy_for_language(lang, dry_run) for lang in args or ["fr", "es"])
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
