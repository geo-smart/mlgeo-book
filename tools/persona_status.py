"""Report which personas are drifting behind the book.

Personas are NOT versioned separately. A git tag pins the whole tree, so the
book's version — currently v2.0-2026-edition — is also the personas' version,
and citing the book cites them. What a persona does need is provenance: which
edition it was written for, and when it was last used to review the book.

This reports the gap. A persona whose `written-for` is an older edition, or
that has not been re-run while the book moved underneath it, is a persona whose
findings are stale. It cannot tell you whether a persona is *good* — only
whether it has looked at the book lately.

Usage:
  python tools/persona_status.py            # table
  python tools/persona_status.py --stale    # exit 1 if any persona predates the current edition
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERSONA_DIRS = [ROOT / "personas", ROOT / "translations" / "personas"]


def current_edition() -> str:
    out = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return out.stdout.strip() or "(no tag)"


def commits_touching_book_since(tag: str) -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", f"{tag}..HEAD", "--", "book/"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return int(out.stdout.strip() or 0) if out.returncode == 0 else -1


def frontmatter(path: Path) -> dict:
    text = path.read_text()
    if not text.startswith("---\n"):
        return {}
    block = text.split("---\n", 2)[1]
    fields = {}
    for line in block.splitlines():
        m = re.match(r"^(\w[\w-]*):\s*(.+?)(?:\s+#.*)?$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def persona_files() -> list:
    files = []
    for d in PERSONA_DIRS:
        # 00_* is supporting material (learning outcomes), not a reader.
        files += [
            p for p in sorted(d.rglob("*.md"))
            if re.match(r"^\d\d_", p.name) and not p.name.startswith("00")
        ]
    return files


def main() -> None:
    edition = current_edition()
    drift = commits_touching_book_since(edition)
    print(f"current edition: {edition}")
    if drift > 0:
        print(f"book/ has moved {drift} commit(s) since that tag\n")
    else:
        print("book/ unchanged since that tag\n")

    stale = []
    print(f"{'persona':<52} {'type':<18} {'written-for':<22} last-run")
    print("-" * 104)
    for p in persona_files():
        fm = frontmatter(p)
        written = fm.get("written-for", "—")
        kind = fm.get("type", "—")
        if fm.get("language"):
            kind = f"{kind} ({fm['language']})"
        if written != edition:
            stale.append(p)
            written += "  ⚠"
        print(f"{str(p.relative_to(ROOT)):<52} {kind:<18} {written:<22} {fm.get('last-run', '—')}")

    if stale:
        print(f"\n{len(stale)} persona(s) written for an older edition than {edition}.")
        print("Re-read them against the current book, revise what no longer fits, and")
        print("update `written-for` and `last-run` — see CONTRIBUTING.md.")
        if "--stale" in sys.argv:
            sys.exit(1)
    else:
        print(f"\nAll personas are current for {edition}.")


if __name__ == "__main__":
    main()
