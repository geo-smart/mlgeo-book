"""Split a page into the five channels that need different linguistic rules.

Running an ordinary spellchecker over `.ipynb` JSON produces noise: base64
images, executed model reports, station codes, and identifiers all look like
misspelled words. The fix is not a bigger ignore-list, it is to stop treating a
notebook as one undifferentiated blob.

Channels:
  prose       markdown cells, and .md text outside code — general language rules apply
  comments    `#` comments inside code cells — light rules; jargon expected
  strings     user-facing string literals in code — translate only when the
              notebook is deliberately localized and re-executed
  identifiers names, arguments, attributes, imports — drift here is an ERROR
  outputs     executed results — never linted, never translated

Usage:
  python tools/extract_channels.py <path> [--channel prose] [--json]
"""

import argparse
import ast
import json
import re
from pathlib import Path

FENCE_RE = re.compile(r"^```.*?^```", re.M | re.S)
INDENTED_RE = re.compile(r"^(?: {4,}|\t).*$", re.M)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
DIRECTIVE_FENCE_RE = re.compile(r"^```\{[^}]+\}", re.M)


def md_prose(text: str) -> str:
    """Markdown minus code. Directive fences hold prose, so keep their bodies."""
    out = []
    pos = 0
    for m in FENCE_RE.finditer(text):
        out.append(text[pos : m.start()])
        block = m.group(0)
        if DIRECTIVE_FENCE_RE.match(block):
            out.append("\n".join(block.split("\n")[1:-1]))
        pos = m.end()
    out.append(text[pos:])
    joined = "".join(out)
    joined = INDENTED_RE.sub("", joined)
    return INLINE_CODE_RE.sub("", joined)


def code_channels(source: str) -> tuple:
    """(comments, user-facing strings, identifiers) from one code cell."""
    comments = re.findall(r"#(.*)$", source, re.M)
    strings, identifiers = [], []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return comments, strings, identifiers
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Only strings a reader could see; short keys and dunders are API.
            if len(node.value) > 3 and " " in node.value:
                strings.append(node.value)
        elif isinstance(node, ast.Name):
            identifiers.append(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.append(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            identifiers.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.append(node.name)
        elif isinstance(node, ast.alias):
            identifiers.append(node.asname or node.name)
    return comments, strings, identifiers


def extract(path: Path) -> dict:
    ch = {"prose": [], "comments": [], "strings": [], "identifiers": [], "outputs": []}
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                ch["prose"].append(md_prose(src))
            elif cell["cell_type"] == "code":
                comments, strings, idents = code_channels(src)
                ch["comments"] += comments
                ch["strings"] += strings
                ch["identifiers"] += idents
                ch["outputs"].append(json.dumps(cell.get("outputs", [])))
    else:
        ch["prose"].append(md_prose(path.read_text()))
    return ch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--channel", choices=["prose", "comments", "strings", "identifiers", "outputs"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    ch = extract(Path(args.path))
    if args.channel:
        ch = {args.channel: ch[args.channel]}
    if args.json:
        print(json.dumps(ch, ensure_ascii=False, indent=1))
    else:
        for name, items in ch.items():
            print(f"--- {name} ({len(items)})")
            for item in items:
                if item.strip():
                    print(item if name == "prose" else f"  {item}")


if __name__ == "__main__":
    main()
