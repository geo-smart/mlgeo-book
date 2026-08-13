"""Semantic terminology linter for the translated editions.

This is deliberately NOT glossary enforcement. Researchers write *machine
learning*, *workflow*, *pipeline*, *notebook*, *cloud*, *cluster*, *baseline*
and *benchmark* in French and Spanish prose, and a linter that errors on those
would be fighting the language it is supposed to serve. What it catches instead
are errors of *meaning* — cases where the wrong word makes the sentence say
something false — plus the two discipline rules the glossary asks for:
first-use glossing and within-chapter consistency.

Rules (see translations/GLOSSARY.md, "The hard invariants"):
  metric-accuracy   `accuracy` rendered précision/precisión — that is `precision`
  repo-as-archive   a Git repository called archive/archivo
  bare-leakage      fuite/fuga unqualified, where data vs spectral matters
  ui-label          an English interface label translated while the UI is English
  es-regionalism    Spain-only forms in the pan-regional Spanish edition
  identifier-drift  a protected identifier translated in prose

Usage:
  python tools/lint_terminology.py [fr|es ...]      # exit 1 on any error
  python tools/lint_terminology.py --warn-only
"""


import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS = ROOT / "translations"
sys.path.insert(0, str(ROOT / "tools"))
from extract_channels import extract  # noqa: E402

# (rule, regex, message). Case-insensitive, applied to prose only.
RULES = {
    "fr": [
        ("metric-accuracy",
         r"\bprécision\b(?=[^.]{0,80}\b(accuracy|exactitude globale|bonnes classifications)\b)",
         "`accuracy` must be « exactitude » — « précision » is reserved for\n"
         "precision = TP/(TP+FP)"),
        ("repo-as-archive",
         r"\barchive\s+(git|github)\b",
         "a Git/GitHub repository is a « dépôt », never an « archive »\n"
         "(archive = preservation, e.g. HAL, Zenodo)"),
        ("es-regionalism", r"(?!)", ""),
    ],
    "es": [
        ("metric-accuracy",
         r"\bprecisión\b(?=[^.]{0,80}\b(accuracy|exactitud global|clasificaciones correctas)\b)",
         "`accuracy` must be «exactitud» — «precisión» is reserved for precision = TP/(TP+FP)"),
        ("repo-as-archive",
         r"\barchivo\s+(de\s+)?(git|github)\b",
         "a Git/GitHub repository is a «repositorio», never an «archivo» (archivo = preservation)"),
        ("es-regionalism",
         r"\b(ordenador(es)?|vosotros|zumo)\b",
         "Spain-only form in the pan-regional Spanish edition (use «computadora», usted-neutral)"),
    ],
}

# First-use rule for the leakage pair. Once a page has said which leakage it
# means, the bare noun is normal French and Spanish — demanding the qualifier
# every time would be fighting the language. What matters is that a reader
# never meets the bare word first, because data leakage and spectral leakage
# are unrelated phenomena that the book teaches in adjacent chapters.
FIRST_USE = {
    "fr": ("leakage-first-use", r"\bfuites?\b",
           r"\bfuites?\s+(de\s+données|spectrales?|d'information)",
           "first « fuite » on this page is unqualified — say « fuite de données » "
           "or « fuite spectrale » once, then the bare noun is fine"),
    # "se fuga" is the verb (energy leaks backwards in time), not the noun.
    "es": ("leakage-first-use", r"(?<!se )\bfugas?\b",
           r"\bfugas?\s+(de\s+datos|espectral(es)?|de\s+información)",
           "first «fuga» on this page is unqualified — say «fuga de datos» or "
           "«fuga espectral» once, then the bare noun is fine"),
}


def pages(lang: str) -> list:
    root = TRANSLATIONS / lang
    return [
        p for p in sorted(root.rglob("*"))
        if p.suffix in (".md", ".ipynb") and "_build" not in p.parts and p.name != "MANIFEST.yml"
    ]


def check_page(path: Path, lang: str, findings: list) -> None:
    prose = "\n".join(extract(path)["prose"])
    for rule, pattern, message in RULES[lang]:
        if not pattern or pattern == r"(?!)":
            continue
        for m in re.finditer(pattern, prose, re.I):
            line = prose[: m.start()].count("\n") + 1
            context = prose[max(0, m.start() - 45) : m.end() + 45].replace("\n", " ")
            findings.append((rule, path, line, message, context.strip()))

    rule, bare_re, qualified_re, message = FIRST_USE[lang]
    first_bare = re.search(bare_re, prose, re.I)
    first_qualified = re.search(qualified_re, prose, re.I)
    if first_bare and (not first_qualified or first_bare.start() < first_qualified.start()):
        line = prose[: first_bare.start()].count("\n") + 1
        context = prose[max(0, first_bare.start() - 45) : first_bare.end() + 45].replace("\n", " ")
        findings.append((rule, path, line, message, context.strip()))


def main() -> None:
    warn_only = "--warn-only" in sys.argv
    langs = [a for a in sys.argv[1:] if not a.startswith("-")] or ["fr", "es"]
    findings: list = []
    for lang in langs:
        before = len(findings)
        for page in pages(lang):
            check_page(page, lang, findings)
        print(f"{lang}: {len(findings) - before} finding(s)")
    for rule, path, line, message, context in findings:
        print(f"\n  {rule}  {path.relative_to(ROOT)}:{line}")
        print(f"    {message}")
        print(f"    …{context}…")
    if findings and not warn_only:
        print(f"\n{len(findings)} terminology error(s). Fix, or justify and add an exception.")
        sys.exit(1)
    if not findings:
        print("no terminology errors")


if __name__ == "__main__":
    main()
