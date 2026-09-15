# Book figures, regenerated

Every script here draws one or more figures in `book/img/` from scratch with
matplotlib. They replaced images that had been copied from textbooks and blogs
(Géron's *Hands-On ML*, ISLR, Lilian Weng, Gregory Gundersen, Ahmet Taspinar,
Ronneberger et al. 2015) without a licence that allowed redistribution under
this book's CC BY 4.0. The audit that found them is in
`docs/ai-logs/2026-09-15-figure-ip-audit.md`.

Rules for a figure in this directory:

- Drawn from the concept as the lesson teaches it, not traced from the original.
- Labels use the lesson's own notation (`w`, `θ`, `x_1`, fold numbers, …).
- Output keeps the filename and extension the book already references, so no
  page or translation needs a link change; run `tools/copy_translation_assets.py`
  after regenerating so the FR/ES copies refresh.
- One palette: teal `#116b66`, orange `#b3402a`, grey `#6e675c`,
  fills `#cfe3ee` / `#d9ead3` / `#f9cb9c`. Default sans font. PNG at 200 dpi.

Regenerate everything:

    for f in tools/figures/*.py; do pixi run python "$f"; done
