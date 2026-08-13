# MLGeo: Machine Learning in the Geosciences (ESS 469/569)

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://geo-smart.github.io/mlgeo-book)
[![GeoSMART Library Badge](book/img/curricula_badge.svg)](https://geo-smart.github.io/curriculum)

**Editions:** [English](https://geo-smart.github.io/mlgeo-book) ·
[Français](https://geo-smart.github.io/mlgeo-book/fr/) ·
[Español](https://geo-smart.github.io/mlgeo-book/es/)

## Citing this book

Authors and citation metadata are in [`CITATION.cff`](CITATION.cff). The book
does not have a DOI yet — the JOSE manuscript in `JOSE_PAPER/` is *in
preparation* and unreviewed, so it should not be cited as a published article.
[`docs/CITATION_AND_DOI.md`](docs/CITATION_AND_DOI.md) explains how to mint an
archival DOI through Zenodo (three steps, and `.zenodo.json` is already
prepared) and what to cite in the meantime.

## Scope

This material was developed for ESS 469/569 at the University of Washington,
and its datasets are largely US and Pacific-Northwest. Instructors adopting it
elsewhere should plan to substitute regional data and institutional context —
see [adopting this book](book/about_this_book/adopting_this_book.md). The
French and Spanish editions localize prose (examples, institutions, hazards)
but, apart from the GNSS notebook 1.7, still run on the English edition's data.

## Make this book yours

The book is CC BY 4.0 and the code is MIT. Fork it, retarget it, teach it. We
would rather you contributed improvements back, but taking it and running is a
legitimate outcome — that is what open educational resources are for.

Retargeting is driven by **personas**: short profiles of specific readers that
an AI review agent adopts while reading the book, so gaps surface as "what this
person still cannot do" rather than as generic feedback. There are two
independent axes, and they compose:

| Axis | Where | Steers |
|---|---|---|
| **Scientific audience** | [`personas/`](personas/) — 12 readers | Discipline, seniority, prior coding skill, what they must be able to do afterwards |
| **Language and culture** | [`translations/personas/`](translations/personas/) — 8 French, 5 Spanish | Register, terminology, tolerance for English jargon, regional institutions and hazards |

A Chilean hydrology master's programme is the *hydrology master's student*
persona crossed with the *Southern Cone Spanish* persona. Rewrite two or three
files for the people actually in your room, re-run the review, and act on what
disagrees.

Two things worth knowing before you rely on it. The personas are **fictional** —
a way to hold a specific reader in mind, not evidence that a real community
accepted the result; real human review is recorded separately in
[`docs/REVIEW_RECORD.md`](docs/REVIEW_RECORD.md). And persona reviews of *this*
book produced confident, wrong claims alongside the good ones, so verify
anything factual against primary sources before shipping it.

[CONTRIBUTING.md](CONTRIBUTING.md) has the full workflow, including how to
contribute personas, regional datasets, or a whole adapted edition back.

## Repository Overview

This repository is the single source of truth for the MLGeo curriculum book (2026 edition). It is edited directly: there is no separate instructor/student repository pair anymore, and the former auto-generation pipeline from `geo-smart/mlgeo-instructor` is retired. Solutions to exercises live in this repo and are rendered as collapsible/hidden cells in the published book rather than being stripped into a second repository.

The 2024 edition of the book is preserved at the [v1.0-2024-edition release](https://github.com/geo-smart/mlgeo-book/releases/tag/v1.0-2024-edition).

## Making Changes

Book content lives in `book/`. Edit the markdown pages and notebooks there, then build locally before pushing. The book is built with [Jupyter Book 2 / MyST](https://next.jupyterbook.org); configuration and table of contents live in `myst.yml`.

```sh
pixi install        # install the pinned environment (see pixi.toml)
pixi run build      # build the book (executes notebooks)
pixi run serve      # live-preview server (myst start)
```

Notebooks are executed at build time; a page with a failing cell fails the build. CI runs the same build on every pull request.

### Student response sections

Exercises marked for student response keep their solution in place, wrapped in a dropdown admonition so readers attempt the exercise before revealing the answer:

````markdown
:::{admonition} Solution
:class: dropdown
...solution here...
:::
````

## Contributing

Open a pull request against `main`. CI must pass before merge: the full book build, which executes every notebook. A link check also runs and reports in the job log, but external links flake often enough that it is advisory rather than blocking.
