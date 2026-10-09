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

## Set up your course environment (students)

All course code runs in one [pixi](https://pixi.sh) environment, pinned for
macOS (Apple silicon) and Linux. On Windows, work inside WSL2 or a GitHub
Codespace, as described in
[Homework 1](book/Chapter1-GettingStarted/1.9_workbench_setup_hw1.md). Install
pixi first:

```sh
curl -fsSL https://pixi.sh/install.sh | sh   # or: brew install pixi
pixi --version                               # restart the terminal if not found
```

Then pick the manifest that matches where your work lives:

| You work in | Use this `pixi.toml` | Where it comes from |
|---|---|---|
| A clone or fork of this book | `pixi.toml` at the top of this repository | Already there |
| Your own repository, started blank (e.g. `MLGEO2026_UWNETID`) | [`environments/student/pixi.toml`](environments/student/pixi.toml) and its `pixi.lock` | Copy both into the top of your repository |

### Option A: clone or fork the book

```sh
git clone https://github.com/geo-smart/mlgeo-book.git   # or your fork's URL
cd mlgeo-book
pixi install
pixi run jupyter lab
```

The first `pixi install` downloads a few gigabytes. Fork the repository on
GitHub first if you want to push your own commits; you cannot push to
`geo-smart/mlgeo-book`.

### Option B: start from a blank repository

Create the repository on GitHub (tick "Add a README file"), clone it, and copy
the student manifest and its lockfile into the top level:

```sh
git clone https://github.com/<your-username>/MLGEO2026_UWNETID.git
cd MLGEO2026_UWNETID
curl -fsSLO https://raw.githubusercontent.com/geo-smart/mlgeo-book/main/environments/student/pixi.toml
curl -fsSLO https://raw.githubusercontent.com/geo-smart/mlgeo-book/main/environments/student/pixi.lock
pixi install
pixi run lab                                  # starts JupyterLab
git add pixi.toml pixi.lock
git commit -m "Add MLGeo course environment"
git push
```

If you already have a clone of the book next to your repository,
`cp ../mlgeo-book/environments/student/pixi.{toml,lock} .` does the same as the
two `curl` lines.

Do not copy the root `pixi.toml` into a blank repository. It installs the
book's `mlgeo_synth` package from the book's own folder, so outside the book
every `pixi run` fails with "does not appear to be a Python project", and its
build and check tasks expect the book's `book/` and `tools/` folders. The
student manifest pins the same package versions as the book's lockfile,
installs `mlgeo_synth` from a snapshot of the book on GitHub (about 190 MB,
downloaded once and then cached), drops the book-building tools, and adds a
`pixi run lab` task.

Check that the environment works:

```sh
pixi run python -c "import numpy, torch, sklearn, obspy, mlgeo_synth; print('environment ok')"
```

From then on, change the environment only through pixi: `pixi add <package>`
for conda-forge packages, `pixi add --pypi <package>` for PyPI-only ones, and
commit `pixi.toml` and `pixi.lock` together every time. You may rename
`name = "mlgeo-student"` under `[workspace]` to your repository's name.

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

Personas have no version number of their own — the book's release tag versions
them, and each file records which edition it was `written-for` and when it was
`last-run`. `python tools/persona_status.py` reports which have fallen behind.

[CONTRIBUTING.md](CONTRIBUTING.md) has the full workflow, including how to
contribute personas, regional datasets, or a whole adapted edition back.

## Repository Overview

This repository is the single source of truth for the MLGeo curriculum book (2026 edition). It is edited directly: there is no separate instructor/student repository pair anymore, and the former auto-generation pipeline from `geo-smart/mlgeo-instructor` is retired. Solutions to exercises live in this repo and are rendered as collapsible/hidden cells in the published book rather than being stripped into a second repository.

The 2024 edition of the book is preserved at the [v1.0-2024-edition release](https://github.com/geo-smart/mlgeo-book/releases/tag/v1.0-2024-edition).

## Making Changes

Book content lives in `book/`. Edit the markdown pages and notebooks there, then build locally before pushing. The book is built with [Jupyter Book 2 / MyST](https://next.jupyterbook.org); configuration and table of contents live in `myst.yml`.

```sh
pixi install         # install the pinned environment (see pixi.toml)
pixi run build-fast  # render from committed outputs — seconds, no execution
pixi run serve-fast  # live preview, no execution
pixi run build       # execute every notebook, then build — minutes
pixi run serve       # live preview with execution
pixi run check       # the quality gates CI runs
```

Use `build-fast` while editing prose, links, or the table of contents: notebooks
ship with their outputs committed and that is what the site renders, so it shows
exactly what readers see. Use the executing variants when you changed code.
`build-fr` and `build-es` build the translated editions, which never execute.

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
