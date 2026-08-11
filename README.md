# MLGeo: Machine Learning in the Geosciences (ESS 469/569)

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://geo-smart.github.io/mlgeo-book)
[![GeoSMART Library Badge](book/img/curricula_badge.svg)](https://geo-smart.github.io/curriculum)

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
