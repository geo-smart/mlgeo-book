# GeoSMART Curriculum Jupyter Book (ESS 469/569)

[![Deploy](https://github.com/geo-smart/mlgeo-instructor/actions/workflows/deploy.yaml/badge.svg)](https://github.com/geo-smart/mlgeo-instructor/actions/workflows/deploy.yaml)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://geo-smart.github.io/mlgeo-instructor)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geo-smart/mlgeo-instructor/HEAD?urlpath=lab)
[![GeoSMART Library Badge](book/img/curricula_badge.svg)](https://geo-smart.github.io/curriculum)
[![Student Version](book/img/student_version_badge.svg)](https://geo-smart.github.io/mlgeo-book/)

## Repository Overview

This repository stores configuration for GeoSMART curriculum content, specifically the teacher version of the book. Only this version of the book should ever be edited, as the student version is automatically generated on push by github actions.

## Making Changes

Edit the book content by modifying the `_config.yml`, `_toc.yml` and `*.ipynb` files in the `book` directory. The book is hosted on Github Pages and will be automatically updated on push, and the student book will also be created automatically on push.

To modify the exact differences between this book and the student book, edit `.github/workflows/clean_book.py`. When you push, a github action will clone the repo and run this python file which modifies certain parts of `*.ipynb` file contents, then pushes to the student repo. To edit the student repo's README, edit `STUDENT_README.md`. The Github Actions workflow also automatically replaces `README.md` with `STUDENT_README.md` in the student repo.

### `Student Response Sections`

One modifications made by the `clean_book.py` workflow is to clear sections marked for student response. Code cells marked for student response may contain code in the teacher version of the book, but will have their code removed and replaced with a TODO comment in the student version.

To mark a code cell to be cleared, insert a markdown cell directly preceding it with the following content:

````markdown
```{admonition} Student response section
This section is left for the student to complete.
```
````

## Serving Locally

Activate the `curriculum_book` conda environment (or any conda environment that has the necessary jupyter book dependencies). Navigate to the root folder of the curriculum book repository in anaconda prompt, then run `python server.py`.

On startup, the server will run `jb build book` to build all changes to the notebook and create the compiled HTML. The server code can take a `--no-build` flag (or `--nb` shorthand) if you don't want to build any changes you've made to the notebooks. In the case that you don't want to build changes made to the notebooks, you can just run `python serer.py --nb` from any terminal with python installed.
