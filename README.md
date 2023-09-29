# GeoSMART Curriculum Jupyter Book Site (ESS 469/569)

[![Deploy](https://github.com/geo-smart/curriculum-book/actions/workflows/deploy.yaml/badge.svg)](https://github.com/geo-smart/curriculum-book/actions/workflows/deploy.yaml)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://geo-smart.github.io/curriculum-book)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geo-smart/curriculum-book/HEAD?urlpath=lab)
[![GeoSMART Library Badge](book/img/curricula_badge.svg)](https://geo-smart.github.io/curriculum)
[![Student Version](book/img/student_version_badge.svg)](https://geo-smart.github.io/curriculum-book-student/)

## About

This repository stores configuration for GeoSMART curriculum content, specifically the teacher version of the book. Only this version of the book should ever be edited, as the student version is automatically generated on push.

## Serving

Activate the `curriculum_book` conda environment (or any conda environment that has the necessary jupyter book dependencies). Navigate to the root folder of the curriculum book repository in anaconda prompt. Run `python server.py`.

The server code can take a `--no-build` flag (or `--nb` shorthand) if you don't want to build any changes you've made to the notebooks, and this can just be run from a regular terminal.
