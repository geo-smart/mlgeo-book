# Changelog

All notable changes to this template will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Calendar Versioning](https://calver.org)

## 2.0.0 (2026-08)
- 2026 edition: rewrite of the front matter and Chapter 1 for the agentic-AI era — course AI-use policy (new section 1.8), AI-use disclosure, AI review and critique in the final project, and audience-translation and downstream-impact deliverables
- Toolchain moved to Jupyter Book 2 / MyST built with pixi (`pixi install; pixi run build`); single repository, no separate instructor/student versions
- Deep learning is PyTorch-only (Keras/TensorFlow removed)
- Added the `mlgeo_synth` synthetic data library for physically motivated synthetic data with documented ground truth
- New chapters 5-7: reproducible workflows in the agent era, building and evaluating agents, use cases with audience translation and downstream impact
- Fair-evaluation thread (honest test sets, in-domain vs out-of-domain generalization, baselines) woven through the curriculum

## 2022-06-14
- Created skeleton repository from uwhackweek/jupyterbook-template

## 2022-09-28
- Adapted Use Case book into new curriculum book with improved structure

## 2022-11-15
- Set up github actions and personal access token for CI with student version