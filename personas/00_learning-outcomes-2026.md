# MLGeo 2026 — Target Learning Outcomes (Phase 0 draft)

Working rubric for the 2026 audience review. Every persona reviewer judges the book
against these outcomes. If reviewers ask for a public outcomes page, this draft
graduates into `about_this_book/` after Marine's edits.

Provenance: revised from the ESS 469/569 2024 syllabus learning objectives
(`book/about_this_book/MLGeo_2024.pdf`, p.1–2), rewritten at measurable Bloom levels
and extended for the 2026 edition (fair evaluation, physics-generated synthetics,
agentic AI, uncertainty, audience translation).

## Who this is for

PhD students and senior undergraduates across the geosciences and adjacent
engineering and computer science, plus the professors, postdocs, lab scientists, and
program staff who train or hire them. A completer can take a raw, messy geoscience
data stream to a defensible, reproducible, honestly evaluated model — and can say
exactly which parts an AI assistant did.

## Outcomes

By the end of the book, students will be able to:

| # | Outcome | Bloom | Chapters | Assessed by |
|---|---------|-------|----------|-------------|
| 1 | **Describe** the canonical uses of ML across the geosciences (discovery, automation, signal processing, emulation, forecasting) and **match** method families to data type, sample size, and question | Understand | readmes, 3.1 | timed Canvas MCQ quizzes (auto-graded) |
| 2 | **Build** reproducible computational environments (Git, pixi, notebooks) and **run** the same workflow on a laptop, HPC, or cloud instance | Apply | Ch 1, 5.4 | HW1 workbench setup |
| 3 | **Transform** diverse raw data streams — sensor time series from daily sampling to 100 Hz and acoustic, geospatial imagery and gridded fields, tabular and point observations — into AI-ready datasets, and **repair** real instrument pathologies (gaps, drift, timing errors, irregular sampling, label noise) | Analyze / Create | Ch 2 | Ch 2 exercises, final project data section |
| 4 | **Apply** signal-processing and statistical transforms (filtering, Fourier and wavelet transforms, resampling, dimensionality reduction) and **predict** their effect on what a model can learn downstream | Apply / Evaluate | 2.6–2.12 | Ch 2 exercises |
| 5 | **Design** the evaluation before the model: choose domain baselines, construct leakage-aware splits (temporal, spatial, grouped), and **report** metrics with uncertainty | Evaluate / Create | Ch 3 (3.8 hosts spatial/grouped CV), leaderboard | leaderboard (3.5, 4.10), HW |
| 6 | **Construct and train** classic ML and deep models (regression, forests, boosting; MLP, CNN, RNN, transformer, autoencoder) in scikit-learn and PyTorch, and **diagnose** good versus failing training runs | Apply / Create / Analyze | Ch 3–4 | HW-CML, HW-DL, 4.5 lab |
| 7 | **Exploit** physical knowledge as a data asset: **generate** synthetic training and benchmark data from physical models (spatiotemporal fields, waveforms, time series), **embed** physical constraints in losses and architectures, and **validate** against known ground truth | Create / Evaluate | 2.10, 4.7, mlgeo_synth thread | 4.5 lab, eval-set exercise |
| 8 | **Quantify** predictive uncertainty — bootstrap/ensembles for epistemic, MC dropout at inference, uncertainty as model output (quantile or distributional heads) — check calibration, and **judge** when a model is extrapolating beyond its training distribution | Evaluate | woven through Ch 3–4 (3.8–3.9, 4.5, 4.10) | Ch 3–4 exercises, 4.5 lab, final project |
| 9 | **Version and track** data, models, and experiments so another scientist reruns the pipeline and gets the same numbers | Apply / Evaluate | Ch 5 | final project repo (30%) |
| 10 | **Evaluate** AI agents and LLM output as scientific instruments: **write** a task spec, **build** an eval set with ground truth, **score** outputs, and **analyze** failure modes | Evaluate / Create | Ch 6 | 6.3 eval-set exercise, pre-submission-agent capstone |
| 11 | **Integrate** AI assistance into research while keeping intellectual ownership: **disclose** use, **verify** outputs, and **defend** every methodological choice without the assistant | Apply / Evaluate | 1.8, 6.4 | disclosure statements, presentations |
| 12 | **Translate** the same result for distinct audiences and **appraise** the downstream uses and consequences of a deployed model | Create / Evaluate | Ch 7 | final project deliverables (7.1, 7.2) |

Traceability to 2024: objectives 1–2 carry over the 2024 computing and use-case
bullets; #3–4 upgrade "apply standard data manipulation strategies" and "understand
at least qualitatively" to Analyze/Evaluate levels; #6 merges the workflow and
classic-ML/DL bullets; #9 absorbs open science and reproducibility; #12 replaces
"analyze and write a structured scientific paper" with audience-specific
communication. Outcomes 5, 7, 8, 10, 11 are new in 2026.

## Cross-cutting design principles

Reviewers check the book against these principles as well as the outcome table.

**1. Data-stream parity.** Published AI-for-Earth work leans geospatial because
images suit deep learning. Most working geoscientists instead live with sensor
data: high sampling rates (daily to 100 Hz to acoustic), instrument drift, gaps and
timing errors, and stations that are sparse in space but placed where the science
demanded. The book must represent geospatial and sensor streams equally, and treat
that diversity of data streams as the central technical challenge of ML in the
geosciences, not an inconvenience to normalize away.

**2. Physics is an asset.** Unlike most ML application domains, geoscience has
physical laws and physical models that generate synthetic data — spatiotemporal
fields or plain time series — with known ground truth. The book uses this
throughout: synthetics for training and benchmarking (mlgeo_synth), physical
constraints in models, ground truth for honest scoring.

**3. Baselines before models.** No learned model appears without a domain baseline
(persistence, climatology, STA/LTA, empirical correlation) and a leakage-aware
split. The leaderboard and hidden test sets exist to enforce this habit.

**4. Textbook depth, scalable delivery.** The book carries the full outcome set at
graduate depth. In-class delivery dilutes selectively for 469 (senior
undergraduates): fewer reading assignments, Apply-level rather than Create-level
expectations on outcomes 7 and 10, assisting rather than leading final projects —
mirroring the 2024 469/569 split. Persona reviewers at undergraduate level should
judge whether a *diluted path* through each chapter exists, not whether every
notebook is undergraduate-easy.

**5. Own what the AI did.** Every graded artifact separates the student's
contribution from the assistant's, and students must be able to defend any choice
in their pipeline without the assistant in the room.

## Decisions (Marine, 2026-08-12)

1. **Spatial/grouped cross-validation lives in 3.8** robust training (extend the
   existing notebook; no new section).
2. **Uncertainty quantification is woven through Chapters 3 and 4**, not confined
   to the 4.5 lab: bootstrap and ensembles for epistemic uncertainty, MC dropout at
   inference, and uncertainty crafted as part of the model output (quantile or
   distributional heads), with calibration checks.
3. **100 Hz seismic is the high-rate exemplar.** No separate acoustic worked
   example in Ch 2; outcome 3 keeps acoustic in scope conceptually.
4. **Assessment redesign.** Outcome 1 moves to timed Canvas MCQ quizzes with
   automatic grading. Reading reports become a four-stage arc that ends in agent
   building: (1) AI-assisted literature review on a given topic, (2) anatomy of
   good scientific papers, (3) students discuss and build their own quality
   standards as an explicit rubric, (4) students turn that rubric into their own
   pre-submission review agent. The arc feeds outcomes 10–12 and gives the Ch 6
   agent material a course-long runway.

See `00b_phase2-directives.md` for how Phase 2 must treat these.
