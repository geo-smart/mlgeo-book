# Adopting This Book

This page is for readers outside ESS 469/569 at UW: an instructor standing up their own course, a lab lead onboarding new group members, an advisor deciding whether to assign it, or a program officer judging what it teaches. The book's machinery — leaderboard, quizzes, hidden test sets — assumes nothing UW-specific; here is how to run it elsewhere, and how to read the book without running any of it.

## Stand up your own course instance

**Fork the repository** ([github.com/geo-smart/mlgeo-book](https://github.com/geo-smart/mlgeo-book)). All tutorials, homework, the `mlgeo_synth` synthetic-data package, and the grading infrastructure live in this one repo. `pixi install` then `pixi run build` reproduces the book locally.

**Leaderboard CI.** Students submit predictions by pull request; a GitHub Actions workflow scores each PR and updates the standings page on merge. The recipe, including the private-seed system, is in [`leaderboard/INSTRUCTOR.md`](https://github.com/geo-smart/mlgeo-book/blob/main/leaderboard/INSTRUCTOR.md) (not rendered in the book). Three tracks: classification, scored against a canonical public split of a Zenodo seismic dataset with a private re-split as the hidden set; a forecast diagnostic track against public CO2 data, deliberately exploitable and carrying no grading weight; and a graded hidden forecast track on an `mlgeo_synth` series whose truth file lives only on the instructor's machine (`leaderboard/private/`, gitignored), so public CI shows the section as pending and your local scoring run produces the real table. Any `mlgeo_synth` generator accepts a seed, so homework variants regenerate for free.

**Regenerate the hidden seeds yearly.** Once a cohort has seen a test set, it is a validation set. The rotation checklist in `INSTRUCTOR.md` covers it: new private seeds stored outside the repo, a refreshed CO2 holdout horizon, a regenerated hidden forecast series (the generator recipe is in the file), cleared submissions, and a dry run of the scoring script before week one. Report both public and hidden scores to your class; the gap between them is the lesson.

**Quizzes on any LMS.** The concept-check MCQ banks (outcome 1) are instructor-private source files, one per chapter, in a plain format that converts to Canvas via [text2qti](https://github.com/gpoore/text2qti) or pastes into any LMS. They are deliberately kept out of the public repository so answer keys do not circulate; adopting instructors can request the banks from the authors (see the contact in the README) or write their own from the chapter learning outcomes — scenario-based questions that test judgment about a situation, not recall. Nothing else requires Canvas: submissions arrive by PR, and anything that does not (reports, presentations) works over whatever your institution uses.

## Compute and AI-access floor

**Hardware.** An 8 GB RAM laptop with about 10 GB free disk is the baseline. No GPU is required: Chapters 1–3 are lightweight, and Chapter 4's models are deliberately small and train on CPU in bounded time (minutes per notebook, not hours). Chapter 5.5 streams a remote Zarr store and needs internet plus a few tens of MiB of transfer; everything else runs from local or downloaded-once data. Windows users have a sanctioned path through WSL2 or GitHub Codespaces ([1.9](../Chapter1-GettingStarted/1.9_workbench_setup_hw1.md)).

**AI assistant.** [1.8](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md) assumes each student has an agentic coding assistant by the end of week one. Free tiers change too fast to enumerate here, but the invariant holds: student and education programs from the major providers (GitHub Copilot's education tier is the longest-standing) have kept a no-cost agentic option available, and the course needs only one working assistant per student, not a specific product.

**The fully free path.** Where hosted assistants are unavailable or disallowed, an open-weights model runs locally: OLMo 2 via [Ollama](https://ollama.com) needs about 4.5 GB of disk and 8 GB of RAM, and works (slowly) on CPU. Chapter 6 is built so this is never a blocker: every graded notebook there runs offline against recorded transcripts and simulated agents, and the live-model sections are optional.

**Restricted networks.** For national labs, enclaves, and industry networks that block outbound internet, the "Restricted environments" section of [5.4](../Chapter5-ModelWorkflows/5.4_compute_beyond_laptop.md) gives the four substitutions: package mirrors and offline pixi installs, container transfer through an approved gateway, self-hosted CI, and self-hosted experiment tracking.

## Reader pathways

**Advisor / code-free path** (for a supervisor who wants the concepts and the standards without running a notebook): the chapter readmes end to end, plus [Chapter 1](../Chapter1-GettingStarted/readme.md) including [1.8](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md), [2.1](../Chapter2-DataManipulation/2.1_Data_Definitions.md), [3.1](../Chapter3-MachineLearning/3.1_concepts_supervision.md), [6.4](../Chapter6-AgenticAI/6.4_disclosure_and_norms.md), and [Chapter 7](../Chapter7-UseCases/readme.md). Prose and figures only; enough to supervise a student through the pipeline, set disclosure expectations, and read their evaluation critically.

**Six-week onboarding track** (a lab lead training a new hire who already programs): week 1, Chapter 1 through the workbench setup; weeks 2–3, the Chapter 2 core (2.1–2.6 and 2.13); week 4, [3.8](../Chapter3-MachineLearning/3.8_robust_training.ipynb) — the cross-validation and leakage material pays for the whole track; week 5, Chapter 5; week 6, [6.3](../Chapter6-AgenticAI/6.3_build_an_eval_set.ipynb) and [6.4](../Chapter6-AgenticAI/6.4_disclosure_and_norms.md). The new hire ends able to build a defensible dataset, split it honestly, keep work reproducible, and evaluate the AI assistance they will inevitably use.

**Standard quarter path**: the book in order, Chapters 1–7, with the [final project](1.10_MLGEO_FinalProject.md) running from week one and the [reading arc](../Chapter6-AgenticAI/6.5_reading_arc.md) staged across the term. The undergraduate (469) delivery dilutes expectations, not content: Apply-level rather than Create-level on the synthetics and agent-evaluation outcomes, and assisting rather than leading final projects.

## Licenses and attribution

The text and figures are [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); the code, including notebook source, is [MIT](https://opensource.org/license/mit). Both allow reuse and adaptation, including commercial, with attribution. For reused or adapted material, cite the book and link the source, for example:

> Adapted from *Machine Learning in the Geosciences* (Denolle et al., GeoSMART / University of Washington), https://github.com/geo-smart/mlgeo-book, CC-BY-4.0.

If you adopt the book for a course, an issue or note on the repository saying so helps us count adoptions and tell you when hidden-seed recipes or datasets change.
