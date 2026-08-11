# Chapter 5: Workflows, Reproducibility, and Rigor in the Agent Era

Chapters 1 through 4 taught you to build models and evaluate them fairly. This chapter teaches you to run the whole operation so that anyone — including you, six months from now, and including an AI agent working in your repository — can rerun it and get the same answer.

The timing matters. In 2026 much of the code in a research project is drafted by agents. That raises the value of the practices in this chapter rather than lowering it: an agent can produce a plausible, wrong, unreproducible analysis faster than you can read it. Pinned environments, scripted transforms, tracked experiments, and continuous integration are how you stay in control of work you did not type yourself.

## What is in this chapter

1. **[5.1 Reproducibility](5.1_reproducibility.md)** — Reproducibility versus replicability, and the reproducibility stack: pinned environments, seeds, immutable raw data, containers, and executable checks. This book's own build is the case study.
2. **[5.2 Experiment tracking](5.2_experiment_tracking.ipynb)** — A hands-on lab. You build a minimal experiment tracker in about 30 lines and use it to run a small hyperparameter study, then see what MLflow and Weights & Biases add on top of the same ideas.
3. **[5.3 Data and model versioning](5.3_data_model_versioning.md)** — Why git fails for data, what DVC and friends do, the minimum viable practice with checksums, and how to version models with model cards. Plus: copying this book's CI pattern into your project repository.
4. **[5.4 Compute beyond the laptop](5.4_compute_beyond_laptop.md)** — When to move off your laptop, the ladder of options from departmental HPC to commercial cloud, cost discipline, and cloud-optimized data access.

## Learning outcomes

By the end of this chapter you can:

- state the difference between reproducibility and replicability and say which one a given check tests;
- pin an environment so a collaborator (or CI, or an agent) reruns your code with the same library versions;
- track experiments so that every reported number can be traced to a run, a commit, and a dataset version;
- decide when a project needs more compute than a laptop, and pick the cheapest adequate option;
- set up CI that executes your notebooks on every pull request.

These skills feed directly into the final project rubric ([reproducibility and documentation criteria](../about_this_book/1.10_MLGEO_FinalProject.md)).
