# Chapter Overview

## Chapter 3: Classic Machine Learning in Geosciences

This chapter covers classic machine learning (CML) for geoscience: models that learn from feature tables rather than raw waveforms or images. Classic ML is fast to build, cheap to run, and easy to interrogate. That makes it the right place to learn the habits that carry over to deep learning: baselines, honest data splits, and evaluation that matches how a model will actually be used.

### The arc of the chapter

1. **Concepts** (3.1) — the training-supervision taxonomy: supervised, unsupervised, semi-supervised, self-supervised, reinforcement, and active learning, and where each shows up in geoscience.
2. **Classification and regression** (3.2) — the two supervised problem types, a first end-to-end workflow, and the train/validation/test split.
3. **Clustering** (3.3) — unsupervised structure discovery: distance metrics, k-means from scratch, silhouette and elbow diagnostics, hierarchical clustering, and a volcanic-seismicity exercise.
4. **Binary classification** (3.4) — event-vs-noise detection, classifier comparison, and the metrics that matter when classes are imbalanced (precision, recall, PR curves vs ROC).
5. **Multiclass classification** (3.5) — four seismic source types, per-class confusion matrices, one-vs-rest ROC, and the class leaderboard exercise.
6. **Logistic regression from scratch** (3.6) — the one lesson where the black box is opened: the loss function, gradient descent, and automatic differentiation with PyTorch.
7. **Trees, forests, and boosting** (3.7) — decision trees, random forest regression, feature importance, and gradient boosting as the modern tabular default.
8. **Robust training** (3.8) — cross-validation for correlated data: why random splits lie on autocorrelated geoscience series, and the time-aware and grouped splits that do not.
9. **Ensemble learning** (3.9) — voting, bagging, boosting, and stacking.
10. **What became of AutoML** (3.10) — a short history of automated model search, the pieces that survived (hyperparameter optimization with Optuna, strong gradient-boosting defaults), and a critical-evaluation exercise on AI-generated modeling code.

Dimensionality reduction (PCA, t-SNE) is covered in Chapter 2.12 and is used here as a preprocessing step, not re-taught.

### The fair-evaluation thread

A single discipline runs through every notebook in this chapter:

- **Baselines first.** Before any model, establish what a trivial predictor achieves: the majority class for classification, the historical mean for regression. A model that cannot beat the baseline has learned nothing.
- **Never evaluate on training data.** Model quality is measured on data the model has not seen. Cross-validation happens inside the training set; the test set is touched once.
- **Splits must respect data structure.** Autocorrelated series and clustered spatial data need temporal, blocked, or grouped splits (3.8).
- **Hidden test sets exist.** The instructor holds regenerated variants of the course datasets with private seeds. Scores that were tuned against a public test set will show a gap on the hidden one.

Lesson 3.5 puts this into practice with a **class leaderboard**: students train a classifier of their choice on a canonical split of a real seismic-source dataset, submit predictions by pull request, and are scored by continuous integration against both the public and the hidden test sets.

### Tools

The chapter uses `scikit-learn` as the workhorse, `lightgbm` and scikit-learn's histogram gradient boosting for boosted trees, `optuna` for hyperparameter search, and `pytorch` in 3.6 to introduce automatic differentiation. Course datasets come from the `mlgeo_synth` package (physically motivated synthetic generators), the course data repository, and a curated Zenodo archive of Pacific Northwest seismic events.

### Learning outcomes

By the end of this chapter, you will be able to:

- Frame a geoscience problem as classification, regression, or clustering, and pick an appropriate first model.
- Establish baselines and evaluate models with metrics suited to the problem, including imbalanced classes.
- Design data splits and cross-validation schemes that respect temporal and spatial correlation.
- Train, tune, and compare tree ensembles and other classic models, and report results honestly.
- Read a machine-generated modeling script and find its flaws.

### Assignments

- **Homework**: one homework assignment (a whole-rock geochemistry classification problem) covering data preparation, PCA, clustering, and model comparison.
- **Final project milestone**: one milestone guideline (3.20) applying these methods to your own project dataset.
