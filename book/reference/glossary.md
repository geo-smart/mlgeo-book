# Glossary

Terms used throughout the book, from both sides of its audience: machine-learning vocabulary for geoscience students, and geoscience/signal-processing vocabulary for computing students. Entries are alphabetical.

```{glossary}

Ablation
: An experiment that removes one component of a model or workflow (a feature, a loss term, a layer) and retrains, so the score difference measures what that component contributes. Notebook 4.7 ablates the physics term of a PINN loss this way.

Agent
: A large language model wrapped in a loop that can call tools (run code, read files, search), observe the results, and act again toward a goal, rather than producing a single text reply. Chapter 6 builds and evaluates agents for research tasks.

AI-ready data
: Data organized so a model can consume it directly and a reader can trust it: consistent units and sampling, documented provenance and licenses, clean labels, and machine-readable formats. Chapter 2 turns raw downloads into AI-ready datasets.

Baseline
: The simplest credible model for a task — persistence or climatology for forecasting, the majority class for classification, linear regression for tabular data. Any complex model must beat the baseline to justify its complexity.

Calibration
: The agreement between a model's stated confidence and reality: among predictions made with 80% probability, about 80% should be correct, and 90% prediction intervals should cover about 90% of outcomes. Calibration is measured (reliability diagrams, interval coverage), never assumed.

Coda
: In seismology, the tail of a seismic record after the main P and S arrivals: scattered waves whose amplitude decays gradually with time. Coda shape helps distinguish source types (an explosion's coda differs from an earthquake's).

Corner frequency
: The frequency at which an earthquake's amplitude spectrum bends from flat to decaying. It scales inversely with rupture duration, so larger earthquakes have lower corner frequencies; it is a standard feature for characterizing seismic sources.

Data leakage
: Any path by which information unavailable at prediction time reaches the model during training — fitting a scaler on the full dataset before splitting, test samples correlated with training samples, future values informing past predictions. Leakage inflates scores that then collapse on genuinely new data. See also *leakage (spatial and temporal)* and, for the unrelated signal-processing term, *spectral leakage*.

Deep ensemble
: Several copies of the same network trained from different random seeds. Averaging their predictions usually improves accuracy, and their disagreement on a sample estimates epistemic uncertainty (notebook 4.5).

Dimensionality reduction
: Compressing many features into few while preserving structure, either to visualize data (PCA, t-SNE, UMAP) or to feed a smaller representation to a model. An autoencoder's bottleneck is a learned form of it.

Epistemic vs. aleatoric uncertainty
: Epistemic uncertainty comes from what the model does not know — too little data, unseen conditions — and shrinks as data grows; ensembles and MC dropout estimate it. Aleatoric uncertainty is randomness in the process or measurement itself (sensor noise) and does not shrink with more training data.

Eval set
: A curated collection of test cases with known answers and an automatic scoring rule, used to measure an LLM or agent system instead of trusting impressions. Notebook 6.3 builds one from synthetic data with known ground truth.

Expected calibration error (ECE)
: A single-number summary of a reliability diagram: the average gap between stated confidence and observed accuracy, weighted by how many predictions fall in each confidence bin. Computed alongside reliability diagrams in notebooks 3.6 and 4.5.

Feature engineering
: Constructing informative model inputs from raw data using domain knowledge: STA/LTA ratios, spectral statistics, rolling means, kurtosis. Classical machine learning (Chapter 3) depends on it; deep networks learn many features from raw data instead.

Flicker noise
: Noise whose power grows toward low frequencies (roughly as 1/f), common in GNSS position series and electronic instruments. Unlike white noise it does not average away with longer records, so it biases trend and uncertainty estimates if modeled as white.

[Git](https://git-scm.com)
: The version control system used throughout the course to record, compare, and share the history of code and text.

[GitHub](https://github.com)
: The hosting service where the course repositories, datasets, and student projects live, built around Git plus issues and pull requests.

Grouped cross-validation
: Cross-validation that keeps all samples sharing a group — the same station, field site, or earthquake — in the same fold. Scores then measure generalization to new groups instead of interpolation within familiar ones (Chapter 3.8).

Heteroscedastic noise
: Noise whose magnitude varies across samples — a mixed network of lab-grade and field-grade instruments, or stations with different site conditions. When the per-sample noise level is known, variance-weighted losses recover accuracy that ignoring the metadata gives up (notebook 4.5).

Hidden test set
: Held-out data that is never inspected or touched during model development and is scored once, at the end. The class forecasting leaderboard (4.10) uses one; touching it repeatedly turns it into a validation set.

Instrument response
: The transfer function of a sensor and recording system, mapping true ground motion to the recorded counts. Seismic and geodetic records must be corrected for it before amplitudes carry physical units.

Interpolation vs extrapolation
: Prediction inside the region covered by training data versus outside it. Most learned models degrade sharply, and sometimes silently, when extrapolating; the training distribution is a contract, and split design (3.8) plus out-of-range checks (4.5) test whether a score speaks to one regime or the other.

Jupyter notebook
: A document mixing runnable code, its outputs, and narrative text; the format of most chapters of this book.

Leakage (spatial and temporal)
: The two leakage flavors that dominate geoscience data. Random splits of autocorrelated data place near-duplicates on both sides of the split: neighboring pixels or stations (spatial), overlapping or adjacent time windows (temporal). The fixes are blocked or buffered spatial splits and splits that respect time order.

Linear probe
: A classifier made of a frozen pretrained encoder plus a small trained linear head. It measures what the pretrained features alone carry; *fine-tuning*, by contrast, also updates the encoder weights. Notebook 4.6 runs a linear probe against training from scratch.

MASE
: Mean absolute scaled error: a forecast's mean absolute error divided by that of a naive baseline (persistence or seasonal-naive) on the same series. MASE below 1 beats the baseline; above 1, the model loses to it (notebook 4.10).

MC dropout (Monte Carlo dropout)
: Running a trained network many times with dropout left on at prediction time. Each pass samples a slightly different subnetwork, and the spread of the outputs approximates epistemic uncertainty with a single trained model. Compared head-to-head with a deep ensemble in notebook 4.5.

[MyST](https://mystmd.org)
: Markedly Structured Text, the Markdown flavor and build system this book is written and published with.

Physics-informed neural network (PINN)
: A network trained with the residual of a governing equation (conservation law, diffusion equation) added to the loss alongside data misfit, so predictions are pulled toward physically consistent solutions (notebook 4.7).

[pixi](https://pixi.sh)
: The package and environment manager used to install the course software stack. It resolves conda-ecosystem packages into a lockfile, so every student and the CI build the identical environment.

[pooch](https://www.fatiando.org/pooch/)
: A small Python library that downloads a data file from a URL, caches it locally, and verifies its SHA256 checksum, so an analysis provably ran on the intended file.

Spectral leakage
: In signal processing, the smearing of energy from one frequency into neighboring frequency bins when the analyzed window does not contain a whole number of cycles. Tapering (windowing) reduces it. Unrelated to *data leakage*.

STA/LTA
: Short-term average over long-term average of signal amplitude: a running ratio that spikes when a transient arrives on top of background noise. The classic trigger for detecting earthquakes in continuous seismic data, and a standard engineered feature.

Supervised, unsupervised, and self-supervised learning
: Supervised learning fits inputs to human-provided labels; unsupervised learning finds structure (clusters, low-dimensional embeddings) without labels; self-supervised learning manufactures labels from the data itself — mask part of the input and predict it — so pretraining can use unlimited unlabeled archives (notebook 4.6).

Tolerance-based reproducibility
: Declaring a result reproduced when a rerun matches within a stated numerical tolerance, rather than bit-for-bit. Floating-point arithmetic, parallel execution order, and hardware differences make exact equality the wrong test for scientific pipelines (Chapter 5).

Transformer
: The architecture built from stacked self-attention layers with positional encodings, introduced by Vaswani et al. (2017). It underlies large language models and current forecasting systems; notebook 4.4 builds a small one from parts.

[Zarr](https://zarr.dev)
: A storage format for chunked, compressed N-dimensional arrays, designed so cloud object stores can serve partial reads in parallel. The common choice for large gridded geoscience data alongside NetCDF.

```
