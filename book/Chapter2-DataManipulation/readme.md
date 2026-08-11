# Chapter Overview

## Chapter 2: AI-Ready Geoscience Data

This chapter is the AI-ready-data pillar of the course. Machine learning projects in the geosciences succeed or fail on the quality of their data, and most of the work is upstream of any model: understanding what the data are, reading and writing standard formats, cleaning tables, reshaping arrays, resampling, characterizing distributions, transforming and filtering signals, generating honest synthetic data, engineering features, and reducing dimensionality. The chapter ends with a capstone lesson that defines AI-ready data operationally — provenance, metadata, tidy shapes, benchmark splits, and leakage controls — and that definition is graded in the final project.

### The Arc of the Chapter

The lessons build in order:

1. **2.1 Data Definitions** — data modalities in geoscience; arrays vs data frames; common and cloud-optimized formats.
2. **2.2 Data Formats** — hands-on reading and writing of CSV, GeoJSON, GeoTIFF, netCDF, HDF5, Parquet, and Zarr; comparing file sizes on disk.
3. **2.3 Pandas DataFrames** — series and data frames, datetime handling, filtering, grouping, aggregating, and mapping station metadata.
4. **2.4 Preparing DataFrames** — cleaning a whole-rock geochemistry table: missing data, sentinel values, correlations, and per-class distributions.
5. **2.5 Arrays** — NumPy and Xarray arrays, indexing and reshaping, labeled dimensions, and a first look at PyTorch tensors.
6. **2.6 Resampling** — downsampling, interpolation, and the bootstrap, applied to synthetic and real GNSS time series.
7. **2.7 Statistical Considerations** — moments, distributions, and the Gutenberg-Richter law, on synthetic and real geochemical data.
8. **2.8 Spectral Transforms** — Fourier and wavelet transforms of seismograms and 2D fields.
9. **2.9 Filtering** — low-, high-, and band-pass filtering; zero-phase vs causal filters; separating trend, seasonal cycle, and noise.
10. **2.10 Synthetic Data** — building synthetic seismograms and spectrum-matched noise; when synthetic data is admissible in science.
11. **2.11 Feature Engineering** — hand-built and automated features for time series, with a real seismic waveform benchmark.
12. **2.12 Dimensionality Reduction** — PCA, EOFs on climate fields, ICA, and t-SNE.
13. **2.13 AI-Ready Data (Capstone)** — the operational checklist: data cards, benchmark splits, leakage-by-preprocessing, and correct splits for autocorrelated data.

The chapter closes with the **final project assignment for this pillar (2.20)**: build an AI-ready dataset for your own project, with a data card and benchmark splits, following the 2.13 checklist.

### Learning Outcomes

By the end of this chapter, you will:

- Recognize the data types, modalities, and formats common in the geosciences, including cloud-optimized formats.
- Manipulate tabular data with Pandas and array data with NumPy and Xarray.
- Characterize data with statistical moments, distributions, and resampling methods.
- Apply Fourier and wavelet transforms and design digital filters.
- Generate synthetic data responsibly and disclose its use.
- Engineer features and reduce dimensionality for downstream ML tasks.
- Assemble an AI-ready dataset with documented provenance, a data card, and leakage-safe benchmark splits.

### Assignments

- **Final Assignment (2.20)**: Build an AI-ready dataset for your final project. Apply the 2.13 checklist: document provenance and licensing, write a data card, define benchmark splits, and demonstrate that your preprocessing does not leak information across splits.
