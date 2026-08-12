# Machine Learning in the Geosciences

The **GeoS**cience **MA**chine Learning **R**esources and **T**raining (GeoSMART) framework provides an educational pathway in open source scientific computing, general ML theory, toolkits, and deployment.

This book supports the course Machine Learning in the Geosciences (ESS 469/569) at the University of Washington. The book, tutorials, and homeworks live in this single repository; students run the notebooks locally or on a cloud service of their choice.

Instructors:

- Marine Denolle (mdenolle@uw.edu)
- Akshay Mehra (akmehra@uw.edu)

This project is supported by the GeoSMART team (Stefan Todoran, Nicoleta Cristea, Anthony Arendt, Scott Henderson, Ziheng Sun, Yiyu Ni, Akash Kharita).

## Overview

The course introduces machine learning in the geosciences, the basics of computing, and applied ML methodology. It works with canonical and topical data sets in seismology, oceanography, cryosphere, planetary sciences, geology, and geodesy. The methods taught include unsupervised clustering, logistic regression, random forest, support vector machines, and deep learning with PyTorch.

The course rests on three pillars, plus a fourth layer that runs through everything in the 2026 edition:

1. **AI-ready data**: turning raw geoscientific observations into data sets a model can learn from.
2. **Classic machine learning**: feature-based methods, trained and evaluated honestly.
3. **Deep learning**: neural networks in PyTorch, from perceptrons to modern architectures.
4. **Working with agentic AI**: students in 2026 write code alongside AI assistants that read repositories, run code, and propose changes. The course treats this as a skill to be taught, not a shortcut to be policed. Critical evaluation of AI output, translation of results for different audiences, and articulation of downstream impact are graded skills, on the same footing as model accuracy.

# Who this book is for

PhD students and senior undergraduates across the geosciences and adjacent engineering and computer science, plus the professors, postdocs, lab scientists, and program staff who train or hire them. A completer can take a raw, messy geoscience data stream to a defensible, reproducible, honestly evaluated model — and can say exactly which parts an AI assistant did.

# Learning outcomes

By the end of the book, students will be able to:

| # | Outcome | Bloom level | Where in the book | Graded artifact |
|---|---------|-------------|-------------------|-----------------|
| 1 | Describe the canonical uses of ML across the geosciences (discovery, automation, signal processing, emulation, forecasting) and match method families to data type, sample size, and question | Understand | chapter readmes, 3.1 | timed Canvas quizzes |
| 2 | Build reproducible computational environments (Git, pixi, notebooks) and run the same workflow on a laptop, HPC, or cloud instance | Apply | Chapter 1, 5.4 | HW1 workbench setup |
| 3 | Transform diverse raw data streams — sensor time series from daily sampling to 100 Hz, geospatial imagery and gridded fields, tabular and point observations — into AI-ready datasets, and repair real instrument pathologies (gaps, drift, timing errors, irregular sampling, label noise) | Analyze / Create | Chapter 2 | Chapter 2 exercises, final project data section |
| 4 | Apply signal-processing and statistical transforms (filtering, Fourier and wavelet transforms, resampling, dimensionality reduction) and predict their effect on what a model can learn downstream | Apply / Evaluate | 2.6–2.12 | Chapter 2 exercises |
| 5 | Design the evaluation before the model: choose domain baselines, construct leakage-aware splits (temporal, spatial, grouped), and report metrics with uncertainty | Evaluate / Create | Chapter 3 (spatial and grouped CV in 3.8), leaderboard | leaderboard (3.5, 4.10), homework |
| 6 | Construct and train classic ML and deep models (regression, forests, boosting; MLP, CNN, RNN, transformer, autoencoder) in scikit-learn and PyTorch, and diagnose good versus failing training runs | Apply / Create / Analyze | Chapters 3–4 | classic-ML and deep-learning homework, 4.5 lab |
| 7 | Exploit physical knowledge as a data asset: generate synthetic training and benchmark data from physical models, embed physical constraints in losses and architectures, and validate against known ground truth | Create / Evaluate | 2.10, 4.7, `mlgeo_synth` | 4.5 lab, 6.3 eval-set exercise |
| 8 | Quantify predictive uncertainty (bootstrap and ensembles, MC dropout at inference, quantile and distributional outputs), check calibration, and judge when a model is extrapolating beyond its training distribution | Evaluate | woven through Chapters 3–4 (3.8–3.9, 4.5, 4.10) | Chapter 3–4 exercises, final project |
| 9 | Version and track data, models, and experiments so another scientist reruns the pipeline and gets the same numbers | Apply / Evaluate | Chapter 5 | final project repository (30%) |
| 10 | Evaluate AI agents and LLM output as scientific instruments: write a task spec, build an eval set with ground truth, score outputs, and analyze failure modes | Evaluate / Create | Chapter 6 | 6.3 eval-set exercise, review-agent capstone |
| 11 | Integrate AI assistance into research while keeping intellectual ownership: disclose use, verify outputs, and defend every methodological choice without the assistant | Apply / Evaluate | 1.8, 6.4 | disclosure statements, presentations |
| 12 | Translate the same result for distinct audiences and appraise the downstream uses and consequences of a deployed model | Create / Evaluate | Chapter 7 | final project deliverables (7.1, 7.2) |

Data visualization concepts are introduced and used throughout the book.

# Prerequisites

**Prerequisites**: MATH 207 and MATH 208, or MATH 307 or 308, or AMATH 351 or 352, CS160 or CS163, or permission from the instructor.

**Recommended skills**: Knowledge of Python, AMATH301, 100- or 200-level courses in the Earth sciences. We provide refreshers on computing as part of the course.

# Syllabus

- **Part I: AI-ready GeoData**: geoscientific data, their modalities and dimensions, basic characteristics, feature extraction, dimensionality reduction, and how to format an AI-ready data set from geoscientific data.
- **Part II: Classic Machine Learning**: model training, evaluation, assessment of generalization, and good practice for reliable training of classic algorithms after feature engineering (e.g., K-means, random forest, k-NN).
- **Part III: Deep Learning**: fundamental concepts in deep learning — perceptrons and fully connected networks, convolutional and recurrent networks, a small transformer for sequence forecasting, autoencoders, and physics-informed neural networks — plus training practice: optimization, regularization, diagnosing failing runs, and uncertainty in model outputs.

Later chapters extend the pillars: reproducible workflows in the agent era (Chapter 5), building and evaluating AI agents (Chapter 6), and use cases, audience translation, and downstream impact (Chapter 7).

# Technical skills building

Throughout the course, students build skills in shell, version control with git and GitHub, Python programming, high performance computing, and data visualization in Python.

- _Shell_: introduced early in the course, used as needed.
- _Version control_: introduced early and used at every lecture.
- _Python programming_: progressively introduced. We detail the use of numpy, (geo)pandas, and scikit-learn, with PyTorch as the deep learning framework.
- _Visualization in Python_: introduced early with Matplotlib and Plotly, used in every Python lecture.
- _High performance computing_: used in the second half of the course and during the final project.
- _Agentic AI assistants_: introduced in Chapter 1 (see the [course AI-use policy](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md)) and used, with disclosure, throughout.

# Readings: from papers to a review agent

Reading assignments follow a four-stage arc across the quarter:

1. **AI-assisted literature review**: students drive an AI assistant through a literature review on an assigned topic and verify every citation themselves.
2. **Anatomy of good scientific papers**: dissect exemplar papers — what makes methods defensible, figures honest, and claims supported.
3. **Build your own quality standards**: through class discussion, each student writes an explicit quality rubric in their declared genre — journal paper or stakeholder deliverable, matching the two final-project tracks.
4. **Build a pre-submission review agent**: students turn their rubric into a review agent and test it with the evaluation machinery of Chapter 6 (task spec, eval set, failure analysis) against papers with known strengths and flaws.

Week-by-week scheduling for the arc lives in the course syllabus.

# Course infrastructure

This book contains all tutorials and homeworks. Students work in VS Code or JupyterLab with an agentic AI assistant, keep their work on GitHub, and manage software environments with [pixi](https://pixi.sh). To build the book locally:

```
pixi install
pixi run build
```

Each student creates a personal course repository named `MLGEO2026_UWNETID`, copies the environment files from this book into it, and keeps homeworks and project work there under version control.

# Licenses

The text and figures of this book are licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) (CC-BY-4.0). The code, including the source code in the notebooks, is licensed under the [MIT License](https://opensource.org/license/mit). You may reuse and adapt both with attribution.
