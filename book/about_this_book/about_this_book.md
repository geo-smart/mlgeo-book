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

# Learning objectives

By the end of the quarter, students should be able to:

- Demonstrate computing skills in Python, Jupyter notebooks, and Git version control, and deploy scripts on local computers or cloud instances.
- Develop and apply standard machine learning workflows: data preparation; model design; model training, validation, and evaluation.
- Apply standard data manipulation strategies in the geosciences: data types (time series and geospatial), data formats, data visualization, dimensionality reduction, and feature engineering.
- Describe and practice open science principles, reproducibility, and digital scholarship.
- Describe canonical ML examples across geoscience disciplines.
- Understand at least qualitatively how techniques such as the Fourier and wavelet transforms or principal component analysis manipulate data, and interpret their output.
- Use an agentic AI assistant productively and critically: direct it, verify its output, and document its role in the work.
- Communicate the same scientific result to different audiences, and state the societal, environmental, or economic relevance of a project.

[Detailed syllabus (PDF)](MLGeo_2024.pdf)

Data visualization concepts are introduced and used throughout the book.

# Prerequisites

**Prerequisites**: MATH 207 and MATH 208, or MATH 307 or 308, or AMATH 351 or 352, CS160 or CS163, or permission from the instructor.

**Recommended skills**: Knowledge of Python, AMATH301, 100- or 200-level courses in the Earth sciences. We provide refreshers on computing as part of the course.

# Syllabus

- **Part I: AI-ready GeoData**: geoscientific data, their modalities and dimensions, basic characteristics, feature extraction, dimensionality reduction, and how to format an AI-ready data set from geoscientific data.
- **Part II: Classic Machine Learning**: model training, evaluation, assessment of generalization, and good practice for reliable training of classic algorithms after feature engineering (e.g., K-means, random forest, k-NN).
- **Part III: Deep Learning**: fundamental concepts in deep learning — fully connected layers, convolutional neural networks, sequence-to-sequence learning, canonical architectures (deep networks, ResNets, U-Nets) — training strategies such as data augmentation, regularization, and physics-informed losses, and modern topics such as foundation models and large language models for geoscience.

Later chapters extend the pillars: reproducible workflows in the agent era (Chapter 5), building and evaluating AI agents (Chapter 6), and use cases, audience translation, and downstream impact (Chapter 7).

# Technical skills building

Throughout the course, students build skills in shell, version control with git and GitHub, Python programming, high performance computing, and data visualization in Python.

- _Shell_: introduced early in the course, used as needed.
- _Version control_: introduced early and used at every lecture.
- _Python programming_: progressively introduced. We detail the use of numpy, (geo)pandas, and scikit-learn, with PyTorch as the deep learning framework.
- _Visualization in Python_: introduced early with Matplotlib and Plotly, used in every Python lecture.
- _High performance computing_: used in the second half of the course and during the final project.
- _Agentic AI assistants_: introduced in Chapter 1 (see the [course AI-use policy](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md)) and used, with disclosure, throughout.

# Readings and webinars

Each week, students write a short report about a paper or a webinar. Use the template on Canvas and answer the questions when appropriate. Report PDFs are due Wednesdays at 11:59 pm PT on Canvas. The instructor spends 15 minutes Monday morning summarizing the reading and webinar reports. Papers can be found and uploaded on a shared private course Google Drive [here](https://drive.google.com/drive/folders/1dyxfslCLzFFTYtX_vbjudlzaXvOxkepe?usp=sharing) (accessible with a UW email address).

# Course infrastructure

This book contains all tutorials and homeworks. Students work in VS Code or JupyterLab with an agentic AI assistant, keep their work on GitHub, and manage software environments with [pixi](https://pixi.sh). To build the book locally:

```
pixi install
pixi run build
```

Each student creates a personal course repository named `MLGEO2026_UWNETID`, copies the environment files from this book into it, and keeps homeworks and project work there under version control.
