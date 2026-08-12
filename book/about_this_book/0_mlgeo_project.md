# The MLGeo Project

This page walks through the stages of designing a machine learning project in the geosciences, and points to the chapters that teach each stage.

## 1. Frame the project

* Motivate the need for Machine Learning in your scientific project. 

Perform a literature review of the outstanding scientific questions and the solutions proposed in the literature. What would be the steps to solve the problem manually? What are the limitations of the current solutions? Will a new ML algorithm be generalizable enough to be applied to 10+ other research problems? What is the 5-10 year potential of the particular problem given new technology, new research facility, new societal relevance? Are there comparable problems for which the tools can be reused?

* What is the state of the data?

Is there a lot of data, and labeled data? Is there human expertise available? Will this be a supervised or unsupervised learning problem? Can the data be accessed from open-access archives that meet the FAIR guidelines (findable, accessible, interoperable, reusable)? What would be its DOI? Are there local regulations or collection agreements that constrain how the data can be shared?


## 2. Organize the project - Chapter 1

Start a GitHub repository with a README.md, create an environment specification (pixi or a YML file), and use human- and machine-readable file and folder names. Make sure the project name has not been used before.

## 3. Data download - Chapter 1

List the data, data information, data labels, and data provenance (including accessibility from open-access data archives). How large is the data? What data format would be optimal to read across languages (Python, C, R, MATLAB, Julia, etc.)? Can it store metadata? How does it perform with I/O?

Is the data geospatial, or time series?

Find an appropriate compute platform for storage and I/O of the data (cloud computing, institutional Linux cluster, etc.).

Create a Jupyter notebook to document data download and storage.

## 4. Data preparation - Chapter 2

* **Explore the data**

Create a Jupyter notebook for preliminary data exploration. Document:
- The name, the data type
- The noise: what is the type of noise (stochastic, outliers, data gaps, etc.)
- Data distribution: Gaussian, uniform, logarithmic, etc.
- Data labels (or target attributes)


Visualize a subset of the data.

Study basic data correlations between attributes.

How would you solve the problem manually given this data?

Identify transformations that may be useful (such as STFT, CWT, PCA).

Save preliminary plots and notebooks. Document findings.

* **Data preconditioning - Chapter 2**

Copy the data and work on these copies.

Write functions for all data transformation so that they can be automatically called (and easily debugged). These functions will be used for training, validating, and test sets.

**Clean data**: fix or remove outliers, fill missing values (zero, mean, median), or drop data (when there are too many data gaps, for instance).
Save the clean copy of the data in a different file.

Be careful with synthetic data. Toy random data (noise drawn from a generator, with no physics behind it) tells you little: an algorithm that works on it may behave completely differently on real observations, so avoid it for anything beyond a smoke test. Physically motivated synthetic data with documented ground truth is a different matter. Because you know the true answer, it is admissible for method development, benchmarking, and hidden test sets. Chapter 2.10 and the book's `mlgeo_synth` package generate this kind of data. For the science itself, prefer data collected from the real world.

* **Feature preparation - Chapter 2**

**Drop** attributes that are not useful for the task.

**Transform** features (such as STFT).

Explore quick and promising features (e.g., PGA for ground motions).

**Scale** the features to standardize or normalize them. ML algorithms will not perform well in most cases without normalization of the input features or data. Scaling is not a requirement, but it tends to improve the training behavior.
+ *Min-max scaling*: removes the minimal value, then normalizes by the maximum value of the distribution so that the amplitudes range between 0 and 1. It is appropriate when the features are positive numbers. The scikit-learn built-in function ``sklearn.preprocessing.MinMaxScaler()`` performs:

<code>X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))</code>

<code>X_scaled = X_std * (max - min) + min</code>
Scikit learn has built in functions to do the scaling.

+ *Standardization*: removes the mean and divides by the standard deviation. The output distribution does not have bounds. It is more stable than min-max scaling because it is less sensitive to outliers. The scikit-learn built-in function is ````sklearn.preprocessing.StandardScaler()````

There are other ways to normalize the input features/data.
https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing



## 5. Dimensionality reduction - Chapter 2

Explore possible ways to reduce the dimension of the data (PCA, ICA).

Document the data transformation with notebooks. Reassign data attributes/labels in the new coordinates.

## 6. Model design - Chapter 3 and 4

Find the ***baseline*** model that the ML project is supposed to beat. At the least, your ML algorithm has to beat the *random* baseline, or there are issues in the model design or in the input data.

Try several model algorithms. The *no free lunch theorem* (Wolpert 1995): there is no such thing as the best learning algorithm overall, only an algorithm that is very accurate on a given data set.

The model should have the **minimum complexity** that is required to **minimize the model expected error**.


## 7. Model Training - Chapters 3, 4

Separate the data into three sets: a training set to fit the model, a validation set to tune hyperparameters and guide model design, and a test set touched once, at the end, to report performance.

Design the split before worrying about its proportions. Geoscience data are correlated in time and space, so a random split usually leaks information: samples from the same storm, the same earthquake, or the same station land on both sides of the boundary, and the test score becomes optimistic. Match the split to the correlation structure of the data:

- *Temporal split*: train on the past, test on the future, when samples are ordered in time.
- *Spatial split*: hold out whole regions, with a buffer, when samples are spatially autocorrelated.
- *Grouped split*: keep all samples from one event, station, or site on the same side of the split.

[Chapter 2.13](../Chapter2-DataManipulation/2.13_MLready_data.ipynb) makes benchmark splits and leakage controls part of the definition of an AI-ready dataset; [Chapter 3.8](../Chapter3-MachineLearning/3.8_robust_training.ipynb) teaches leakage-aware cross-validation, including its spatial and grouped variants. Once the split design is leakage-safe, the proportions are secondary (70/15/15 and 60/20/20 are common). Cross-validation over the training and validation portion — with folds that respect the same temporal, spatial, or group structure — estimates the expected error of the learning algorithm and its spread.

Save intermediate results when possible.

Save the seeds of the random number generator to be able to reproduce the results.

Avoid writing your own homegrown library of code. Use reliable sources.

Provide good documentation, especially when working in groups.

Start smaller than the final run. It is suggested that no more than 25% of the available resources should be used in the first model design.

Select a performance measure. 



