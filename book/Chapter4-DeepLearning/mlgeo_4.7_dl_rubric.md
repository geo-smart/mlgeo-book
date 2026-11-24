# 4.7 (continued) The rubric: best practice for deep learning in the geosciences

This rubric is the checklist that closes section 4.7; the lab that measures many of its items is [4.7 Training and evaluating a network](mlgeo_4.7_ModelTraining.ipynb), and the optional [training clinic](mlgeo_4.7_training_clinic.ipynb) applies the same decisions to real miniPNW records. It collects, in one place, the checks that Chapter 4 teaches one network at a time. It has two readers. Students use it to design, train and evaluate a network for their project. An AI assistant (a coding agent asked to build or review a network) uses it as its specification: every item has an identifier, a check, and the evidence the agent must show. A claim without its evidence does not pass.

The rubric is a draft that the class extends. Each lecture of Chapter 4 ends with the items it adds.

## How to use it

1. **Frame the problem** (section A) before writing any model code.
2. **Choose the network from the structure of the data** (section B).
3. **Apply the design checks** for that network (section C).
4. **Train** with the checks of section D, and **evaluate** with section E.
5. **Report** with section F. An agent's report is a table of item identifiers, each with pass or fail and the evidence.

## A. Framing: before any model

| ID | Check | Evidence |
|---|---|---|
| A1 | Name the data sample, the input X, the quantity to predict Y and their units. | One sentence per item; one sample printed. |
| A2 | Name the structure of the input: a table of features, a grid (waveform, image, map), a sequence in time, or a set of stations (graph). | One sentence; see section B. |
| A3 | Name how the model will be used: on new times, new regions, new events, new instruments. This is the deployment axis. | One sentence. |
| A4 | Split along the deployment axis (train on the past, validate on the future; hold out regions or events), with a gap when samples overlap in time or space. | Split code; dates or regions of each partition. |
| A5 | Fit every scaling or normalization on the training samples only. | Code line. |
| A6 | Compute a baseline without learning (persistence, climatology, majority class) and a classical baseline (linear or logistic regression, random forest) on the same split. | Table of scores. |
| A7 | Check the range: do the validation and test observations and inputs lie inside the training range? If not, predict a change or an anomaly, or say the model extrapolates. | Ranges printed. |

## B. Choosing the network from the structure of the data

Each kind of layer assumes something about how the inputs are related (its *inductive bias*; Battaglia et al., 2018, Table 1). Choose the one whose assumption matches the data.

| Input structure | Relation between inputs | Network | What it assumes | Geoscience examples in this book |
|---|---|---|---|---|
| Table of features, no order | none known | Fully connected (MLP) | nothing: every input connects to every neuron | 61 waveform features of seismic sources (4.1); six summary numbers of a GNSS series (4.2) |
| Grid: waveform, spectrogram, image, map | neighbors are related | Convolutional (CNN) | locality, and the same pattern can appear anywhere (translation) | earthquake detector on waveforms (4.3); trend in a gridded field (4.3) |
| Sequence in time | order matters, the past informs the future | Recurrent (RNN, LSTM) or attention (Transformer) | sequential order; the same rule applies at every time | 90 days of GNSS displacement to forecast the next 30 (4.4) |
| Stations or cells with links | arbitrary neighbors | Graph network | relations given by a graph; station order does not matter | shaking at stations without a recording, a synthetic seismic network (4.5) |

The same question can come with different structures. A seismic event can be a table of 61 features (fully connected) or a waveform (convolutional). A GNSS series can be six summary numbers (fully connected) or the 90 daily values (recurrent or attention). Hand-built features have already done part of the work; a network on the raw record has to learn it, and needs more data.

## C. Design checks, by network

### C1. Fully connected network (4.1, 4.2)

| ID | Check | Evidence |
|---|---|---|
| C1.1 | Inputs standardized (zero mean, unit variance on training samples). | Code line. |
| C1.2 | The last layer fits the quantity to predict: one number with no activation for a regression; one score per class, with the softmax inside the loss, for a classification. | Model code. |
| C1.3 | Start small: one hidden layer of 16 to 64 neurons with ReLU; add width or depth only if the training loss stays high. | Number of weights against number of training samples. |
| C1.4 | The no-hidden-layer model (linear or logistic regression) is trained and reported as the baseline. | Score table (A6). |

### C2. Convolutional network (4.3)

| ID | Check | Evidence |
|---|---|---|
| C2.1 | Kernel size and the receptive field (the span of input one output sees) are chosen against the physical scale of the pattern: the period of a seismic phase, the size of a storm cell. | Receptive field in seconds, samples or km. |
| C2.2 | Each record is normalized so that the network cannot use the amplitude as a shortcut, unless amplitude is the signal. | Code line. |
| C2.3 | Pooling or a global average makes the output independent of where the pattern sits, when that is the question (detection); keep position when it matters (picking an arrival time). | Model code and one sentence. |
| C2.4 | The classical method with the same information (template matching, STA/LTA, a regression with the known terms) is run on the same records. | Score table. |

### C3. Recurrent network and attention (4.4)

| ID | Check | Evidence |
|---|---|---|
| C3.1 | Causal: no input from after the forecast time; windows built from the past only. | Window code. |
| C3.2 | Length of the input window chosen against the memory of the process (a season, an earthquake cycle phase). | One sentence. |
| C3.3 | Persistence and seasonal-naive forecasts are computed for every horizon. | Table by horizon. |
| C3.4 | Error reported as a function of the forecast horizon, not as one number. | Plot or table. |

### C4. Graph network (4.5)

| ID | Check | Evidence |
|---|---|---|
| C4.1 | The graph is built from the physics: who is a neighbor (the k nearest stations, or all stations within a radius in km) is stated and chosen against the distance over which the quantity stays correlated. | k or the radius, the number of edges, the smallest and largest degree, and the median distance to a neighbor in km. |
| C4.2 | The edges carry what the physics needs: distance and direction (east and north offsets) as edge features, with their scale. A layer that only knows who is a neighbor (a graph convolution) can only smooth. | The edge features and their units, in the model code. |
| C4.3 | Split by event or by sequence of events (train on the past, validate and test on the future), and test separately on stations the network never saw in training. | Split code; the held-out stations; the score at those stations. |
| C4.4 | Inverse-distance weighting (or kriging) is computed on the same hidden stations of the same data samples as the network; with synthetic data, the floors from the known truth are rows of the same table. | Score table with the baselines as rows. |
| C4.5 | Permutation check: reorder the stations, rebuild the graph, predict again. Each station's prediction must not change beyond the rounding of 32-bit numbers. | The largest change of a prediction, printed. |
| C4.6 | Error against distance to the source, where the shaking changes fastest. | Error near the source (for example within 40 km) next to the error over all stations. |

## D. Training

| ID | Check | Evidence |
|---|---|---|
| D1 | Starting weights are random, drawn with a recorded seed; never zeros. | Seed in the code and the report. |
| D2 | The loss fits the quantity to predict and its errors: cross-entropy for classes (with class weights if classes are unbalanced); mean squared error for Gaussian-like errors; mean absolute error or Huber for heavy tails and outliers. | Loss line and one sentence. |
| D3 | The loss used for training is the mean of the loss of each sample over the batch; the batch size is stated. | Code. |
| D4 | Training and validation loss are plotted at every epoch. | Figure. |
| D5 | Learning rate: start at 0.001 with Adam; if the loss diverges or crawls, change it by factors of 3 to 10 and show the curves. | Figure or table. |
| D6 | Regularize (dropout, weight decay, early stopping) only after the two curves part; keep the model with the lowest validation loss (checkpoint). | Curves before and after. |
| D7 | Train with at least three seeds and report the spread. | Mean and standard deviation. |
| D8 | The test set is used once, after all choices are made on the validation set. | Statement and the code path. |

## E. Evaluation

| ID | Check | Evidence |
|---|---|---|
| E1 | Regression: error in physical units (mm, °C, s), mean absolute error and its spread over seeds, against the baselines of A6. | Table. |
| E2 | Classification: confusion matrix, recall and precision per class, and accuracy against the majority class and the logistic baseline. | Matrix and table. |
| E3 | A difference smaller than the seed spread, or than one standard error of the test score, is reported as a tie. | Numbers. |
| E4 | Error against a physical variable: signal-to-noise ratio, distance, magnitude, season, depth. | Figure. |
| E5 | One test along the deployment axis that was not used for any choice (a later period, another region, real records for a model trained on synthetics). | Score and one sentence. |

## F. Reporting, for students and agents

| ID | Check | Evidence |
|---|---|---|
| F1 | Every number in the report is produced by a script or notebook in the repository, with data source and version. | Path to the script. |
| F2 | The report shows the baseline table, the loss curves and the seed spread before any claim of improvement. | The three items. |
| F3 | An agent states which items it could not check, and why. | List. |
| F4 | An agent does not claim that a network is better than its baseline unless E3 passes. | Numbers. |

## Where each item is taught

| Lecture and notebook | Items |
|---|---|
| Lecture 21, notebook 4.0: one neuron, gradient descent, backpropagation | D1, D3, D4, D5 |
| Notebook 4.1: classification with a fully connected network | A5, A6, C1, D2, D6, D8, E2, E3 |
| Notebook 4.2: regression with a fully connected network | A1, A3, A4, A7, D2, D7, E1 |
| Notebook 4.3: convolutional networks | C2, E4, E5 |
| Notebook 4.4: recurrent networks and attention | C3, A4 |
| Lecture 23b, notebook 4.5: graph networks | A2, A6, C4, D7, E3, E4, E5 |
| Notebook 4.7 (the lab of this section): data curation, architecture and training strategies | D5, D6, D7, E5 |

## Reference

Battaglia, P. W., et al. (2018). Relational inductive biases, deep learning, and graph networks. arXiv:1806.01261, Table 1.
