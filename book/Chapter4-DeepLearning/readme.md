# Chapter Overview

## Chapter 4: Deep Learning

This chapter teaches deep learning by building it up one piece at a time, in PyTorch, on geoscience data. Every architecture is implemented, trained, and diagnosed in a notebook you can run on a laptop. The models are deliberately small; the ideas are not.

### Chapter map

1. **The perceptron** (4.0)
   - A single artificial neuron, implemented from scratch
   - Gradient descent one step at a time: random starting weights, errors, the loss of each sample and its mean, the chain rule, the gradient, the update, batches, the learning rate
   - The perceptron rule of 1958 and why a step activation cannot be trained by gradient descent
   - Backpropagation: the errors sent back through a small network

2. **Training a network for a classification** (4.1)
   - The five steps of every training script: dataset, model, loss, optimizer, training loop
   - Multi-class classification of seismic sources from tabular features
   - The loss of each sample and its mean over a batch; reading learning curves
   - Overfitting and two controls: dropout and batch normalization
   - Saving, checkpointing, and restoring models; PyTorch compared with scikit-learn's MLPClassifier

3. **Training a network for a regression** (4.2)
   - GNSS station P395: forecast the 30-day position from 60 days of history, split in time
   - Persistence and linear regression as baselines
   - Losses for regression: squared, absolute, Huber
   - Test samples outside the training range, and predicting the change instead

4. **Convolutional neural networks** (4.3)
   - Convolution and kernels on images
   - LeNet on MNIST, briefly
   - A 2-D CNN regressing warming trends from a synthetic climate field, read against least-squares baselines
   - A 1-D CNN earthquake detector and its detection floor, head-to-head with the classical STA/LTA trigger on the same traces
   - A reality check on real miniPNW waveforms: the synthetic-to-real gap measured, not tuned away
   - Reading and recoding a published network

5. **Sequence models** (4.4)
   - Context windows and forecast horizons
   - Vanilla RNNs and why gradients vanish
   - LSTMs, self-attention from scratch, and a small transformer encoder
   - All compared on the same forecasting task, against persistence and seasonal-naive baselines

6. **Graph networks** (4.5)
   - A synthetic seismic network as a graph: stations as nodes, the 6 nearest stations as edges, the adjacency matrix and its normalized form
   - Shaking at stations without a recording: a fully connected network, a graph convolution and message passing with distance and direction on the edges, ranked against inverse-distance weighting and floors from the known truth
   - Two tests a fully connected network fails: reorder the stations, add ten new stations; and why station terms need a recording history or Vs30

7. **Autoencoders and self-supervision** (4.6)
   - Dense, convolutional, and denoising autoencoders on seismic spectrograms
   - Masked-autoencoder pretraining, ported to a second domain (gridded climate fields)
   - Reusing a pretrained encoder when labels are scarce
   - A miniPNW transfer probe: features cross the synthetic-to-real gap, decision boundaries do not

8. **Training and evaluating a network: the lab, the clinic and the rubric** (4.7)
   - Pillar 1: training-data curation — label noise, structured label disagreement, class imbalance, sensor noise, heteroscedastic quality
   - Pillar 2: architecture — width, depth, baselines, deep ensembles and MC dropout for uncertainty, calibration vs discrimination, out-of-range behavior
   - Pillar 3: training strategies — learning rate, batch size, early stopping, schedulers
   - Diagnosing broken training runs from their loss curves
   - The training clinic (optional): one convolutional detector on real miniPNW seismograms, six decisions measured station by station
   - Hyperparameter search with Optuna
   - The rubric: which network for which data structure (tables, grids, sequences, graphs)
   - Checks with evidence for framing, design, training, evaluation and reporting, for students and AI assistants

9. **Physics-informed learning** (4.8)
   - Physics constraints as loss terms
   - A cooling-law ablation and a 1-D heat-diffusion PINN
   - A finite-difference baseline that beats the PINN on the forward problem (~750x faster, 100x more accurate)
   - An inverse PINN recovering diffusivity from 40 noisy samples, and a break-the-PINN loss-imbalance exercise
   - Where PINNs stand in 2026, and neural operators as successors

10. **Time-series forecasting shootout** (4.9)
   - Baselines, SARIMA, gradient boosting, LSTM, and a transformer encoder on real geoscience series
   - Honest temporal splits and MASE
   - The class forecasting leaderboard

11. **Final-project milestone** (4.20)
    - Architecture exploration, evaluation, and diagnostics requirements for the deep-learning milestone

Transfer learning appears where it is used: notebook 4.6 closes with a linear probe on a frozen pretrained encoder, which is transfer learning in miniature. Large language models and AI agents are covered in Chapter 6.

### Learning outcomes

By the end of this chapter, you will be able to:
- Implement, train, and evaluate neural networks in PyTorch, from a single neuron to a transformer encoder.
- Diagnose training problems from learning curves and fix them.
- Quantify how data quality, architecture choices, and training strategy each affect model performance.
- Estimate prediction uncertainty with deep ensembles.
- Use self-supervised pretraining when labeled data is scarce.
- Choose and benchmark forecasting models with leakage-free temporal splits.
