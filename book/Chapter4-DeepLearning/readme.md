# Chapter Overview

## Chapter 4: Deep Learning

This chapter teaches deep learning by building it up one piece at a time, in PyTorch, on geoscience data. Every architecture is implemented, trained, and diagnosed in a notebook you can run on a laptop. The models are deliberately small; the ideas are not.

### Chapter map

1. **The perceptron** (4.0)
   - A single artificial neuron, implemented from scratch
   - The perceptron learning rule and its limits
   - Gradient descent compared against ordinary least squares

2. **A first neural network** (4.1)
   - The five steps of every training script: dataset, model, loss, optimizer, training loop
   - Multi-class classification of seismic sources from tabular features
   - Reading learning curves

3. **Multi-layer perceptrons** (4.2)
   - Depth, dropout, and batch normalization
   - Saving, checkpointing, and restoring models
   - PyTorch compared with scikit-learn's MLPClassifier

4. **The three pillars of model development** (4.5)
   - Pillar 1: training-data curation — label noise, class imbalance, sensor noise
   - Pillar 2: architecture — width, depth, baselines, and deep ensembles for uncertainty
   - Pillar 3: training strategies — learning rate, batch size, early stopping, schedulers
   - Diagnosing broken training runs from their loss curves
   - Hyperparameter search with Optuna

5. **Convolutional neural networks** (4.3)
   - Convolution and kernels on images
   - LeNet on MNIST, briefly
   - A 1-D CNN earthquake detector and its detection floor as a function of signal-to-noise ratio
   - Reading and recoding a published network

6. **Sequence models** (4.4)
   - Context windows and forecast horizons
   - Vanilla RNNs and why gradients vanish
   - LSTMs, self-attention from scratch, and a small transformer encoder
   - All compared on the same forecasting task

7. **Autoencoders and self-supervision** (4.6)
   - Dense, convolutional, and denoising autoencoders on seismic spectrograms
   - Masked-autoencoder pretraining
   - Reusing a pretrained encoder when labels are scarce

8. **Physics-informed learning** (4.7)
   - Physics constraints as loss terms
   - A cooling-law ablation and a 1-D heat-diffusion PINN
   - Where PINNs stand in 2026, and neural operators as successors

9. **Time-series forecasting shootout** (4.10)
   - Baselines, SARIMA, gradient boosting, LSTM, and a transformer encoder on real geoscience series
   - Honest temporal splits and MASE
   - The class forecasting leaderboard

10. **Final-project milestone** (4.20)
    - Architecture exploration, evaluation, and diagnostics requirements for the deep-learning milestone

Transfer learning appears where it is used: notebook 4.6 closes with fine-tuning a pretrained encoder, which is transfer learning in miniature. Large language models and AI agents are covered in Chapter 6.

### Learning outcomes

By the end of this chapter, you will be able to:
- Implement, train, and evaluate neural networks in PyTorch, from a single neuron to a transformer encoder.
- Diagnose training problems from learning curves and fix them.
- Quantify how data quality, architecture choices, and training strategy each affect model performance.
- Estimate prediction uncertainty with deep ensembles.
- Use self-supervised pretraining when labeled data is scarce.
- Choose and benchmark forecasting models with leakage-free temporal splits.
