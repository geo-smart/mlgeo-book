# Final-Project Milestone: Deep Learning on Your AI-Ready Dataset

**Objective**: Demonstrate that you can implement, train, diagnose, and critically evaluate deep learning models on your own AI-ready dataset, benchmark them against classical machine learning, and deliver reproducible software.

Everything required below is taught with working code in this chapter. When a requirement names a notebook, reuse that notebook's pattern on your own data.

---

## 1. Dataset Preparation and Exploration (10%)

- **AI-ready data utilization (4%)**: Use the AI-ready dataset you prepared earlier, with consistent preprocessing across all models. Describe the inputs, their physical meaning, modalities, and dimensions.
- **Exploratory data analysis (3%)**: Visualizations and summaries of data distribution and temporal/spatial structure.
- **Problem setup (3%)**: Define the task (regression or classification) and shape the data for each architecture (windowing for sequence models, reshaping for CNNs).

---

## 2. Benchmarking Against Classical ML (10%)

- **Baseline models (5%)**: Report your classical ML results from the previous milestone (random forest, gradient boosting, or similar). Minimal new work; the point is a reference line.
- **Performance comparison (5%)**: Compare classical and deep models with the same metrics on the same splits.

---

## 3. Model Architecture Exploration (30%)

- **At least three architectures (8%)**: Implement and train at least three of: MLP, 1-D or 2-D CNN, LSTM, transformer encoder, autoencoder + classification/regression head. All five are built in notebooks 4.1-4.6. U-Net is optional, not required. Justify each choice against your data and problem type. Write out each architecture with layer dimensions and activation functions.
- **Hyperparameter exploration (7%)**: Explore learning rate, layer sizes, and other hyperparameters systematically, following the notebook 4.5 lab. Document every experiment; a table of runs beats scattered prose.
- **Ablation study (5%)** — required: Remove one component of your best model (a layer block, dropout, batch normalization, a feature group, or a loss term) and report the change in performance. One ablation done carefully is enough; state what it tells you about the component.
- **Physics-informed loss (4%)**: Add a physics-informed or otherwise domain-aware loss term where your problem supports one (notebook 4.7 pattern). If your problem does not support one, say why in a short paragraph; a well-argued negative answer earns full credit.
- **Innovation (6%)**: Hybrid architectures, custom loss functions, or geoscience-specific data augmentation.

---

## 4. Performance Evaluation (20%)

- **Quantitative evaluation (6%)**: Metrics for all models: accuracy, precision, recall, F1, RMSE, or domain-specific measures. Multi-class problems report per-class precision and recall. State optimizer, learning rate, and batch size for every trained model.
- **Generalization and out-of-distribution testing (7%)**: Evaluate on unseen or out-of-distribution data and discuss overfitting or underfitting.
- **Chosen experiment (4%)** — pick ONE:
  - *Fine-tuning vs from scratch*: pretrain an encoder on your data without labels (autoencoder or masked reconstruction, notebook 4.6 pattern), then fine-tune with a small labeled fraction and compare against training from scratch on the same fraction.
  - *Deep-ensemble uncertainty*: train your best model from 5 random seeds, report ensemble-mean performance and per-sample prediction variance, and show which samples the ensemble disagrees on (notebook 4.5 pattern).
- **Visualization of results (3%)**: Confusion matrices, loss-vs-epoch plots, error maps, or equivalents.

---

## 5. Software Delivery and Code Quality (15%)

- **Standard training practice (7%)**: Modular code, one notebook per clear section. Address: (1) data preparation with train/validation/test description, (2) model architecture and design, (3) training strategy (batch size, optimizer, scheduler) with learning curves, (4) evaluation and generalization.
- **Saving results (4%)**: Save model weights, training logs, and performance metrics to CSV/JSON files committed with the repository.
- **Code quality and documentation (4%)**: Readable, commented, reproducible. The repository README states how to run the notebooks and in which order.

---

## 6. Reporting and Interpretation (10%)

- **Scientific communication (3%)**: Clear, concise report with appropriate figures and tables.
- **Domain insights (2%)**: What the results mean for the geoscience problem: physical relevance, data limitations, potential applications.
- **Training-diagnostics appendix (5%)** — required: Learning curves for your final model AND for at least one failed or flawed run (diverging loss, overfitting, a learning rate that crawled). Diagnose the failed run in 2-3 sentences using the vocabulary of notebook 4.5. Failed runs are evidence of systematic work, not something to hide.

---

## 7. Computational and Ethical Considerations (5%)

- **Compute reporting (3%)**: Training time, hardware used, and memory footprint for each model, and how compute cost influenced your choices.
- **Ethics and AI-use disclosure (2%)**: Reflect on biases in your data and transparency of predictions. Include a mandatory one-paragraph AI-use disclosure: which AI assistants or code-generation tools you used, for what parts of the work, and how you verified their output. Using AI tools is allowed; not disclosing them is not.

---

**Total: 100%**
