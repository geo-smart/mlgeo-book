###  **Deep Learning Exploration with AI-Ready Datasets**

**Objective**: Evaluate students' ability to explore and implement deep learning models for their AI-ready datasets, benchmark these models against classical machine learning methods, deliver high-quality software, and analyze results critically.

---

#### **1. Dataset Preparation and Exploration (10%)**
- **AI-Ready Data Utilization (4%)**: Demonstrates the use of the previously prepared AI-ready dataset effectively, ensuring consistency in preprocessing across models. The report should include a description of the input  \& physical meaning, their modalities, dimension.
- **Exploratory Data Analysis (EDA) (3%)**: Includes visualizations and summaries to understand data distribution, temporal/spatial features, or domain-specific nuances.
- **Problem Setup (3%)**: Clearly defines the problem (e.g., regression/classification) and aligns the data with deep learning requirements (e.g., reshaping for CNNs, sequence creation for RNNs).

---

#### **2. Model Benchmarking Against CML (10%)**
- **Baseline Models (5%)**: Reports results from previous classical machine learning benchmarks (e.g., random forests, SVMs, or gradient boosting) with minimal additional work.
- **Performance Comparison (5%)**: Provides a high-level comparison of CML methods to deep learning models using relevant metrics (e.g., accuracy, RMSE, F1-score).

---

#### **3. Model Architecture Exploration (35%)**
- **Implementation and Justification (8%)**: Test canonical architectures, try at least three deep learning architectures (e.g., FCN, CNN, RNN, U-Net). Justifies architecture choice based on dataset and problem type. In the report, write out the network overall architectures with the dimensions, the choice of activation functions.
- **Parameter Tuning (8%)**: Explores hyperparameters (e.g., learning rate, number of layers, filter sizes) and documents experiments systematically.
- **Incorporation of Physics-Informed Loss (4%)**: Implements physics-informed loss where appropriate, with a clear explanation of its relevance to the geoscientific problem.
- **Innovation and Complexity (8%)**: Includes innovative approaches like hybrid architectures, custom loss functions, or data augmentation specific to geoscience applications.
- **Exploration and Analysis (7%)**: Investigates losses, activation functions, and layer design, demonstrating a strong understanding of model behavior.

---

#### **4. Performance Evaluation (20%)**
- **Quantitative Evaluation (6%)**: Provides comprehensive metrics for all models, including accuracy, precision, recall, F1, RMSE, or domain-specific measures. Note that multi-class classifications have precision and recall values for all classes. Write out the choice of optimizer, learning rate, and batch size.
- **Generalization Testing (7%)**: Evaluates model performance on unseen or out-of-distribution data and discusses overfitting or underfitting tendencies.
- **Discussion on Narrow vs. General AI (4%)**: Reflects on the role of the implemented models as narrow AI and contrasts this with the broader concept of general AI, tying the discussion to the problem domain and dataset.
- **Visualization of Results (3%)**: Uses visualizations like confusion matrices, ROC curves, loss vs. epoch plots, or spatial/temporal error maps.

---

#### **5. Software Delivery and Code Quality (20%)**
- **Standard Practice for Training Neural Networks (10%)**:
    - Code is modular and organized in a single notebook for each clear section.
    - Clearly address the 1) data preparation with a description of training, validation, and testing data, 2)  the model architecture and design, 3)  the training strategies (batch size, optimizer) and show learning curves, 4) evaluation of performance and generalization. 
    - Explores hyperparameters (e.g. model choice, training parameters) and discuss how they help training by interpretation of learning curves.
- **Saving Results (5%)**:
    - Saves model weights, training logs, and performance metrics to a CSV/JSON file.
- **Code Quality and Documentation (5%)**:
    - Follows best practices for readability, commenting, and modularity, ensuring reproducibility.
    - The repository README should clearly state how to run the notebooks and in which order.

---

#### **6. Reporting and Interpretation (5%)**
- **Scientific Communication (3%)**: Presents results clearly and concisely in a well-structured report or notebook, with appropriate figures, tables, and explanations.
- **Domain Insights (2%)**: Discusses implications of findings for geoscience, such as physical relevance, data limitations, or potential for real-world applications.

---

#### **7. Ethical and Computational Considerations (5%)**
- **Computational Efficiency (3%)**: Documents computational costs (e.g., training time, memory usage) and discusses their impact on model choice.
- **Ethical Considerations (2%)**: Reflects on ethical implications, including biases in data, transparency of model predictions, and alignment with societal goals.

---

**Total: 100%**
