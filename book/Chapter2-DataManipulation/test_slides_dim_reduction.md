---
marp: true
theme: gaia
paginate: true
style: |
  section {
    font-size: 30px;
  }
---

### Lecture: Dimensionality Reduction in Geoscientific Data

---

**Overview**

Dimensionality reduction is a fundamental concept in data analysis, particularly important in geosciences due to the complex and high-dimensional nature of geoscientific data. This lecture will cover:

1. **What is the Dimension of Geoscientific Data?** 

2. **Why Would We Want to Reduce the Dimensionality?**

3. **How Do We Usually Perform Dimensionality Reduction?**

---

#### 1. **What is the Dimension of Geoscientific Data?**

**Apparent Dimensions vs. Intrinsic Dimensions**

- ***Apparent Dimensions***: These are the obvious dimensions in which data is collected or represented. For example:
  - **Spatial Dimensions**: Latitude, longitude, and depth or elevation.
  - **Temporal Dimension**: Time in time series data.
  - **Spectral Dimension**: Different wavelengths or frequencies in spectral data.
- ***Intrinsic Dimensions***: The minimal number of variables necessary to accurately describe or reconstruct the dataset. This refers to the underlying factors or processes that govern the data.

---

**Example: Earthquake Amplitudes from Complex Waveforms**

- **Complex Waveforms**: Seismic waveforms are high-dimensional, consisting of thousands of data points representing ground motion over time.
- **Controlling Variables**:
  - **Magnitude**: Determines the energy released and thus the amplitude of seismic waves.
  - **Distance**: Affects the attenuation of seismic waves; the farther from the source, the lower the amplitude.
- **Intrinsic Dimensionality**: Despite the complexity of the waveform, the primary factors controlling the amplitude can be reduced to just magnitude and distance.

---

#### 2. **Why Would We Want to Reduce the Dimensionality?**

**Simplification of Data**

- **Eliminate Redundancy**: High-dimensional data may contain correlated or redundant features.
- **Enhance Interpretability**: Lower-dimensional data is easier to visualize and interpret, aiding in hypothesis generation and testing.

---

**Improved Computational Efficiency**

- **Reduced Computational Load**: Fewer dimensions mean faster computations, which is crucial when dealing with large datasets.
- **Storage Efficiency**: Less storage space is required for lower-dimensional data representations.

---

**Mitigation of the Curse of Dimensionality**

- **Overfitting Prevention**: High-dimensional datasets can lead to models that overfit, capturing noise rather than the underlying pattern.
- **Statistical Reliability**: High dimensions require exponentially more data to achieve the same level of statistical confidence.

---
**Enhanced Model Performance**

- **Noise Reduction**: Dimensionality reduction can filter out noise, improving the signal-to-noise ratio.
- **Focus on Relevant Features**: By concentrating on the most informative variables, models can make more accurate predictions.

--- 

**Facilitated Data Visualization**

- **2D and 3D Plots**: Reducing data to two or three dimensions allows for visual exploration of patterns and relationships.

---

#### 3. **How Do We Usually Perform Dimensionality Reduction?**

Dimensionality reduction techniques fall into two categories:

- **Feature Selection**: Selecting a subset of the original variables based on some criteria.
- **Feature Extraction**: Transforming the data into a lower-dimensional space.

**Common Techniques**

**A. Principal Component Analysis (PCA)**

- **Purpose**: Identify the directions (principal components) that maximize variance in the data.
- **How It Works**:
  - Computes eigenvectors (principal components) and eigenvalues from the covariance matrix of the data.
  - Projects the data onto the principal components.
- **Application in Geosciences**:
  - **Climate Studies**: Identifying dominant climate patterns (e.g., El Niño Southern Oscillation).
  - **Geological Data**: Simplifying mineralogical compositions.

---

**Implementing PCA with SciKitLearn**

```python
import numpy as np
from sklearn.decomposition import PCA

# Simulated geoscientific dataset with 1000 samples and 50 features
data = np.random.rand(1000, 50)

# Apply PCA to reduce to 5 principal components
pca = PCA(n_components=5)
reduced_data = pca.fit_transform(data)

print("Explained variance ratios:", pca.explained_variance_ratio_)
```

---

**B. Autoencoders**

- **Type**: Neural network models designed for unsupervised learning.
- **Purpose**: Learn a compressed representation (encoding) of the data and reconstruct it from the encoding.
- **Application**:
  - Handling nonlinear relationships in the data.
  - Useful for large datasets where traditional methods are computationally intensive.

---
**Implementing Autoencoders with PyTorch**


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simulated geoscientific dataset
data = np.random.rand(1000, 50)
data_tensor = torch.from_numpy(data).float()

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 2)  # Bottleneck layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 50),
            nn.Sigmoid()  # Assuming data is normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = Autoencoder()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Prepare DataLoader
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch[0]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Extract the compressed representation
with torch.no_grad():
    compressed_data = model.encoder(data_tensor)
    print(f'Compressed data shape: {compressed_data.shape}')
```

**C. Variational Autoencoders (VAEs)**

- **Purpose**: VAEs are a type of autoencoder that learns the probability distribution of the data.
- **Application**: Useful for generating new data samples and capturing complex data distributions.

**Implementing a VAE with PyTorch (Optional Advanced Topic)**

---

**Practical Considerations**

- **Data Preprocessing**:
  - Normalize or standardize data to ensure each feature contributes equally.
- **Choosing the Right Technique**:
  - **Linear vs. Nonlinear Methods**: PCA handles linear relationships; autoencoders can capture nonlinear patterns.
- **Interpreting Results**:
  - Understanding the physical meaning of reduced dimensions is crucial for practical applications.

---

**Conclusion**

- **Understanding Dimensions**: Differentiating between apparent and intrinsic dimensions helps simplify complex datasets.
- **Benefits of Dimensionality Reduction**:
  - Simplifies analysis and visualization.
  - Reduces computational resources.
  - Improves model performance.
- **Techniques Using PyTorch**: Implementing dimensionality reduction techniques in PyTorch allows for seamless integration with deep learning models and GPU acceleration.

---

**Key Takeaways**

- **Dimensionality Reduction is Essential**: It's critical for effectively handling complex geoscientific data.
- **Technique Selection Matters**: Choose methods based on data characteristics and analysis goals.
- **PyTorch Advantages**: Offers flexibility and efficiency for implementing custom models.

---

**References for Further Reading**

- **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**: *Deep Learning*. MIT Press.
- **Paszke, A., et al. (2019)**: *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems.

---

**Questions and Discussions**

- **Q**: Why use PyTorch for dimensionality reduction?
- **A**: PyTorch provides dynamic computation graphs and GPU support, making it suitable for complex models and large datasets common in geosciences.

- **Q**: How do autoencoders handle nonlinearities?
- **A**: By using nonlinear activation functions and multiple layers, autoencoders can model complex, nonlinear relationships in the data.

---

**Exercises**

1. **Implement PCA with PyTorch on Real Geoscientific Data**
   - Load a dataset (e.g., mineral compositions, remote sensing imagery).
   - Apply PCA using PyTorch and visualize the principal components.

2. **Build and Train an Autoencoder on Seismic Data**
   - Use a seismic waveform dataset.
   - Implement an autoencoder in PyTorch.
   - Analyze the compressed representations and reconstruction accuracy.

3. **Experiment with Different Autoencoder Architectures**
   - Modify the number of layers and neurons.
   - Observe the effects on compression and reconstruction.

---

By understanding and applying dimensionality reduction techniques using PyTorch, geoscientists can effectively analyze complex datasets, leading to deeper insights and more robust models.