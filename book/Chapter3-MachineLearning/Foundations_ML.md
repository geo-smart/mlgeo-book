# 3.1 Choosing a Machine Learning Approach


This chapter aims to introduce key concepts in Machine Learning: Classification, Regression, strategies to train these models
 
 We will discuss how these methods apply to geoscientific data and provide practical, hands-on exercises using synthetic datasets.

## 5.1 Classification in Geosciences

### Definition

Classification is a supervised learning task where the **objective is to assign input data into one of several predefined categories or classes** based on their features. It involves learning a mapping from input features to discrete class labels using a labeled dataset.

In geosciences, classification is essential for tasks such as:

* **Lithological Mapping**: Identifying different rock types.
* **Land Cover Classification**: Categorizing terrain into forest, urban areas, water bodies, etc.
* **Mineral Prospectivity Mapping**: Classifying areas based on mineral potential.

### Common Algorithms

* **Decision Trees**: Hierarchical models that make decisions based on feature values.
* **k-Nearest Neighbors (k-NN)**: Classifies data points based on the classes of their nearest neighbors.
* **Support Vector Machines (SVM)**: Finds the optimal hyperplane that separates classes.
* **Artificial Neural Networks (ANN)**: Models inspired by biological neural networks capable of capturing complex patterns.

### Practical Exercise: Synthetic Lithology Classification

Objective: Classify synthetic rock samples into three lithology types based on density and magnetic susceptibility.

#### Data Generation

We will create a synthetic dataset representing three rock types: Granite, Basalt, and Sandstone. Each type will have characteristic values for density and magnetic susceptibility.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples per class
n_samples = 100

# Generate features for Granite
granite_density = np.random.normal(2.7, 0.05, n_samples)
granite_susceptibility = np.random.normal(0.0001, 0.00005, n_samples)
granite_label = ['Granite'] * n_samples

# Generate features for Basalt
basalt_density = np.random.normal(3.0, 0.05, n_samples)
basalt_susceptibility = np.random.normal(0.001, 0.0002, n_samples)
basalt_label = ['Basalt'] * n_samples

# Generate features for Sandstone
sandstone_density = np.random.normal(2.4, 0.05, n_samples)
sandstone_susceptibility = np.random.normal(0.00005, 0.00002, n_samples)
sandstone_label = ['Sandstone'] * n_samples

# Combine data
density = np.concatenate([granite_density, basalt_density, sandstone_density])
susceptibility = np.concatenate([granite_susceptibility, basalt_susceptibility, sandstone_susceptibility])
labels = np.concatenate([granite_label, basalt_label, sandstone_label])

# Create DataFrame
data = pd.DataFrame({
    'Density': density,
    'Magnetic Susceptibility': susceptibility,
    'Lithology': labels
})

Exercise Steps

	1.	Data Visualization
Plot the data to visualize the distribution of rock types based on their physical properties.

import seaborn as sns

sns.scatterplot(
    x='Density',
    y='Magnetic Susceptibility',
    hue='Lithology',
    data=data
)
plt.title('Rock Types Based on Density and Magnetic Susceptibility')
plt.show()


	2.	Data Splitting
Split the dataset into training and testing sets.

X = data[['Density', 'Magnetic Susceptibility']]
y = data['Lithology']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


	3.	Model Training
Train a k-NN classifier on the training data.

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


	4.	Model Evaluation
Evaluate the classifier on the test data.

y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


	5.	Discussion
	•	Analyze the confusion matrix and classification report.
	•	Discuss possible reasons for misclassifications.
	•	Consider how overlapping feature distributions might affect the model.

5.2 Regression in Geosciences

5.2.1 Definition

Regression is a supervised learning task focused on predicting continuous numerical values based on input features. It models the relationship between dependent and independent variables.

Applications in geosciences include:

	•	Groundwater Level Prediction
	•	Soil Property Estimation
	•	Temperature and Precipitation Modeling

5.2.2 Common Algorithms

	•	Linear Regression: Models linear relationships between variables.
	•	Polynomial Regression: Captures non-linear relationships by including polynomial terms.
	•	Support Vector Regression (SVR): Extension of SVM for regression tasks.
	•	Neural Networks: Handle complex, non-linear relationships.

5.2.3 Practical Exercise: Synthetic Soil Moisture Prediction

Objective: Predict soil moisture content based on environmental factors.

Data Generation

We will simulate soil moisture influenced by temperature and humidity.

# Number of samples
n_samples = 300

# Generate environmental variables
temperature = np.random.uniform(15, 35, n_samples)  # in degrees Celsius
humidity = np.random.uniform(30, 90, n_samples)      # in percentage

# Generate soil moisture as a function of temperature and humidity
soil_moisture = (
    0.5 * humidity - 0.3 * temperature + np.random.normal(0, 2, n_samples)
)

# Create DataFrame
data = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Soil Moisture': soil_moisture
})

Exercise Steps

	1.	Data Visualization
Plot soil moisture against temperature and humidity.

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Temperature'], data['Humidity'], data['Soil Moisture'])
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Humidity (%)')
ax.set_zlabel('Soil Moisture')
plt.show()


	2.	Data Splitting

X = data[['Temperature', 'Humidity']]
y = data['Soil Moisture']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


	3.	Model Training
Train a linear regression model.

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)


	4.	Model Evaluation
Evaluate the model using Mean Squared Error (MSE) and R² score.

from sklearn.metrics import mean_squared_error, r2_score

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


	5.	Discussion
	•	Interpret the coefficients of the regression model.
	•	Discuss the model’s ability to generalize to new data.
	•	Consider the impact of noise and how to improve the model.

5.3 Unsupervised Learning in Geosciences

5.3.1 Clustering for Classification

In unsupervised learning, we aim to discover patterns or groupings in data without predefined labels. Clustering algorithms group data points based on feature similarity.

5.3.2 Practical Exercise: Clustering Synthetic Seismic Data

Objective: Cluster synthetic seismic events to identify potential fault zones.

Data Generation

We will simulate seismic events with features such as magnitude and depth.

# Number of samples
n_samples = 200

# Generate seismic events in two clusters
# Cluster 1: Shallow, lower magnitude
magnitude_cluster1 = np.random.normal(3.0, 0.5, int(n_samples * 0.6))
depth_cluster1 = np.random.normal(5, 2, int(n_samples * 0.6))

# Cluster 2: Deeper, higher magnitude
magnitude_cluster2 = np.random.normal(5.0, 0.5, int(n_samples * 0.4))
depth_cluster2 = np.random.normal(15, 3, int(n_samples * 0.4))

# Combine data
magnitude = np.concatenate([magnitude_cluster1, magnitude_cluster2])
depth = np.concatenate([depth_cluster1, depth_cluster2])

# Create DataFrame
data = pd.DataFrame({'Magnitude': magnitude, 'Depth': depth})

Exercise Steps

	1.	Data Visualization
Plot the seismic events to observe potential clusters.

plt.scatter(data['Depth'], data['Magnitude'])
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.title('Seismic Events')
plt.show()


	2.	Clustering
Apply K-means clustering to the data.

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Depth', 'Magnitude']])


	3.	Visualization of Clusters

sns.scatterplot(
    x='Depth', y='Magnitude', hue='Cluster', data=data, palette='Set1'
)
plt.title('Clustered Seismic Events')
plt.show()


	4.	Discussion
	•	Analyze the characteristics of each cluster.
	•	Consider geological interpretations, such as different fault zones.
	•	Discuss limitations and potential improvements.

5.3.3 Dimensionality Reduction for Regression

Dimensionality Reduction techniques like Principal Component Analysis (PCA) simplify high-dimensional data while retaining essential information.

Practical Exercise: PCA on Synthetic Geochemical Data

Objective: Reduce the dimensionality of multivariate geochemical data before regression.

Data Generation

We will simulate geochemical data with several correlated elements.

# Number of samples
n_samples = 150

# Simulate correlated geochemical elements
element_A = np.random.normal(50, 10, n_samples)
element_B = element_A + np.random.normal(0, 5, n_samples)
element_C = 0.5 * element_A + np.random.normal(0, 3, n_samples)
element_D = np.random.normal(20, 5, n_samples)

# Target variable: Mineral grade
grade = 2 * element_A + 1.5 * element_B + np.random.normal(0, 10, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'Element A': element_A,
    'Element B': element_B,
    'Element C': element_C,
    'Element D': element_D,
    'Grade': grade
})

Exercise Steps

	1.	PCA Application
Apply PCA to the elemental data.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features = ['Element A', 'Element B', 'Element C', 'Element D']
x = data.loc[:, features].values
y = data['Grade'].values

# Standardize the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(
    data=principal_components, columns=['PC1', 'PC2']
)


	2.	Explained Variance
Analyze the amount of variance explained by each principal component.

print(pca.explained_variance_ratio_)


	3.	Regression with Principal Components
Train a regression model using the principal components.

X_train, X_test, y_train, y_test = train_test_split(
    principal_df, y, test_size=0.3, random_state=42
)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


	4.	Discussion
	•	Evaluate the performance compared to using all original features.
	•	Discuss the trade-off between dimensionality reduction and information loss.
	•	Consider how PCA can help in dealing with multicollinearity.

5.4 Framing Supervised and Unsupervised Learning

5.4.1 Supervised Learning

	•	Classification: Predicts discrete class labels based on input features.
	•	Regression: Predicts continuous numerical values.

Key aspects:

	•	Requires labeled data for training.
	•	Focuses on learning the mapping from inputs to outputs.

5.4.2 Unsupervised Learning

	•	Clustering (for Classification): Groups data points based on similarity without predefined labels.
	•	Dimensionality Reduction (for Regression): Simplifies data by reducing the number of features.

Key aspects:

	•	Works with unlabeled data.
	•	Aims to uncover hidden structures or patterns.

5.4.3 Integration in Geosciences

Both supervised and unsupervised learning techniques are vital in geosciences for:

	•	Data Exploration: Understanding underlying patterns before applying predictive models.
	•	Feature Engineering: Creating meaningful input features for models.
	•	Model Improvement: Enhancing model performance through dimensionality reduction.

Conclusion

This chapter provided a foundational understanding of classification and regression within the context of geoscientific AI. Through practical exercises with synthetic data, you have had the opportunity to apply these concepts hands-on. These exercises are designed to reinforce your learning and prepare you for tackling real-world geoscientific challenges using machine learning.

Exercises Summary

	•	Classification Exercise: Classify synthetic lithology types using k-NN.
	•	Regression Exercise: Predict soil moisture content from temperature and humidity.
	•	Clustering Exercise: Identify clusters in synthetic seismic data.
	•	Dimensionality Reduction Exercise: Apply PCA to geochemical data before regression.

References

	•	Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
	•	Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
	•	Witten, I. H., Frank, E., Hall, M. A., & Pal, C. J. (2016). Data Mining: Practical Machine Learning Tools and Techniques. Morgan Kaufmann.

Note: The synthetic datasets provided in this chapter are simplified representations meant for educational purposes. In practice, geoscientific data can be more complex and may require advanced preprocessing and analysis techniques.