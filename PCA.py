import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv("C:/Users/J/Downloads/iris.csv")

# Prepare X and Y          ;;;;
X = df.values[:, :-1]
Y = df.values[:, -1]

# Standardize the features
X_standard = (X - X.mean(axis=0)) / X.std(axis=0)

# Calculate the covariance matrix
cov = np.cov(X_standard.T)

# Eigen decomposition
lambdas, vs = np.linalg.eig(cov)

# Sort eigenvalues and eigenvectors
sorted_index = np.argsort(lambdas)[::-1]
sorted_eigenvalue = lambdas[sorted_index]
sorted_eigenvectors = vs[:, sorted_index]

# Select the desired number of components
n_components = 2
eigenvector_subset = sorted_eigenvectors[:, :n_components]

# Transform the data
X_reduced = np.dot(X_standard, eigenvector_subset)

# Print the information/variance in PC1
variance_pc1 = (sorted_eigenvalue[0] / np.sum(sorted_eigenvalue)) * 100
print("Information/Variance in PC1: {:.2f}%".format(variance_pc1))

# Scatter plot
plt.xlabel('Principal Component - 1', fontsize=12)
plt.ylabel('Principal Component - 2', fontsize=12)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y, cmap='viridis')
plt.show()
