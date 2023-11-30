# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

# Reading the dataset from a CSV file into a pandas DataFrame
data = pd.read_csv("C:/Users/J/Downloads/glass.csv")

# Displaying the first few rows of the dataset and its information
print(data.head())
print(data.info())

# Preparing data: separating features (x) and target variable (y)
x = data.drop('Type', axis=1)
y = data.Type

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Creating an SVM model with a linear kernel
ml = SVC(kernel='linear')

# Training the SVM model on the training data
ml.fit(x_train, y_train)

# Printing the support vectors and the number of support vectors for each class
print(ml.support_vectors_)
print(ml.n_support_)

# Making predictions on the test set
y_pred = ml.predict(x_test)

# Evaluating the performance of the SVM model
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
