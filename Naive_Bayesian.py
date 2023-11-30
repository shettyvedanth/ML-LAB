import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('C:/Users/J/Downloads/covid.csv')
print(data)

# Label encoding
le = preprocessing.LabelEncoder()
pc = le.fit_transform(data['pc'].values)
wbc = le.fit_transform(data['wbc'].values)
mc = le.fit_transform(data['mc'].values)
ast = le.fit_transform(data['ast'].values)
bc = le.fit_transform(data['bc'].values)
ldh = le.fit_transform(data['ldh'].values)
y = le.fit_transform(data['diagnosis'].values)

# Feature matrix
X = np.array(list(zip(pc, wbc, mc, ast, bc, ldh)))
print(X)

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)

# Train Naive Bayes classifier
naivee = MultinomialNB()
naivee.fit(xtrain, ytrain)

# Predictions
ypred = naivee.predict(xtest)

# Model evaluation
print("Accuracy: ", accuracy_score(ytest, ypred))
print("Classification Report:\n", classification_report(ytest, ypred))

# ROC curve
lr_probs = naivee.predict_proba(xtest)[:, 1]
lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)

# Plot ROC curve
plt.plot(lr_fpr, lr_tpr, marker='.', label='Naive Bayes Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
