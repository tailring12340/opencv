import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

from sklearn.model_selection import train_test_split

col1 = 0
col2 = 1

X = iris.data[:,[col1,col2]] 
y = iris.target.copy()
y[y==2] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X_train.shape", X_train.shape, "X_test.shape", X_test.shape, "y_train.shape", y_train.shape, "y_test.shape", y_test.shape)

plt.figure(figsize=[10,8])
plt.scatter(X[:,0], X[:,1], c=y)
plt.colorbar()

from sklearn.svm import LinearSVC

model = LinearSVC(C=1)
model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(score)

import mglearn

plt.figure(figsize=[10,8])
mglearn.plots.plot_2d_classification(model, X_train, eps=0.5, cm='spring')
mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)

plt.show()
