# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:10:08 2021

@author: elisabethputri
"""

import numpy as np
import pandas

np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize = 14)
mpl.rc('xtick', labelsize = 12)
mpl.rc('ytick', labelsize = 12)

# Import and screening the data
wine = pandas.read_csv('winequality-white.csv', delimiter=';')
wine.describe()
wine.shape
wine.head()

X = wine.drop(['quality', 'alcohol'], axis=1).to_numpy()
y = np.c_[wine['alcohol']]

# plotting the data (using all x values)
import matplotlib.pyplot as plt
plt.plot(X,y)
plt.xlabel('$X$')
plt.ylabel('$y$')
plt.show()

# splitting the data into training data and test data

import sklearn.model_selection as model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.65,test_size=0.35, random_state=101)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Solving linear regression using closed form solution
X_m = np.c_[np.ones((4898,1)), X]
w1 = np.linalg.pinv(X_m).dot(y)
print(w1)

X_new = np.array([[0], [2]]) # min n max xvalues
X_new_m = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_m.dot(w1) # predicting y values for each X_new values

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.show()

# Solving with scikit-learn
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)


# Accuracy checking training and test data
lin_reg.score(X_train, y_train)
ypred = lin_reg.predict(X_test)

# is it a good model?
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, ypred)


# particular choices in creating model?  why?
# Choose the model with minimal MSE because it means the error is minimized