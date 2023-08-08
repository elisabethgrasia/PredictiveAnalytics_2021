# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 23:27:21 2021

@author: elisabethputri
"""


# Common imports
import numpy as np
import os
import pandas as pd

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

np.random.seed(47)

#Common imports for PCA: 
# SVM related
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

# PCA
from sklearn.decomposition import PCA
from sklearn import preprocessing

# preprocessing
from sklearn.preprocessing import StandardScaler

Asteroid= pd.read_csv('Asteroid_Updated.csv')
df = Asteroid
temp = df

print(temp.shape)

aaa=pd.isnull(df['name'])
name_indices_nan = np.where(np.asarray(aaa)==True)[0]
processed_dataframe= temp.drop(name_indices_nan)

print("Nan in columns after removing nan in names \n",processed_dataframe.isnull().sum())

for column in processed_dataframe:
    if processed_dataframe[column].isnull().sum() > 8000:
        processed_dataframe= processed_dataframe.drop([column], axis=1)
          
print("Nan in columns after removing some columns \n",processed_dataframe.isnull().sum()
      
for column in processed_dataframe:
    if processed_dataframe[column].isnull().any():
        aaa=pd.isnull(processed_dataframe[column])
        indices_nan = np.where(np.asarray(aaa)==True)[0]
        processed_dataframe = processed_dataframe.reset_index(drop=True)
        processed_dataframe= processed_dataframe.drop(indices_nan)

print("Nan in columns after removing rows \n",processed_dataframe.isnull().sum())

pdf = processed_dataframe 
pdf.info()



## loading the packages
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# packages for MSE and train-test split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


degree = 15
X = np.c_[pdf["diameter"]]
y = np.c_[pdf["per_y"]]
train_errors = []
val_errors = []

# train_test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)


for d in range(1, degree):
    polyfeature = PolynomialFeatures(degree=d, include_bias=False)
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
        ("poly_features", polyfeature),
        ("lin_reg", lin_reg),
    ])
    polynomial_regression.fit(X_train, y_train)
    y_train_predict = polynomial_regression.predict(X_train)
    y_test_predict = polynomial_regression.predict(X_val)
    # training and test errors
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_test_predict))
    
polyfeature = PolynomialFeatures(degree=3, include_bias=False)
lin_reg = LinearRegression()
polynomial_regression = Pipeline ([
    ("poly_features", polyfeature),
    ("lin_reg", lin_reg),
])
polynomial_regression.fit(X, y)
y_predict = polynomial_regression.predict(X)

plt.plot(y_predict, 'r')
plt.plot(y, 'b')
plt.show()
    
# the plot
plt.plot(range(1, degree), train_errors, "-+", linewidth = 2, markersize = 12, label="train")
plt.plot(range(1, degree), val_errors, "-+", linewidth=3, markersize = 12, label = "test")
plt.legend(loc="upper right", fontsize = 14)
plt.xlabel("Degree of polynomial", fontsize = 14)
plt.ylabel("MSE", fontsize = 14)
plt.title('Polynomial Regression Error for diameter vs orbital period')
plt.grid(True)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge



X_dummy = pd.get_dummies(data = pdf, drop_first = True)
X_dummy.head()
y = pdf["diameter"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummy)
lin_reg_2b = LinearRegression()
lin_reg_2b.fit(X_scaled, y)
print(lin_reg_2b.coef_)

ridge_coef = []
alphas = np.arange(0, 10, 0.2)

for alpha in alphas:
    ridge_reg = Ridge(alpha=alpha, random_state = 47)
    ridge_reg.fit(X_scaled, y)
    ridge_coef.append(ridge_reg.coef_[0])

ridge_coef2 = np.array(ridge_coef)
ridge_coef2.shape

nFeatures = X_scaled.shape[1]
cmap = mpl.cm.tab10(np.linspace(0, 1, nFeatures))

plt.figure(figsize=(7, 7))
for i in range(nFeatures):
    plt.plot(alphas, ridge_coef2[:, i], color=cmap[i,:], label=X_dummy.columns[i])
plt.xlabel("$\lambda$", fontsize = 18)
plt.ylabel("$\hat{w}^{ridge}$  ", rotation=0, fontsize = 18)
plt.title("Ridge regression coefficients")
plt.legend(loc=2, fontsize=10)
plt.axis([0, 10, -6, 9])
plt.show()


alphas = np.arange(0, 10, 0.2)
test_error = []

for alpha in alphas:
    ridge_reg = Ridge(alpha=alpha, random_state=42)
    ridge_reg.fit(x_train_scaled, y_train)
    ypred = ridge_reg.predict(x_test_scaled)
    test_error.append(mean_squared_error(y_test, ypred))

indA = np.argmin(test_error)
print('Minimum test error: ', test_error[indA], ' at alpha equals:', alphas[indA])

plt.plot(alphas, test_error)
plt.show()




alphas = np.arange(0.05, 4, 0.05)
test_error = []

for alpha in alphas:
    lasso_reg = Lasso(alpha=alpha, random_state=42, max_iter = 1e4)
    lasso_reg.fit(x_train_scaled, y_train)
    ypred = lasso_reg.predict(x_test_scaled)
    test_error.append(mean_squared_error(y_test, ypred))  
    
indA = np.argmin(test_error)
print('Minimum test error: ', test_error[indA], ' at alpha equals:', alphas[indA])

plt.plot(alphas, test_error)
plt.show()

# Normalising data to be calculated in CV error
from sklearn.model_selection import KFold

for alpha in alphas:
    kfold_err = []
    kf = KFold(n_splits=5, shuffle=True, random_state = 42)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]



lasso_reg2 = Lasso(alpha=2.2, random_state=42)
lasso_reg2.fit(x_train_scaled, y_train)  
lasso_reg2.coef_



