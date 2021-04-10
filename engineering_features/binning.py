# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:01:41 2020

@author: Leejeongbin
"""


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

# 가장 영향이 크다고 판단된 MedInc에 대해서만 binning 적용. 100정도에서만 아주 약간 더 나은 모습 보임. 비효율적인듯.
housing = fetch_california_housing()

X = housing.data[:,0]
bins = np.linspace(min(X), max(X), 100)
print(format(bins))
which_bin = np.digitize(X, bins = bins)
housing.data[:,0] = which_bin

X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_scaled, y_train)

y_train_hat = reg.predict(X_train_scaled)
y_test_hat = reg.predict(X_test_scaled)

print('train R_Square : ', r2_score(y_train, y_train_hat))
print('test R_Square : ', r2_score(y_test, y_test_hat))

