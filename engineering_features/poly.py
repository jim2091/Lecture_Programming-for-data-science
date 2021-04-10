# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:01:41 2020

@author: Leejeongbin
"""


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

housing = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=42)

poly = PolynomialFeatures(degree = 2, interaction_only = False ,include_bias=False)
poly.fit(X_train)

X_train_poly = poly.transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
scaler.fit(X_train_poly)
X_train_poly_scaled=scaler.transform(X_train_poly)
X_test_poly_scaled=scaler.transform(X_test_poly)


reg = LinearRegression()
reg.fit(X_train_poly_scaled, y_train)


y_train_hat = reg.predict(X_train_poly_scaled)
y_test_hat = reg.predict(X_test_poly_scaled)

print('train R_Square : ', r2_score(y_train, y_train_hat))

print('test R_Square : ', r2_score(y_test, y_test_hat))
