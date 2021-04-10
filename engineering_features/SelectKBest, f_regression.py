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
from sklearn.feature_selection import SelectKBest, f_regression

# 아무것도 하지 않았을 때와 동일한 k=8에서 가장 높았음.

housing = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=42)

select = SelectKBest(f_regression, k=8)
select.fit(X_train, y_train)

X_train_selectesd = select.transform(X_train)
X_test_selectesd = select.transform(X_test)


scaler = StandardScaler()
scaler.fit(X_train_selectesd)
X_train_selectesd_scaled=scaler.transform(X_train_selectesd)
X_test_selectesd_scaled=scaler.transform(X_test_selectesd)


reg = LinearRegression()
reg.fit(X_train_selectesd_scaled, y_train)


y_train_hat = reg.predict(X_train_selectesd_scaled)
y_test_hat = reg.predict(X_test_selectesd_scaled)

print('train R_Square : ', r2_score(y_train, y_train_hat))

print('test R_Square : ', r2_score(y_test, y_test_hat))