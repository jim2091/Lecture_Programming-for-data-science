# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:33:38 2020

@author: Leejeongbin
"""


import mglearn

X, y = mglearn.datasets.make_forge()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)

y_test_hat = clf.predict(X_test)
print(y_test)
print(y_test_hat)

from sklearn.metrics import accuracy_score
y_train_hat = clf.predict(X_train)
print(accuracy_score(y_train, y_train_hat))
y_test_hat = clf.predict(X_test)
print(accuracy_score(y_test, y_test_hat))

      