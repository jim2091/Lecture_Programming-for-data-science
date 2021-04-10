# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:07:19 2020

@author: wjdql
"""

import pandas as pd
from sklearn.model_selection import train_test_split
winequality = pd.read_csv('C:/Users/wjdql/Desktop/A6/winequality-white.csv')
X_train, X_test, y_train, y_test = train_test_split(
    winequality.iloc[:,0:11], winequality.iloc[:,11], random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

pipe = Pipeline([('preprocessing', None), ('classifier', SVC())])
hyperparam_grid = [
    {'classifier': [MLPClassifier(solver='adam', max_iter=3000)], 'preprocessing': [StandardScaler(), MinMaxScaler(), None],
     'classifier__hidden_layer_sizes': [(100), (10,10), (1000), (10,10,10)],
     'classifier__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
     'classifier__activation': ['tanh', 'relu']},
    {'classifier': [SVC()], 'preprocessing': [StandardScaler(), MinMaxScaler(), None],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=100)], 'preprocessing': [None],
     'classifier__max_features': [1, 2, 3]},
    {'classifier': [KNeighborsClassifier(metric='minkowski')], 'preprocessing': [StandardScaler(), MinMaxScaler(), None],
     'classifier__n_neighbors': [3, 5, 7, 9, 11],
     'classifier__p': [1, 2, 3, 4, 5],
     'classifier__weights': ['uniform', 'distance']}]

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, hyperparam_grid, scoring='accuracy', refit=True, cv=kfold)
grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))