# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:33:38 2020

@author: wjdql
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV

abalone = pd.read_csv('C:/Users/wjdql/Desktop/A6/abalone.csv')
concretecs = pd.read_csv('C:/Users/wjdql/Desktop/A6/concretecs.csv')
parkinsons = pd.read_csv('C:/Users/wjdql/Desktop/A6/parkinsons.csv')
skillcraft = pd.read_csv('C:/Users/wjdql/Desktop/A6/skillcraft.csv')
winequality = pd.read_csv('C:/Users/wjdql/Desktop/A6/winequality-white.csv')

d_list = [abalone, concretecs, parkinsons, skillcraft, winequality]

for d in d_list:
    X_train, X_test, y_train, y_test = train_test_split(
        d.iloc[:, :-1], d.iloc[:, -1], test_size = 0.5, random_state=42)


    pipe = Pipeline([('preprocessing', None), ('regressor', SVR())])
    hyperparam_grid = [
        {'regressor': [SVR()], 'preprocessing': [StandardScaler(), MinMaxScaler(), None],
         'regressor__gamma': [0.1, 10, 1000],
         'regressor__C': [0.001, 0.01, 0.1],
         'regressor__epsilon': [0.001, 0.01, 0.1]},
        {'regressor': [MLPRegressor(solver='adam', max_iter=1000)], 'preprocessing': [StandardScaler(), MinMaxScaler(), None],
         'regressor__hidden_layer_sizes': [(100,), (30,30), (10,10,10)],
         'regressor__alpha': [0.0001, 0.01, 1],
         'regressor__activation': ['tanh', 'relu']},
        {'regressor': [RandomForestRegressor(n_estimators=100)], 'preprocessing': [None],
         'regressor__max_features': [1, 2, 3]}]
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid = GridSearchCV(pipe, hyperparam_grid, scoring= 'neg_root_mean_squared_error', refit=True, cv=kfold)
    grid.fit(X_train, y_train)
    
    print("Best hyperparams:\n{}\n".format(grid.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))
