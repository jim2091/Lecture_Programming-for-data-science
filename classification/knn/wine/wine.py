# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 02:20:33 2020

@author: Leejeongbin
"""


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd



wine = pd.read_csv('C:/Users/wjdql/Desktop/A6/winequality-white.csv')

X_train, X_test, y_train, y_test = train_test_split(
    wine.iloc[:,0:11], wine.iloc[:,11], random_state=2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

k_list=[]
p_list=[]
weight_list=[]
train_accuracy_list=[]
test_accuracy_list=[]

for k in range(1,5):
    for p in range(1,3):
        clf = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = p, weights='uniform')
        clf.fit(X_train_scaled, y_train)
        y_train_hat = clf.predict(X_train_scaled)
        y_test_hat = clf.predict(X_test_scaled)
        k_list.append(k)
        p_list.append(p)
        weight_list.append('uniform')
        train_accuracy_list.append(accuracy_score(y_train, y_train_hat))
        test_accuracy_list.append(accuracy_score(y_test, y_test_hat))
        


wine_dict = {'k':k_list,
             'p':p_list,
             'weight':weight_list,
             'train_accuracy':train_accuracy_list,
             'test_accuracy':test_accuracy_list}
        


df = pd.DataFrame(wine_dict)
print(df)
