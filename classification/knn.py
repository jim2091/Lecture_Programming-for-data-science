# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:51:21 2020

@author: wjdql
"""


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

digits = load_digits() 

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, stratify=digits.target, random_state=50)

k_list=[]
p_list=[]
weight_list=[]
train_accuracy_list=[]
test_accuracy_list=[]

for k in range(1,16):
    for p in range(1,6):
        clf = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = p, weights='uniform')
        clf.fit(X_train, y_train)
        y_train_hat = clf.predict(X_train)
        y_test_hat = clf.predict(X_test)
        k_list.append(k)
        p_list.append(p)
        weight_list.append('uniform')
        train_accuracy_list.append(accuracy_score(y_train, y_train_hat))
        test_accuracy_list.append(accuracy_score(y_test, y_test_hat))
        


for k in range(1,16):
    for p in range(1,6):
        clf = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = p, weights='distance')
        clf.fit(X_train, y_train)
        y_train_hat = clf.predict(X_train)
        y_test_hat = clf.predict(X_test)
        k_list.append(k)
        p_list.append(p)
        weight_list.append('distance')
        train_accuracy_list.append(accuracy_score(y_train, y_train_hat))
        test_accuracy_list.append(accuracy_score(y_test, y_test_hat))
        
digits_dict = {'k':k_list,
             'p':p_list,
             'weight':weight_list,
             'train_accuracy':train_accuracy_list,
             'test_accuracy':test_accuracy_list}
        


df = pd.DataFrame(digits_dict)
print(df)

df.to_csv("digits_knn.csv",mode='w',encoding='utf-8-sig')