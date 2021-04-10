# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:12:43 2020

@author: wjdql
"""


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

digits = load_digits() #Multi-class classification 문제임을 파악.

#별도의 전처리는 필요없어보임.

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, stratify=digits.target, random_state=50)

training_accuracy = []
test_accuracy = []

n_settings = [1, 2, 5, 10, 20, 50, 100, 200]

for n in n_settings:
    clf = RandomForestClassifier(n_estimators=n, random_state=0)
    clf.fit(X_train, y_train)
    
    y_train_hat = clf.predict(X_train)
    training_accuracy.append(accuracy_score(y_train, y_train_hat))
    
    y_test_hat = clf.predict(X_test)
    test_accuracy.append(accuracy_score(y_test, y_test_hat))
    

    
digits_dict = {'n':n_settings,
             'training_accuracy':training_accuracy,
             'test_accuracy':test_accuracy}
        


df = pd.DataFrame(digits_dict)
print(df)

df.to_csv("digits_forest.csv",mode='w',encoding='utf-8-sig')