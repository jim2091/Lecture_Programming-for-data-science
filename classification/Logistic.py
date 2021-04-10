# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:24:17 2020

@author: wjdql
"""


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

digits = load_digits() #Multi-class classification 문제임을 파악.

#별도의 전처리는 필요없어보임.

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, stratify=digits.target, random_state=50)


train_accuracy=[]
test_accuracy=[]

C_settings = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for C in C_settings:
    clf = LogisticRegression(C=C, multi_class = 'multinomial', max_iter = 3000)
    clf.fit(X_train, y_train)
    
    y_train_hat = clf.predict(X_train)
    train_accuracy.append(accuracy_score(y_train, y_train_hat))
    
    y_test_hat = clf.predict(X_test)
    test_accuracy.append(accuracy_score(y_test, y_test_hat))
    

    
digits_dict = {'C':C_settings,
             'train_accuracy':train_accuracy,
             'test_accuracy':test_accuracy}
        


df = pd.DataFrame(digits_dict)
print(df)

df.to_csv("digits_Logistic.csv",mode='w',encoding='utf-8-sig')