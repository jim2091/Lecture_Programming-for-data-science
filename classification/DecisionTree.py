# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:50:34 2020

@author: wjdql
"""


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

digits = load_digits() #Multi-class classification 문제임을 파악.

#별도의 전처리는 필요없어보임.

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, stratify=digits.target, random_state=50)


train_accuracy=[]
test_accuracy=[]

m_settings = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

for m in m_settings:
    clf = DecisionTreeClassifier(max_depth=m, random_state=50)
    clf.fit(X_train, y_train)
    
    y_train_hat = clf.predict(X_train)
    train_accuracy.append(accuracy_score(y_train, y_train_hat))
    
    y_test_hat = clf.predict(X_test)
    test_accuracy.append(accuracy_score(y_test, y_test_hat))
    

    
digits_dict = {'m':m_settings,
             'train_accuracy':train_accuracy,
             'test_accuracy':test_accuracy}
        


df = pd.DataFrame(digits_dict)
print(df)

df.to_csv("digits_Decision_tree.csv",mode='w',encoding='utf-8-sig')