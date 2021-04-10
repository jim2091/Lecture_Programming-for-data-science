# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 01:13:43 2020

@author: Leejeongbin
"""


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

spambase = pd.read_csv('C:/Users/Leejeongbin/Desktop/spamdata/data/spambase.data',header=None)


X_train, X_test, y_train, y_test = train_test_split(
    spambase.loc[:,0:56], spambase[57], stratify=spambase[57], random_state=50)
'''
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

a_list = []
n_list = []
training_accuracy = []
test_accuracy = []

a_settings = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
hidden_layer_sizes_settings = [(100), (10,10), (1000), (10,10,10)]

for a in a_settings:
    for n in hidden_layer_sizes_settings:
    
        clf = MLPClassifier(max_iter=1000,hidden_layer_sizes=n, alpha=a, random_state=50)
        clf.fit(X_train_scaled, y_train)
        
        a_list.append(a)
        n_list.append(n)
        
        y_train_hat = clf.predict(X_train_scaled)
        training_accuracy.append(accuracy_score(y_train, y_train_hat))
    
        y_test_hat = clf.predict(X_test_scaled)
        test_accuracy.append(accuracy_score(y_test, y_test_hat))
        

digits_dict = {'a':a_list,
               'n':n_list,
               'training_accuracy':training_accuracy,
               'test_accuracy':test_accuracy}

df = pd.DataFrame(digits_dict)
print(df)

df.to_csv("neural_network.csv",mode='w',encoding='utf-8-sig')
'''