# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:20:41 2020

@author: Leejeongbin
"""


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

osi = pd.read_csv('C:/Users/Leejeongbin/Desktop/HW4/online_shoppers_intention.csv')

#one hot encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(osi['Month'])
osi['Month'] = le.transform(osi['Month'])
le.fit(osi['VisitorType'])
osi['VisitorType'] = le.transform(osi['VisitorType'])
le.fit(osi['Weekend'])
osi['Weekend'] = le.transform(osi['Weekend'])
le.fit(osi['Revenue'])
osi['Revenue'] = le.transform(osi['Revenue'])
osi_target=osi['Revenue']

#단순히 차원 축소를 통한 시각화 목적이므로 별도의 split은 진행하지 않음

scaler = StandardScaler()
scaler.fit(osi.iloc[:,0:17])
osi_scaled = scaler.transform(osi.iloc[:,0:17])

# PCA

pca = PCA(n_components=2)

pca.fit(osi_scaled)
osi_pca = pca.transform(osi_scaled)

print("Original shape: {}".format(str(osi_scaled.shape)))
print("Reduced shape: {}".format(str(osi_pca.shape)))

plt.figure(figsize=(10,10))
plt.xlim(osi_pca[:,0].min(), osi_pca[:,0].max())
plt.ylim(osi_pca[:,1].min(), osi_pca[:,1].max())


colors = ['red', 'blue']
for i in range(len(osi)):
    plt.text(osi_pca[i,0], osi_pca[i,1], str(osi_target[i]),
    color = colors[osi_target[i]],
    fontdict = {'weight' : 'bold', 'size':9})
    plt.xlabel('1st PC')
    plt.ylabel('2nd PC')
    

#t-SNE
    
tsne = TSNE(random_state=2)
osi_tsne = tsne.fit_transform(osi_scaled)

plt.figure(figsize=(10,10))
plt.xlim(osi_tsne[:,0].min(), osi_tsne[:,0].max())
plt.ylim(osi_tsne[:,1].min(), osi_tsne[:,1].max())
for i in range(len(osi)):
    plt.text(osi_tsne[i,0], osi_tsne[i,1], str(osi_target[i]),
    color = colors[osi_target[i]],
    fontdict = {'weight' : 'bold', 'size':9})
    plt.xlabel('1st PC')
    plt.ylabel('2nd PC')
    
