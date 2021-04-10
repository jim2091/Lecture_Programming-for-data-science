# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:20:41 2020

@author: Leejeongbin
"""


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
iris = load_iris()

from sklearn.pipeline import Pipeline

p = Pipeline([("s", StandardScaler()), ("m", LinearRegression)])

'''
#단순히 차원 축소를 통한 시각화 목적이므로 별도의 split은 진행하지 않음

scaler = StandardScaler()
scaler.fit(iris.data)
iris_scaled = scaler.transform(iris.data)

# PCA

pca = PCA(n_components=2)

pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)

print("Original shape: {}".format(str(iris_scaled.shape)))
print("Reduced shape: {}".format(str(iris_pca.shape)))

plt.figure(figsize=(10,10))
plt.xlim(iris_pca[:,0].min(), iris_pca[:,0].max())
plt.ylim(iris_pca[:,1].min(), iris_pca[:,1].max())


colors = ['red', 'blue', 'green']
for i in range(len(iris.data)):
    plt.text(iris_pca[i,0], iris_pca[i,1], str(iris.target[i]),
    color = colors[iris.target[i]],
    fontdict = {'weight' : 'bold', 'size':9})
    plt.xlabel('1st PC')
    plt.ylabel('2nd PC')
    
    
#t-SNE
    
tsne = TSNE(random_state=2)
iris_tsne = tsne.fit_transform(iris.data)

plt.figure(figsize=(10,10))
plt.xlim(iris_tsne[:,0].min(), iris_tsne[:,0].max())
plt.ylim(iris_tsne[:,1].min(), iris_tsne[:,1].max())
for i in range(len(iris.data)):
    plt.text(iris_tsne[i,0], iris_tsne[i,1], str(iris.target[i]),
    color = colors[iris.target[i]],
    fontdict = {'weight' : 'bold', 'size':9})
    plt.xlabel('1st PC')
    plt.ylabel('2nd PC')
'''