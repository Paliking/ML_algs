# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 19:24:50 2016

@author: Pablo
"""

from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


# import some data to play with
iris = datasets.load_iris()
X = iris.data # all features
y = iris.target

X = scale(X)
#print(X)

pca = PCA(n_components=3)
XX = pca.fit_transform(X)


var = pca.explained_variance_ratio_
print ('zachovana variancia: ',var.sum())

var1 = np.cumsum(np.round(var, decimals=4)*100)
plt.plot(np.arange(len(var1))+1,var1)
plt.show()

plt.scatter(XX[:, 0], XX[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA')
plt.show()