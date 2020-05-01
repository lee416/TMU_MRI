
#### exploratory data
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
iris = datasets.load_iris()
type(iris)
# > sklearn.utils.Bunch
print(iris.keys())
# > dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
type(iris.data), type(iris.target)
# > (numpy.ndarray, numpy.ndarray)
iris.data.shape
# > (150, 4)
iris['target'].shape



#### classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

plt.style.use('ggplot')
iris = datasets.load_iris()

knn =KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'],iris[''])
# check data shape
iris['data'].shape
iris['target'].shape

X_new = (4,3)

prediction = knn.predict(X_new)
X_new.shape


