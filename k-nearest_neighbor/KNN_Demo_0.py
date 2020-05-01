from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load dataset
iris = datasets.load_iris()
#print(iris) # looks
#print(iris.keys()) # header
#iris.data.shape # matrix size
#iris.target.shape 


# value input
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
#print(df) # zero matrix

# show data figure
figure = pd.plotting.scatter_matrix(df, c = y, figsize = [6, 6],s=150, marker = 'D')


# KNN demo
# 01
X_new = np.array([5.6, 2.8, 3.9, 1.1],
                 [5.7, 2.6, 3.8, 1.3],
                 [4.7, 3.2, 1.3, 0.2])

knn = KNeighborsClassifier(n_neighbors=6)
prediction = knn.predict(X_new)
#print('Prediction {}'.format(prediction))


# Iris
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])
prediction = knn.predict(X)
print('Prediction {}'.format(prediction))




