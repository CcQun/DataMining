from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv('iris.csv')

data = dataset.iloc[:, 1:4]

y_pred = KMeans(n_clusters=3, init='random').fit_predict(data)

ax = plt.subplot(111, projection='3d')

mark = ['r', 'b', 'g']

for i in range(len(y_pred)):
    ax.scatter(data.values[i][0], data.values[i][1], data.values[i][2], c=mark[y_pred[i]])

plt.show()
