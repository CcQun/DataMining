from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('iris.csv')

for i in range(4):
    for j in range(4):
        if i != j:
            data = dataset.iloc[:, [i + 1, j + 1]]
            y_pred = KMeans(n_clusters=3, init='k-means++').fit_predict(data)
            loc = i * 4 + (j + 1)
            plt.subplot(4, 4, loc)
            mark = ['r', 'b', 'g']
            for k in range(len(y_pred)):
                plt.scatter(data.values[k][0], data.values[k][1], c=mark[y_pred[k]], s=5)
        else:
            data = dataset.iloc[:, i + 1]
            loc = i * 4 + (j + 1)
            plt.subplot(4, 4, loc)
            plt.hist(data, 20)

plt.show()
