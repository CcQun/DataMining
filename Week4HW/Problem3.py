from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import numpy as np

dataset = pd.read_csv('wine.csv')
dataset['Class'] = dataset['Class'].map({'one': 1, 'two': 2, 'three': 3})
data = dataset.iloc[:, :-1]
label = dataset.Class

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

print('KNN score:', knn_clf.score(X_test, y_test))

y_pred = KMeans(n_clusters=3,
                init=np.array([[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1.04, 3.92, 1065],
                               [12.37, .94, 1.36, 10.6, 88, 1.98, .57, .28, .42, 1.95, 1.05, 1.82, 520],
                               [13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 740]])).fit_predict(
    data)

print('KMeans score:', metrics.calinski_harabasz_score(data, y_pred))