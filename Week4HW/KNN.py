from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

data = pd.read_csv('1.csv')

SpeciesMap = {
    'setosa': 1,
    'versicolor': 2,
    'virginica': 3
}
data['Species'] = data['Species'].map(SpeciesMap)

X = data.iloc[:, :4].values
y = data.Species.values

print('X.shape:', X.shape)
print('y.shape:', y.shape)

# 模型训练和测试
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X, y)

pre = knn_clf.predict([[2.3, 1.1, 2.8, 1.4]])  # 预测

print('predict:', pre)