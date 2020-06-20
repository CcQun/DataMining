from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target
print(x)

dt_model = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
dt_model.fit(X_train, y_train)

predictions = dt_model.predict(X_test)
print(dt_model.score(X_test, y_test))
