import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn import linear_model

np.bunc
data = read_csv('test1.csv')

plt.scatter(data.活动推广费, data.销售额)
plt.show()
u = data.corr()
print(u)

regr = linear_model.LinearRegression()
regr.fit(data['活动推广费'].values.reshape(-1, 1), data['销售额'])
regr.predict([[60]])
a, b = regr.coef_, regr.intercept_
area = 75
print(a * area + b)

