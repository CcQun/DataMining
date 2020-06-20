from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn import linear_model

data = read_csv('test1.csv')

regr = linear_model.LinearRegression()
regr.fit(data.活动推广费.values.reshape(-1, 1), data.销售额)

plt.scatter(data.活动推广费, data.销售额, color='blue')
plt.plot(data.活动推广费, regr.predict(data.活动推广费.values.reshape(-1, 1)), color='red', linewidth=4)
plt.show()
