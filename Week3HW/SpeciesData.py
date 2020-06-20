import pandas as pd
from sklearn import linear_model  # 线性模型

data = pd.read_csv('1.csv')

# print(data.isnull().sum())
# print(data.dropna(how='all'))
# print(data.dropna(axis=1))
# print(data.fillna(0))
# print(data.fillna(data.mean()))
# print(data.fillna(data.median()))
# print(data.fillna(data.mode()))
# print(data.fillna(method='pad'))

newData = data.fillna(data.mean())
trainData = newData.iloc[:, 0:2]  # 取读取数据的2、3、4列作为训练数据，每条训练数据都有三个特征
trainLabel = newData["Petal Length"]
regr = linear_model.LinearRegression()
regr.fit(trainData, trainLabel)
print(regr.predict([[4.9, 3]]))

# import numpy as np
# import pandas as pd
# from pandas import DataFrame
#
# data = [["a", 2, 301], ["b", 1, 201], ["c", 2, 201], ["d", 1, 301], ["e", 2, 301]]
# df = pd.DataFrame(data, columns=["A", "B", "C"])
# df.sort_values(by=["C", "B"], ascending=[False, True], inplace=True)
# print(df)
