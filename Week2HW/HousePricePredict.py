# import pandas as pd  # 读取csv文件
# from sklearn import linear_model  # 线性模型
#
# data = pd.read_csv('PriceData.csv')
# regr = linear_model.LinearRegression()  # 线性回归模型
# regr.fit(data.square_feet.values.reshape(-1, 1), data.price)
# print(regr.predict([[1000]]))  # 预测面积为1000时的房价

import pandas as pd  # 读取csv文件
from sklearn import linear_model  # 线性模型

data = pd.read_csv('PriceData.csv')
trainData = data.iloc[:, 1:4]  # 取读取数据的2、3、4列作为训练数据，每条训练数据都有三个特征
trainLabel = data.price
regr = linear_model.LinearRegression()
regr.fit(trainData, trainLabel)
print(regr.predict([[1200, 720, 700]]))  # 预测特征为[[1200, 720, 700]]时的房价
