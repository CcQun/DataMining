import pandas as pd  # 读取csv文件
from sklearn import linear_model  # 线性模型
from sklearn.model_selection import train_test_split

data = pd.read_csv('abalone.csv')
trainData = data.iloc[:, :-1]
trainLabel = data.rings
sexMapping = {
    'F': 0.1,
    'M': 0.5,
    'I': 0.9
}
trainData['sex'] = trainData['sex'].map(sexMapping)
X_train, X_test, Y_train, Y_test = train_test_split(trainData, trainLabel, train_size=.80)
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
pre = regr.predict(X_test).astype('int')
ytest = Y_test.values

pre_train = regr.predict(X_train).astype('int')
ytrain = Y_train.values

loss_train_sum = 0
for i in range(len(pre_train)):
    loss_train_sum += pow(pre_train[i] - ytrain[i], 2)
loss_train_avg = loss_train_sum / len(pre_train)

loss_test_sum = 0
for i in range(len(pre)):
    loss_test_sum += pow(pre[i] - ytest[i], 2)
loss_test_avg = loss_test_sum / len(pre)

print('训练均方误差:', loss_train_avg)
print('测试均方误差:', loss_test_avg)
