import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 通过read_csv读数据集
adv_data = pd.read_csv("test4.csv")
# 清洗不需要的数据
new_adv_data = adv_data.iloc[:, 1:]
# 得到数据集且查看
print('head:', new_adv_data.head(), '\nShape:', new_adv_data.shape)

print(new_adv_data.describe())
# 缺失值检验
print(new_adv_data[new_adv_data.isnull() == True].count())
new_adv_data.boxplot()
plt.show()

print(new_adv_data.corr())
X_train, X_test, Y_train, Y_test = train_test_split(new_adv_data.iloc[:, :3], new_adv_data.sales, train_size=.80)
print("原始数据特征:", new_adv_data.iloc[:, :3].shape, ",训练数据特征:", X_train.shape, ",测试数据特征:", X_test.shape)
print("原始数据标签:", new_adv_data.sales.shape, ",训练数据标签:", Y_train.shape, ",测试数据标签:", Y_test.shape)
print(X_train.shape)

model = LinearRegression()
model.fit(X_train, Y_train)
a = model.intercept_
b = model.coef_
print("最佳拟合线:截距", a, ",回归系数：", b)

score = model.score(X_test, Y_test)
print(score)

Y_pred = model.predict(X_test)
print(Y_pred)
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.show()

Y_pred = model.predict(X_test)
X_train, X_test, Y_train, Y_test = train_test_split(new_adv_data.iloc[:, :3], new_adv_data.sales, train_size=.80)

plt.figure()
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.plot(range(len(X_test)), Y_test, 'r', label="test")
plt.legend(loc="upper right")
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()
