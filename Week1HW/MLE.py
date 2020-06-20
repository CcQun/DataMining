from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

'''
norm.cdf 返回对应的累计分布函数值
norm.pdf 返回对应的概率密度函数值
norm.rvs 产生指定参数的随机变量
norm.fit 返回给定数据下，各参数的最大似然估计（MLE）值
'''
x_norm = norm.rvs(size=200)
# 在这组数据下，正态分布参数的最大似然估计值
x_mean, x_std = norm.fit(x_norm)
print('mean, ', x_mean)
print('x_std, ', x_std)
plt.hist(x_norm, density=True, bins=15)  # 归一化直方图（用出现频率代替次数），将划分区间变为 20（默认 10）
x = np.linspace(-3, 3, 50)  # 在在(-3,3)之间返回均匀间隔的50个数字。
plt.plot(x, norm.pdf(x), 'r-')
plt.show()
