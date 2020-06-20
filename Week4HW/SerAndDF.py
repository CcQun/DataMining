# Series (Series)是能够保存任何类型的数据(整数，字符串，浮点数，Python对象等)的一维标记数组。轴标签统称为索引。
# DataFrame是一种表格型数据结构，它含有一组有序的列，每列可以是不同的值。DataFrame既有行索引，也有列索引，它可以看作是由Series组成的字典，不过这些Series公用一个索引。

import numpy as np
import pandas as pd

# array --> Series
arr = np.array([4, 2, 3, 1])
ser = pd.Series(arr, index=['a', 'b', 'c', 'd'])
print(ser)

# Series --> DataFrame
df1 = pd.DataFrame(ser)
print(df1)

data = {'data1': ser.values, 'data2': ['aa', 'bb', 'cc', 'dd']}
df2 = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
print(df2)

# DataFrame --> array
arr2 = np.array(df2.values)
print(arr2)

# DataFrame排序
print(df2.sort_values(by='data1', ascending=False))

# 数据聚合指使用基于多组观测结果的总结的统计替换多组观测结果
