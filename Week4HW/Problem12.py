import numpy as np
import pandas as pd


# p:样本点维度
# n:样本点个数
# k:聚类中心个数

def final_classify(train_data, crowds):
    p = train_data.shape[1]
    n = len(train_data)
    k = len(crowds)

    new_crowds = crowds
    clsy = np.ndarray((n,))
    new_clsy = np.ndarray((n,))
    while (clsy != new_clsy).any():
        clsy = new_clsy
        new_clsy = classify(train_data, new_crowds)
        print('new_clsy:', new_clsy)
        new_crowds = []
        clusters = []  # 每一个聚类中的样本点的索引
        for i in range(k):
            clusters.append([])
        for i in range(n):
            clusters[new_clsy[i]].append(i)
        for j in range(k):
            if len(clusters[j]) == 0:
                new_crowds.append(crowds[j])
            else:
                sums = np.zeros((p,))
                for m in clusters[j]:
                    sums += train_data[m]
                means = sums / len(clusters[j])
                new_crowds.append(means)

    return (new_crowds, new_clsy)


# 将样本点分类到最近的聚类中心，其维度为(n,)
def classify(train_data, crowds):
    all_distances = get_distances(train_data, crowds)
    clsy = np.argmin(all_distances, axis=0)
    return clsy


# 返回所有样本点到所有聚类中心的欧氏距离，其维度为(k,n)
def get_distances(train_data, crowds):
    all_distances = []  # 保存所有样本点到所有聚类中心的欧氏距离，其维度为(k,n)
    for i in range(len(crowds)):
        distances = []  # 保存所有样本点到一个聚类中心的欧氏距离，其维度为(n,)
        for j in range(len(train_data)):
            distances.append(get_euclidean_distance(train_data[j], crowds[i]))
        all_distances.append(distances)
    return all_distances


# 返回两点之间的欧氏距离，其中point1、point2为两个点的坐标，其维度为(p,)
def get_euclidean_distance(point1, point2):
    return (np.sum((point1 - point2) ** 2)) ** 0.5


# 返回一个bool值，表示分类结果是否改变
def clsy_change(new_clsy, clsy):
    changed = False
    for i in range(len(clsy)):
        if clsy[i] != new_clsy[i]:
            changed = True
            break
    return changed


print('===========Problem1===========')
crowds1 = np.array([[1, 1], [2, 1]])
dataCsv1 = 'p1.csv'
data1 = pd.read_csv(dataCsv1)

train_data1 = data1.iloc[:, 1:].values

result1 = final_classify(train_data1, crowds1)

print('聚类中心:', np.array(result1[0]))
print('聚类结果:', np.array(result1[1]))
print()

print('===========Problem2===========')

# 初始聚类中心
crowds2 = np.array([[12, 15, 13, 28, 24], [7, 11, 10, 19, 21]])
dataCsv2 = 'p2.csv'
data2 = pd.read_csv(dataCsv2)

train_data2 = data2.iloc[:, 1:].values
result2 = final_classify(train_data2, crowds2)

print('聚类中心:', np.array(result2[0]))
print('聚类结果:', np.array(result2[1]))

