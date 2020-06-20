import numpy as np

X = np.array([2, 2, 3])
Y = np.array([1, 1, 2])

# 曼哈顿距离:sumi(|xi-yi|)
ManhattanDistance = np.sum(np.abs(X - Y))
print('ManhattanDistance:', ManhattanDistance)

print(sum((X-Y) ** 2) ** 0.5)

# 切比雪夫距离:maxi(|xi-yi|)
ChebyshevDistance = np.max(np.abs(X - Y))
print('ChebyshevDistance:', ChebyshevDistance)

# 闵可夫斯基距离:sumi((xi-yi)^p) ** (1/p)
# 注意闵可夫斯基距离的一些特例:
#   p = 1:曼哈顿距离
#   p = 2:欧式距离
#   p = ∞:切比雪夫距离
p = 5
MinkowskiDistance = np.sum((X - Y) ** p) ** (1 / p)
print('MinkowskiDistance:', MinkowskiDistance)

# 标准化欧氏距离:sumi(((xi-yi) / si)^2) ** 0.5       si为第分量的标准差
mtx = np.vstack([X, Y])

sk = np.var(mtx, axis=0, ddof=1)
SED1 = np.sqrt(((X - Y) ** 2 / sk).sum())
print('Standardized Euclidean distance:', SED1, '(方法一:根据公式求解)')

from scipy.spatial.distance import pdist

SED2 = pdist(mtx, 'seuclidean')
print('Standardized Euclidean distance:', SED2[0], '(方法二:根据scipy库求解)')

# 马氏距离 D(Xi,Xj)=sqrt(dot(dot((Xi-Xj).T,SI),(Xi-Xj)))
# 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
print()
X = np.array([[3, 5, 2, 8],
              [4, 6, 2, 4]])  # 总计10个样本，每个样本2维
XT = X.T

S = np.cov(X)  # 两个维度之间协方差矩阵
SI = np.linalg.inv(S)  # 协方差矩阵的逆矩阵

n = XT.shape[0]
MahalanobisDistance1 = []
for i in range(0, n):
    for j in range(i + 1, n):
        delta = XT[i] - XT[j]
        d = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
        MahalanobisDistance1.append(d)

print('MahalanobisDistance:', MahalanobisDistance1, '(方法一:根据公式求解)')

from scipy.spatial.distance import pdist

MahalanobisDistance2 = pdist(XT, 'mahalanobis')
print('MahalanobisDistance:', MahalanobisDistance2, '(方法二:根据scipy库求解)')
