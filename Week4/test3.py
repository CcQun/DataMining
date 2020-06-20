import numpy as np

data = np.array([[1, 1], [2, 1], [4, 3], [5, 4]])
C1 = np.array([1.5, 1])
C2 = np.array([4.5, 3.5])
dc1 = []
for i in range(len(data)):
    dis = np.sqrt(sum((data[i] - C1) ** 2))
    dc1.append(dis)

print(dc1)

dc2 = []
for i in range(len(data)):
    dis = np.sqrt(sum((data[i] - C2) ** 2))
    dc2.append(dis)

print(dc2)
