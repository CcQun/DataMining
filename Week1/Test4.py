import matplotlib.pyplot as plt
import numpy as np


def getFact(num):
    factorial = 1
    for i in range(1, num + 1):
        factorial = factorial * i
    return factorial


x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = np.zeros(9)
first = []
for i in range(2000):
    y[int(str(getFact(i + 1))[0]) - 1] += 1

plt.plot(x, y, "r-", linewidth=2)
plt.show()
