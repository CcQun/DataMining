# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

lambd = 0.5
x = np.arange(0, 15, 0.1)
y = lambd * np.exp(-lambd * x)
plt.plot(x, y)
plt.show()