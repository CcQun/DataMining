import numpy as np
import matplotlib.pyplot as plt

x = np.random.poisson(lam=5, size=10000)
pillar = 15
a = plt.hist(x, pillar, color='g')
plt.plot_date(a[1][0:pillar], a[0], 'r')
plt.grid()
plt.show()
