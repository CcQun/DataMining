import numpy as np
import matplotlib.pyplot as plt

# Poisson分布
x = np.random.poisson(lam=5, size=10000)  # lam为λ size为k
pillar = 15
a = plt.hist(x, bins=pillar, density=True, range=[0, pillar], color='g', alpha=0.5)
plt.plot(a[1][0:pillar], a[0], '-o')
plt.grid()
plt.show()