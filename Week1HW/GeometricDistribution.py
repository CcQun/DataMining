import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

p = 0.5
x = np.arange(1, 11)
geometric = geom.pmf(x, p)

plt.plot(x, geometric, '-o')
plt.title('Geometric distribution', fontsize=15)
plt.xlabel('Number of successes')
plt.ylabel('Probability', fontsize=15)
plt.show()
