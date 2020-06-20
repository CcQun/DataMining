from scipy.stats import binom  # 导入伯努利分布
import matplotlib.pyplot as plt
import numpy as np

# 这里的伯努利分布指的是n重伯努利实验，即二项分布
n = 10
p = 0.3
x = np.arange(0, n + 1)
binomial = binom.pmf(x, n, p)

plt.plot(x, binomial, 'o-')
plt.title('Binomial: n = %i, p=%0.2f' % (n, p), fontsize=15)
plt.xlabel('Number of successes')
plt.ylabel('Probability', fontsize=15)
plt.show()
