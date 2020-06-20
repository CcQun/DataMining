# 绘图——均匀分布
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.style as style

# PLOTTING CONFIG 绘图配置
style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (14, 7)
plt.figure(dpi=100)

# PDF（概率密度函数）
plt.plot(np.linspace(-4, 4, 100), stats.uniform.pdf(np.linspace(-4, 4, 100)))
plt.fill_between(np.linspace(-4, 4, 100), stats.uniform.pdf(np.linspace(-4, 4, 100)), alpha=0.15)

# CDF（概率累积函数）
# plt.plot(np.linspace(-4, 4, 100), stats.uniform.cdf(np.linspace(-4, 4, 100)))

# LEGEND 图例
plt.text(x=-1.5, y=0.7, s="pdf(uniform)", rotation=65, alpha=0.75, weight="bold", color="#008fd5")
# plt.text(x=-0.4, y=0.5, s="cdf", rotation=55, alpha=0.75, weight="bold", color="#fc4f30")
plt.show()
