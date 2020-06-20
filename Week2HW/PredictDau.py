import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn import linear_model

x = np.array([154, 157, 158, 159, 160, 161, 162, 163])
y = np.array([155, 156, 159, 162, 161, 164, 165, 166])

regr = linear_model.LinearRegression()
regr.fit(x.reshape(-1, 1), y)
print(regr.predict([[167]]))
