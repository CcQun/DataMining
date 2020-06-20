def testTenured(people):
    if people["rank"] == "Professor" or people["years"] > 6:
        people["tenured"] = True


initTenured = False
data = [{"name": "Tom", "rank": "Assistant Prof", "years": 2, "tenured": initTenured},
        {"name": "Merlisa", "rank": "Assistant Prof", "years": 7, "tenured": initTenured},
        {"name": "George", "rank": "Professor", "years": 5, "tenured": initTenured},
        {"name": "Joseph", "rank": "Assistant Prof", "years": 7, "tenured": initTenured},
        {"name": "Jeff", "rank": "Professor", "years": 4, "tenured": initTenured}]
for people in data:
    testTenured(people)
    print(people)

print('=================================================================')

from numpy import mean, median
from scipy.stats import mode

a = [1, 2, 3, 4, 5, 5, 6]
print(mean(a))
print(median(a))
print(mode(a))

import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
c = np.arange(0, 1, 0.1)  # 不包括1
d = np.linspace(0, 1, 10)  # 包括1
print(a)
print(b)
print(c)
print(d)
print([1, 2] * 2)


