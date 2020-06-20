import pandas as pd

data = pd.read_csv('data.csv')
print(data.dropna(axis=1))
print(data.dropna(axis=0))
print(data.fillna(method='pad'))
print(data.fillna(data.mean()))
print(data.fillna(0))