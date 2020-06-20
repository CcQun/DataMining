import pandas as pd

data = pd.read_csv('cs-training.csv')
print(data.dropna())
print(data.dropna(axis=1))
print(data.fillna(0))
print(data.fillna(data.mean()))
print(data.fillna(method='pad'))