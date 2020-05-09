import pandas as pd
train = pd.read_csv('train.csv')
k = train.comp.unique()
print(k)