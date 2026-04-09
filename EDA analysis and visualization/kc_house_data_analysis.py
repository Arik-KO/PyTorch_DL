import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = '../housing_price_project/data/raw/'
df = pd.read_csv(data_dir+'kc_house_data.csv')

print(df.columns)
print()
print()
print(df.head(5))
print()
print()
print(df.tail(5))
print()
print()
print(df.sample(5))
print(df.describe())
print(df.isnull().sum())
print(df.dtypes)
df = df.drop(columns = ['date', 'id'])

corr_matrix = df.corr()
print(corr_matrix)

#plt.figure(figsize=(10,6),dpi = 200)
#sns.heatmap(corr_matrix, annot = True, fmt = ".2f", cmap = 'coolwarm')
#plt.tight_layout()
#plt.savefig('plots/heatmap_kc_housing_data.png')

y = df.price.values
X = df.drop(columns='price').values
print(type(y))
print(type(X))
print(y.shape)
print(X.shape)
np.save('../housing_price_project/data/raw/X_data.npy', X)
np.save('../housing_price_project/data/raw/y_data.npy', y)
