import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utilis.helper import corr_visualization

stroke_data = pd.read_csv('../data/raw/stroke_data.csv')

# print(stroke_data.head())
# print()
# print()
# print(stroke_data.info())
# print()
# print()
# print(stroke_data.describe())

# print(stroke_data.hypertension[0:30])
# print(stroke_data.heart_disease.unique())
# print(stroke_data.columns)
# print(stroke_data.gender.unique())
# print(stroke_data.ever_married.unique())
# print(stroke_data.work_type.unique())
# print(stroke_data.Residence_type.unique())
# print(stroke_data.smoking_status.unique())
df = pd.get_dummies(stroke_data, columns = ['gender', 'ever_married', 'work_type',
                                             'Residence_type', 'smoking_status'], drop_first = False, dtype = int)


# print(df.isnull().sum())
y = df['stroke'].values
X = df.drop(columns = ['stroke']).values


# print(df.dtypes)
# print(X.shape)
# print(y.shape)

correlation_matrix = df.corr()[['stroke']].drop('stroke')
corr_visualization(correlation_matrix, True, 'strok_data_correlation_heatmap', True)

