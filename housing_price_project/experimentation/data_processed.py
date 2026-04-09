import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import RANDOM_STATE


X_data = np.load('../data/raw/X_data.npy')
y_data = np.load('../data/raw/y_data.npy')

# print(X_data.shape)
# print(y_data.shape)
# print(12958+4319+4320 == X_data.shape[0])

X_train, X_dev, y_train, y_dev = train_test_split(X_data, y_data, shuffle =True,
                                                  random_state=RANDOM_STATE, test_size = 0.4)
X_val, X_test, y_val, y_test = train_test_split(X_dev, y_dev, random_state = RANDOM_STATE, shuffle = False,
                                                test_size = 0.5)

# print(X_train.shape)
# print(X_test.shape)
# print(X_val.shape)

np.save('../data/processed/X_train.npy', X_train)
np.save('../data/processed/X_test.npy', X_test)
np.save('../data/processed/X_val.npy', X_val)
np.save('../data/processed/y_train.npy', y_train)
np.save('../data/processed/y_test.npy', y_test)
np.save('../data/processed/y_val.npy', y_val)