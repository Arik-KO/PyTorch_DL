import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED
import joblib

X = np.load('../data/processed/x_data.npy')
y =  np.load('../data/processed/y_data.npy')

# print(X.shape)
# print(y.shape)

scaler_1 = StandardScaler()
scaler_2 = MinMaxScaler()

X_train, X_custom, y_train, y_custom = train_test_split(X, y, test_size = 0.4,
                                                        stratify=y,shuffle = True,random_state= RANDOM_SEED)

X_val, X_test, y_val, y_test = train_test_split(X_custom, y_custom, stratify=y_custom,
                                                random_state = RANDOM_SEED, test_size = 0.5)

X_train_scaled_1 = scaler_1.fit_transform(X_train)
X_test_scaled_1 = scaler_1.transform(X_test)
X_val_scaled_1 = scaler_1.transform(X_val)

X_train_scaled_2 = scaler_2.fit_transform(X_train)
X_test_scaled_2 = scaler_2.transform(X_test)
X_val_scaled_2 = scaler_2.transform(X_val)

np.save('../data/processed/X_train_scaled_1.npy',X_train_scaled_1)
np.save('../data/processed/X_train_scaled_2.npy',X_train_scaled_2)
np.save('../data/processed/X_val_scaled_1.npy',X_val_scaled_1)
np.save('../data/processed/X_val_scaled_2.npy',X_val_scaled_2)
np.save('../data/processed/X_test_scaled_1.npy',X_test_scaled_1)
np.save('../data/processed/X_test_scaled_2.npy',X_test_scaled_2)
np.save('../data/processed/y_train.npy', y_train)
np.save('../data/processed/y_test.npy', y_test)
np.save('../data/processed/y_val.npy', y_val)

joblib.dump(scaler_1, '../data/processed/scaler_1.pkl')
joblib.dump(scaler_2, '../data/processed/scaler_2.pkl')

# scaler_1 = joblib.load('../data/processed/scaler_1.pkl') -- To load later.