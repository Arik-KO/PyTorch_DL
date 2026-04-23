import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from config import *


def main():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    # loading processed numpy dataset from the data/processed folder
    X_train = np.load('data/processed/X_train.npy')
    # print(X_train.shape)
    X_val = np.load('data/processed/X_val.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')

    # data normalization using minmax scalar
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # converting loaded numpy data to torch.Tensor
    X_train_t = torch.tensor(X_train_scaled, dtype = torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype = torch.float32)
    y_train_t = torch.tensor(y_train, dtype = torch.float32)
    y_val_t = torch.tensor(y_val, dtype = torch.float32)






if __name__ == "__main__":
    main()