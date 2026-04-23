import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



def main():
    x_train_scaled = np.load('data/processed/X_train_scaled_1.npy')
    # print(len(x_train_scaled))
    x_val_scaled = np.load('data/processed/X_val_scaled_1.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')










if __name__ == "__main__":
    main()