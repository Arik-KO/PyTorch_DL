from config import *
import numpy as np
import os
import pandas as pd
import torch.nn as nn
from src.dataset import HousingData
from src.model import MultiLayerPerceptronModel
from src.trainer import Trainer
from torch.utils.data import DataLoader
from utilis.helper import *
import torch

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    x_test = np.load(DATA_DIR + 'X_test.npy')
    y_test = np.load(DATA_DIR + 'y_test.npy')

    # convert to tensor.float32

    X_test_t = torch.tensor(x_test, dtype = torch.float32)
    y_test_t = torch.tensor(y_test, dtype = torch.float32).unsqueeze(1)

if __name__ == "__main__":
    main()