import os
import torch

DATA_DIR = 'data/processed/'


LEARNING_RATE = 1e-4
LAMBDA = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    print(DEVICE)