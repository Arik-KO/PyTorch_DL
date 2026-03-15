import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

LEARNING_RATE = 1e-4
LAMBDA = 1e-3
BATCH_SIZE = 64
EPOCHS = 20