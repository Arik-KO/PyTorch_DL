import torch

DATA_DIR = 'data/processed/'
RANDOM_SEED = 42
LEARNING_RATE = 1e-3
HIDDEN_LAYERS = [64, 16]
DROPOUT = 0.0
EPOCHS = 100
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'MLP_v5'
