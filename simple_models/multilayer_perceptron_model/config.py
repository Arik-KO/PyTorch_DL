import torch

DATA_DIR = '/data/processed/'
LEARNING_RATE = 1e-3
HIDDEN_LAYERS = [64, 32, 16, 1]
DROPOUT = 0.2
EPOCHS = 50
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

