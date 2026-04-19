import torch




RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_LAYERS = [54,32,16]
EPOCHS = 20
LAMBDA = 0.0
DROPOUT = 0.0

