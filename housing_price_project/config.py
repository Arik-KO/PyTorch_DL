import torch

HIDDEN_LAYERS = [64,32,16]
RANDOM_STATE = 42
LAMBDA = 0.0
DROPOUT = 0.1
BATCH_SIZE = 64
EMBED_DIM = 32
HEAD = 4
D_K = EMBED_DIM // HEAD
FF_DIM = D_K * HEAD
NUM_LAYERS = 2
LEARNING_RATE = 1e-4
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'Price_Prediction_Neural_Network_v1'

if __name__ == "__main__":
    print(DEVICE)