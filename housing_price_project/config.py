import torch

HIDDEN_LAYERS = [64,32,16]
RANDOM_STATE = 42
LAMBDA = 0.0
DROPOUT = 0.0
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'Price_Prediction_Neural_Network_v1'

if __name__ == "__main__":
    print(DEVICE)