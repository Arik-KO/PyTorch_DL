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

    test_dataset = HousingData(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, shuffle = False, num_workers = 0, batch_size = BATCH_SIZE)

    model = MultiLayerPerceptronModel(X_test_t.shape[-1], HIDDEN_LAYERS, 1, DROPOUT)
    model = load_model(model, f'results/models/{MODEL_NAME}.pth')
    model = model.to(DEVICE)

    all_pred = []
    true_pred = []

    with torch.no_grad():  #computation graph not needed
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            test_pred = model(x_batch)
            all_pred.append(test_pred.cpu().numpy().flatten())
            true_pred.append(y_batch.cpu().numpy().flatten())

        y_hat = np.concatenate(all_pred)
        gnd_truth = np.concatenate(true_pred)

        mse = np.mean((y_hat - gnd_truth) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_hat - gnd_truth))
        r2_score = 1 - (np.sum((y_hat - gnd_truth) ** 2) / np.sum((np.mean(gnd_truth) - gnd_truth) ** 2))

        print(f"mean squared error: {mse}")
        print(f"root mean squared error: {rmse}")
        print(f"mean absolute error: {mae}")
        print(f"R2 Score: {r2_score}")

        visualize_performance(gnd_truth, y_hat, MODEL_NAME)

if __name__ == "__main__":
    main()