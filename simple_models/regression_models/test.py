from config import *
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset import ProjectData
from src.model import LinearRegression
from utilis.helper import load_model, visualize_performance
import os


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    X_test  = np.load(DATA_DIR + 'X_test.npy')
    y_test  = np.load(DATA_DIR + 'y_test.npy')


    X_test_t = torch.tensor(X_test, dtype = torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    test_dataset = ProjectData(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LinearRegression(X_test_t.shape[1])
    model = load_model(model, 'results/models/linear_regression_without_L2.pth')
    model = model.to(DEVICE)

    all_pred = []
    true_pred = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            test_pred = model(x_batch)
            all_pred.append(test_pred.cpu().numpy().flatten())
            true_pred.append(y_batch.cpu().numpy().flatten())

    y_hat = np.concatenate(all_pred)
    gnd_truth = np.concatenate(true_pred)

    mse = np.mean( (y_hat - gnd_truth) ** 2 )
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_hat - gnd_truth))

    print(f"mean squared error: {mse}")
    print(f"root mean squared error: {rmse}")
    print(f"mean absolute error: {mae}")

    visualize_performance(gnd_truth, y_hat, 'linear_reg_without_l2')

    result_row = {
        'model_name':'Linear_Regression',
        'Lambda (L2)' : 0,
        'Epochs' : EPOCHS,
        'MSE' : round(mse, 4),
        'RMSE' : round(rmse, 4),
        'MAE' : round(mae, 4),
        'Optimizer': 'Adam',
        'Batch_size': BATCH_SIZE,
        'Random_Seed': RANDOM_SEED,
        'Learning Rate': LEARNING_RATE
    }

    result_df = pd.DataFrame([result_row])
    log_path = 'results/logs/experiment_log.csv'

    if os.path.exists(log_path):
        result_df.to_csv(log_path, mode = 'a', header = False, index = False )
    else:
        result_df.to_csv(log_path, mode = 'w', header = True, index = False)

if __name__ == "__main__":
    main()





