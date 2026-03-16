from config import *
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import ProjectData
from src.trainer import Trainer
from src.model import LinearRegression
from utilis.helper import save_model, plot_losses

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_train = np.load(DATA_DIR + 'X_train.npy')
    X_val   = np.load(DATA_DIR + 'X_val.npy')
    X_test  = np.load(DATA_DIR + 'X_test.npy')
    y_train = np.load(DATA_DIR + 'y_train.npy')
    y_val   = np.load(DATA_DIR + 'y_val.npy')
    y_test  = np.load(DATA_DIR + 'y_test.npy')

    X_train_t = torch.tensor(X_train, dtype = torch.float32)
    X_val_t = torch.tensor(X_val, dtype = torch.float32)
    #X_test_t = torch.tensor(X_test, dtype = torch.float32)

    y_train_t = torch.tensor(y_train, dtype = torch.float32).unsqueeze(1)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)
    #y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    training_dataset = ProjectData(X_train_t, y_train_t)
    validation_dataset = ProjectData(X_val_t, y_val_t)
    train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    model = LinearRegression(X_train_t.shape[1])
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = 0)
    trainer = Trainer(model, criterion, optimizer, DEVICE)

    for epoch in range(EPOCHS):
        train_loss = trainer.train_one_epoch(train_loader)
        trainer.train_losses.append(train_loss)

        val_loss = trainer.validation_function(val_loader)
        trainer.val_losses.append(val_loss)

        print(f"for epoch {epoch}, the training loss is: {train_loss:.4f}, and the validation loss is: {val_loss:.4f}")

    plot_losses(trainer.train_losses, trainer.val_losses, "Linear_reg_without_L2")
    save_model(model, 'results/models/linear_regression_without_L2.pth')


if __name__ == "__main__":
    main()
