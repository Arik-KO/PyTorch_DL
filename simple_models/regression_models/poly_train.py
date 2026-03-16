from config import *
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import ProjectData
from src.trainer import Trainer
from src.model import LinearRegression
from utilis.helper import save_model, plot_losses
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def get_poly_data():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load raw data
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    # Polynomial expansion
    poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
    X_train_poly_ = poly.fit_transform(X_train)
    X_val_poly_   = poly.transform(X_val)
    X_test_poly_  = poly.transform(X_test)

    # Standardization
    poly_scaler = StandardScaler()
    X_train_poly = poly_scaler.fit_transform(X_train_poly_)
    X_val_poly   = poly_scaler.transform(X_val_poly_)
    X_test_poly  = poly_scaler.transform(X_test_poly_)

    # To tensors
    X_train_t = torch.tensor(X_train_poly, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val_poly,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_poly,  dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    # Datasets and loaders
    training_dataset   = ProjectData(X_train_t, y_train_t)
    validation_dataset = ProjectData(X_val_t, y_val_t)
    test_dataset       = ProjectData(X_test_t, y_test_t)

    train_loader = DataLoader(training_dataset,   batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Return everything you might need
    return train_loader, val_loader, test_loader, X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t


def main():
    # Get loaders and tensors
    (train_loader,
     val_loader,
     test_loader,
     X_train_t,
     X_val_t,
     X_test_t,
     y_train_t,
     y_val_t,
     y_test_t) = get_poly_data()

    # Model, loss, optimizer
    model = LinearRegression(X_train_t.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA)

    trainer = Trainer(model, criterion, optimizer, DEVICE)

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = trainer.train_one_epoch(train_loader)
        trainer.train_losses.append(train_loss)

        val_loss = trainer.validation_function(val_loader)
        trainer.val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    # Plot and save
    plot_losses(trainer.train_losses, trainer.val_losses, f"poly_reg_{DEGREE}")
    save_model(model, f"results/models/poly_reg_{DEGREE}.pth")


if __name__ == "__main__":
    main()
