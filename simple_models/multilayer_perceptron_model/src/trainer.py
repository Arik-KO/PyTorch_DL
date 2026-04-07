import torch
from torch.utils.data import DataLoader
import numpy as np


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_one_epoch(self, dataloader:DataLoader) -> float:
        total_loss = 0
        self.model.train()

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(x_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss/len(dataloader)

    def validation_function(self, dataloader:DataLoader) -> list:
        total_val_loss = 0
        self.model.eval()
        all_pred = []
        true_pred = []

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                val_predictions = self.model(x_batch)
                val_loss = self.criterion(val_predictions, y_batch)
                all_pred.append(val_predictions.cpu().numpy().flatten())
                true_pred.append(y_batch.cpu().numpy().flatten())
                total_val_loss += val_loss.item()

            y_hat = np.concatenate(all_pred)
            gnd_truth = np.concatenate(true_pred)
            mse = np.mean((y_hat - gnd_truth)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_hat - gnd_truth))
            r2_error = 1 - (np.sum((y_hat - gnd_truth) ** 2) / np.sum( (np.mean(gnd_truth) - gnd_truth) ** 2) )


        return [total_val_loss/len(dataloader), mse, rmse, mae, r2_error]

