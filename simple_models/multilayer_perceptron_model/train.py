from config import *
import numpy as np
import torch
import torch.nn as nn
from src.dataset import HousingData
from src.model import MultiLayerPerceptronModel
from src.trainer import Trainer
from torch.utils.data import DataLoader
from utilis.helper import *


def main():
    X_train = np.load(DATA_DIR+'X_train.npy')
    X_val = np.load(DATA_DIR + 'X_val.npy')
    y_train = np.load(DATA_DIR+ 'y_train.npy')
    y_val = np.load(DATA_DIR + 'y_val.npy')

    #convet to tensor
    X_train_t = torch.tensor(X_train)
    X_val_t = torch.tensor(X_val)
    y_train_t = torch.tensor(y_train)
    y_val_t = torch.tensor(y_val)

    # utilize dataset class from pytorch
    training_dataset = HousingData(X_train_t, y_train_t)
    validation_dataset =HousingData(X_val_t, y_val_t)
    #print(len(training_dataset))
    #print(training_dataset[0])
    train_loader = DataLoader(training_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 0)
    val_loader= DataLoader(validation_dataset, shuffle = False, batch_size = BATCH_SIZE, num_workers = 0)

    MLPmodel = MultiLayerPerceptronModel(d_in = X_train_t.shape[-1], hidden_layers = HIDDEN_LAYERS, d_out =1, dropout = 0.0)
    MLPmodel = MLPmodel.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(MLPmodel.parameters(), lr = LEARNING_RATE, weight_decay = 0)
    trainer = Trainer(MLPmodel, criterion, optimizer, DEVICE)

    for epoch in range(EPOCHS):
        train_loss = trainer.train_one_epoch(train_loader)
        trainer.train_losses.append(train_loss)

        val_loss = trainer.validation_function(val_loader)
        trainer.val_losses.append(val_loss)

        print(f"for epoch {epoch}|{EPOCHS}, the training loss is: {train_loss:.4f}, and "
              f"the validation loss is: {val_loss:.4f}")

    plot_loss(trainer.train_losses, trainer.val_losses, MODEL_NAME)



if __name__ == "__main__":
    main()