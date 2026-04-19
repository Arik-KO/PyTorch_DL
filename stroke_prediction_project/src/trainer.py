import torch

def trainer():
    def __init__(self, model, device, criterion, optimizer):

        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_one_epoch(self, train_loader):
        train_loss = 0
        self.model.train()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.no_grad()
            predictions = self.model(x_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss/len(train_loader)

    def validation_function(self, val_loader)
        total_val_loss = 0
        self.model.eval()

        with torch.zero_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(x_batch)
                val_loss = self.criterion(predictions, y_batch)
                total_val_loss += val_loss.item()

        return total_val_loss / len(val_loader)