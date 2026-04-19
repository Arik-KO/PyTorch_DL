import torch.nn as nn
import torch

class LogisticRegression(nn.Module):

    def __init__(self, input_dim, output_dim =1):
        super(LogisticRegression, self).__init__()
        self.logistic = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.logistic(x))


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim=1, dropout=0.0):
        super().__init__()
        previous_layer = input_dim
        network_architecture = []

        for layer in hidden_layers:
            network_architecture.append(nn.Linear(previous_layer, layer))
            network_architecture.append(nn.ReLU())
            network_architecture.append(nn.Dropout(dropout))
            previous_layer = layer

        network_architecture.append(nn.Linear(previous_layer, output_dim))
        self.network = nn.Sequential(*network_architecture)

    def forward(self, x):
        return torch.sigmoid(self.network(x))

