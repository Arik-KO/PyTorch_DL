import torch.nn as nn


class MultiLayerPerceptronModel(nn.Module):
    def __init__(self, d_in:int, hidden_layers:list, d_out:int, dropout:float):
        super(MultiLayerPerceptronModel, self).__init__()
        layers = []
        previous_dim = d_in

        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_layer

        layers.append(nn.Linear(previous_dim, d_out))
        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)
