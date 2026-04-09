import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, layers, d_out, dropout):
        super(MLP, self).__init__()
        previous_layer = d_in
        sequence = []

        for layer in layers:
            sequence.append(nn.Linear(previous_layer, layer))
            sequence.append(nn.Dropout(dropout))
            sequence.append(nn.ReLU())
            previous_layer = layer

        sequence.append(nn.Linear(previous_layer,d_out))
        self.network = nn.Sequential(*sequence)

    def forward(self,x):
        return self.network(x)

