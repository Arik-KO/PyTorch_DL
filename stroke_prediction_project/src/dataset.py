from torch.utils.data import Dataset
import torch

class StrokeDataset(Dataset):
    def __init__(self, x_data:torch.Tensor , y_data:torch.Tensor):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
