import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ToyRegressionDataset(Dataset):
    """
    y = Wx + b + noise
    """
    def __init__(self, n_samples=512, n_features=10, noise_std=0.1):
        super().__init__()
        self.X = torch.randn(n_samples, n_features)
        true_w = torch.randn(n_features, 1)
        true_b = torch.randn(1)
        noise = noise_std * torch.randn(n_samples, 1)
        self.y = self.X @ true_w + true_b + noise

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def create_dataloader(dataset, batch_size=32, shuffle=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
