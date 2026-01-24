from trainer.trainer import build_trainer
from test.dataset import ToyRegressionDataset, create_dataloader
from test.model import SimpleDenseModel
import torch
import torch.nn as nn
import torch.optim as optim

# dataset
train_dataset = ToyRegressionDataset(n_samples=512, n_features=10)
val_dataset = ToyRegressionDataset(n_samples=128, n_features=10)

# data loader
train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

# model
model = SimpleDenseModel(input_dim=10)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

model = build_trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
)

# sanity check
model.eval()
x, y = next(iter(val_loader))
with torch.no_grad():
    pred = model(x)
assert pred.shape == y.shape