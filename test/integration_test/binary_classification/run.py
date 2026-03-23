import torch
import torch.nn as nn
import torch.optim as optim
from src.trainer.trainer import Trainer

from test.integration_test.binary_classification.dataset import ToyClassificationDataset, create_dataloader
from test.integration_test.binary_classification.model import SimpleClassifier

# dataset
train_dataset = ToyClassificationDataset(n_samples=512, n_features=10)
val_dataset = ToyClassificationDataset(n_samples=128, n_features=10)

# dataloader
train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

# model
model = SimpleClassifier(input_dim=10)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    config_path="test/integration_test/binary_classification/config.yaml" 
)

trainer.run()

# 5. Sanity Check
model.eval()
x, y = next(iter(val_loader))
with torch.no_grad():
    logits = model(x)
    preds = (torch.sigmoid(logits) > 0.5).float()

assert preds.shape == y.shape
print("Sanity check passed: Prediction shape matches label shape.")