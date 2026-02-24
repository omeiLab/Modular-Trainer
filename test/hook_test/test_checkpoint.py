import pytest
import torch.nn as nn

from src.hooks.after_epoch.checkpoint import AfterEpochCheckpointHook

results = [
    {"val_loss": 0.5, "f1": 0.2},
    {"val_loss": 0.4, "f1": 0.3},
    {"val_loss": 0.6, "f1": 0.25},
    {"val_loss": 0.3, "f1": 0.35},
    {"val_loss": 0.2, "f1": 0.4},
    {"val_loss": 0.1, "f1": 0.5},
    {"val_loss": 0.18, "f1": 0.4},
    {"val_loss": 0.23, "f1": 0.4},
    {"val_loss": 0.1, "f1": 0.6},
    {"val_loss": 0.07, "f1": 0.6},
]

# dummy model for testing
model = nn.Linear(10, 1)

def test_metric_not_exist():
    checkpoint = AfterEpochCheckpointHook(model, metric="mse")
    with pytest.raises(ValueError):
        checkpoint.execute(epoch=1, results=results[0])

def test_maximize():
    checkpoint = AfterEpochCheckpointHook(model, metric="f1", maximize=True)
    ckpt_epoch = []

    # override save_fn
    def save_fn(epoch):
        nonlocal ckpt_epoch
        ckpt_epoch.append(epoch)
    checkpoint.save_fn = save_fn

    for epoch, res in enumerate(results):
        checkpoint.execute(epoch=epoch, results=res)
    assert ckpt_epoch == [0, 1, 3, 4, 5, 8]

def test_minimize():
    checkpoint = AfterEpochCheckpointHook(model, metric="val_loss", maximize=False)
    ckpt_epoch = []

    # override save_fn
    def save_fn(epoch):
        nonlocal ckpt_epoch
        ckpt_epoch.append(epoch)
    checkpoint.save_fn = save_fn

    for epoch, res in enumerate(results):
        checkpoint.execute(epoch=epoch, results=res)
    assert ckpt_epoch == [0, 1, 3, 4, 5, 9]

def test_min_delta():
    checkpoint = AfterEpochCheckpointHook(model, metric="f1", maximize=True, min_delta=0.05)
    ckpt_epoch = []

    # override save_fn
    def save_fn(epoch):
        nonlocal ckpt_epoch
        ckpt_epoch.append(epoch)
    checkpoint.save_fn = save_fn

    for epoch, res in enumerate(results):
        checkpoint.execute(epoch=epoch, results=res)
    assert ckpt_epoch == [0, 1, 4, 5, 8]