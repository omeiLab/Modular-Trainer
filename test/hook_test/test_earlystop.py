import pytest

from src.hooks.after_epoch.early_stop import AfterEpochEarlyStopHook
from src.control.controller import Controller

results = [
    {"val_loss": 0.5, "f1": 0.2},
    {"val_loss": 0.4, "f1": 0.3},
    {"val_loss": 0.6, "f1": 0.25},
    {"val_loss": 0.3, "f1": 0.3},
    {"val_loss": 0.2, "f1": 0.4},
    {"val_loss": 0.1, "f1": 0.5},
    {"val_loss": 0.18, "f1": 0.4},
    {"val_loss": 0.23, "f1": 0.4},
    {"val_loss": 0.1, "f1": 0.5},
    {"val_loss": 0.07, "f1": 0.6},
]

@pytest.fixture
def build_controller():
    return Controller()

def test_metric_not_exist(build_controller):
    controller = build_controller
    checkpoint = AfterEpochEarlyStopHook(controller, patience=1, metric="mse")
    with pytest.raises(ValueError):
        checkpoint.execute(epoch=1, results=results[0])

def test_maximize(build_controller):
    controller = build_controller
    checkpoint = AfterEpochEarlyStopHook(controller, patience=2, metric="f1", maximize=True)
    for epoch, res in enumerate(results):
        checkpoint.execute(epoch=epoch, results=res)
        if not controller.should_continue():
            break
    assert epoch == 3

def test_minimize(build_controller):
    controller = build_controller
    checkpoint = AfterEpochEarlyStopHook(controller, patience=2, metric="val_loss", maximize=False)
    for epoch, res in enumerate(results):
        checkpoint.execute(epoch=epoch, results=res)
        if not controller.should_continue():
            break
    assert epoch == 7

def test_min_delta(build_controller):
    controller = build_controller
    checkpoint = AfterEpochEarlyStopHook(controller, patience=3, metric="f1", maximize=True, min_delta=0.05)
    for epoch, res in enumerate(results):
        checkpoint.execute(epoch=epoch, results=res)
        if not controller.should_continue():
            break
    assert epoch == 8