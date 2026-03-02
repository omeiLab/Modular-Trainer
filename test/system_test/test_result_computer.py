import pytest
import torch

from src.trainer.result_computer import EpochResultComputer
from src.metrics.database import MetricDB, Metric, MetricSpec

def accuracy(*, preds, targets, **_):
    return float((preds == targets).sum() / len(preds))

DUMMY_METRICS = {
    "accuracy": Metric(
        spec=MetricSpec(
            name="accuracy",
            direction="max",
            input="hard_label",
            description="Classification accuracy",
        ),
        fn = accuracy,
    )
}
db = MetricDB(DUMMY_METRICS)

@pytest.fixture
def generate_data():
    true  = torch.tensor([1, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    preds = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    return true, preds

@pytest.fixture
def generate_computer(generate_data):
    true_tensor, pred_tensor = generate_data
    computer = EpochResultComputer(db)
    computer.record_step(preds=pred_tensor, targets=true_tensor)
    return computer

def test_compute_accuracy(generate_computer):
    computer = generate_computer
    results = computer.compute_all(["accuracy"])
    
    assert "accuracy" in results
    val = results["accuracy"]
    
    assert isinstance(val, float)
    assert val == 0.5

def test_loss(generate_computer):
    computer = generate_computer
    computer.record_loss("train_loss", 0.5)
    computer.record_loss("train_loss", 0.3)
    results = computer.compute_all(["train_loss"])
    
    assert "train_loss" in results
    val = results["train_loss"]
    
    assert isinstance(val, float)
    assert val == 0.4

def test_concat(generate_computer):
    true  = torch.tensor([1, 0, 0, 1, 1])
    preds = torch.tensor([0, 1, 1, 0, 1])

    computer = generate_computer
    computer.record_step(preds=preds, targets=true)
    results = computer.compute_all(["accuracy"])

    assert len(computer._preds) == 2
    assert len(computer._targets) == 2
    assert results["accuracy"] == pytest.approx(0.4, abs=1e-8)

def test_reset(generate_computer):
    computer = generate_computer
    computer.reset()
    assert len(computer._preds) == 0
    assert len(computer._targets) == 0
    assert len(computer._losses) == 0
