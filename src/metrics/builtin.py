import torch
from src.metrics.database import Metric, MetricSpec

EPS = 1e-12

def mean_squared_error(*, preds, targets, **_):
    return torch.mean((preds - targets) ** 2).item()

def root_mean_squared_error(*, preds, targets, **_):
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

def mean_absolute_error(*, preds, targets, **_):
    return torch.mean(torch.abs(preds - targets)).item()

def r_squared(*, preds, targets, **_):
    mean = torch.mean(targets)
    ss_total = torch.sum((targets - mean) ** 2)
    ss_residual = torch.sum((preds - targets) ** 2)
    return 1 - (ss_residual / ss_total).item()

def accuracy(*, preds, targets, **_):
    return float((preds == targets).sum() / len(preds))

def precision(*, preds, targets, **_):
    tp = torch.sum((preds == 1) & (targets == 1))
    fp = torch.sum((preds == 1) & (targets == 0))
    return (tp / (tp + fp + EPS)).item()

def recall(*, preds, targets, **_):
    tp = torch.sum((preds == 1) & (targets == 1))
    fn = torch.sum((preds == 0) & (targets == 1))
    return (tp / (tp + fn + EPS)).item()

def f1_score(*, preds, targets, **_):
    p = precision(preds=preds, targets=targets)
    r = recall(preds=preds, targets=targets)
    return 2 * p * r / (p + r + EPS)

BUILTIN_METRICS = {
    "mse": Metric(
        spec=MetricSpec(
            name="mse",
            direction="min",
            input="logits",
            description="Mean squared error",
        ),
        fn = mean_squared_error,
    ),
    "rmse": Metric(
        spec=MetricSpec(
            name="rmse",
            direction="min",
            input="logits",
            description="Root mean squared error",
        ),
        fn = root_mean_squared_error,
    ),
    "mae": Metric(
        spec=MetricSpec(
            name="mae",
            direction="min",
            input="logits",
            description="Mean absolute error",
        ),
        fn = mean_absolute_error,
    ),
    "r2": Metric(
        spec=MetricSpec(
            name="r2",
            direction="max",
            input="logits",
            description="Determination coefficient (R^2)",
        ),
        fn = r_squared,
    ),
    "accuracy": Metric(
        spec=MetricSpec(
            name="accuracy",
            direction="max",
            input="hard_label",
            description="Classification accuracy",
        ),
        fn = accuracy,
    ),
    "precision": Metric(
        spec=MetricSpec(
            name="precision",
            direction="max",
            input="hard_label",
            description="Classification precision",
        ),
        fn = precision,
    ),
    "recall": Metric(
        spec=MetricSpec(
            name="recall",
            direction="max",
            input="hard_label",
            description="Classification recall",
        ),
        fn = recall,
    ),
    "f1": Metric(
        spec=MetricSpec(
            name="f1",
            direction="max",
            input="hard_label",
            description="Classification F1-score",
        ),
        fn = f1_score,
    ),
}