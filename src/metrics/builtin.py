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
    return float((preds == targets).mean())

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
            description="Mean squared error",
        ),
        fn = mean_squared_error,
        reduce="avg"
    ),
    "rmse": Metric(
        spec=MetricSpec(
            name="rmse",
            direction="min",
            description="Root mean squared error",
        ),
        fn = root_mean_squared_error,
        reduce="avg"
    ),
    "mae": Metric(
        spec=MetricSpec(
            name="mae",
            direction="min",
            description="Mean absolute error",
        ),
        fn = mean_absolute_error,
        reduce="avg"
    ),
    "r2": Metric(
        spec=MetricSpec(
            name="r2",
            direction="max",
            description="Determination coefficient (R^2)",
        ),
        fn = r_squared,
        reduce="avg"
    ),
    "accuracy": Metric(
        spec=MetricSpec(
            name="accuracy",
            direction="max",
            description="Classification accuracy",
        ),
        fn = accuracy,
        reduce="avg"
    ),
    "precision": Metric(
        spec=MetricSpec(
            name="precision",
            direction="max",
            description="Classification precision",
        ),
        fn = precision,
        reduce="avg"
    ),
    "recall": Metric(
        spec=MetricSpec(
            name="recall",
            direction="max",
            description="Classification recall",
        ),
        fn = recall,
        reduce="avg"
    ),
    "f1": Metric(
        spec=MetricSpec(
            name="f1",
            direction="max",
            description="Classification F1-score",
        ),
        fn = f1_score,
        reduce="avg"
    ),
}