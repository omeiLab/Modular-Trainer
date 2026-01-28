import numpy as np
from metrics.database import Metric, MetricSpec

EPS = 1e-12

def mean_squared_error(*, preds, targets, **_):
    return np.mean((preds - targets) ** 2)

def root_mean_squared_error(*, preds, targets, **_):
    return np.sqrt(np.mean((preds - targets) ** 2))

def mean_absolute_error(*, preds, targets, **_):
    return np.mean(np.abs(preds - targets))

def r_squared(*, preds, targets, **_):
    mean = np.mean(targets)
    ss_total = np.sum((targets - mean) ** 2)
    ss_residual = np.sum((preds - targets) ** 2)
    return 1 - (ss_residual / ss_total)

def accuracy(*, preds, targets, **_):
    return float((preds == targets).mean())

def precision(*, preds, targets, **_):
    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))
    return tp / (tp + fp + EPS)

def recall(*, preds, targets, **_):
    tp = np.sum((preds == 1) & (targets == 1))
    fn = np.sum((preds == 0) & (targets == 1))
    return tp / (tp + fn + EPS)

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
        fn = mean_squared_error
    ),
    "rmse": Metric(
        spec=MetricSpec(
            name="rmse",
            direction="min",
            description="Root mean squared error",
        ),
        fn = root_mean_squared_error
    ),
    "mae": Metric(
        spec=MetricSpec(
            name="mae",
            direction="min",
            description="Mean absolute error",
        ),
        fn = mean_absolute_error
    ),
    "r2": Metric(
        spec=MetricSpec(
            name="r2",
            direction="max",
            description="Determination coefficient (R^2)",
        ),
        fn = r_squared
    ),
    "accuracy": Metric(
        spec=MetricSpec(
            name="accuracy",
            direction="max",
            description="Classification accuracy",
        ),
        fn = accuracy,
    ),
    "precision": Metric(
        spec=MetricSpec(
            name="precision",
            direction="max",
            description="Classification precision",
        ),
        fn = precision,
    ),
    "recall": Metric(
        spec=MetricSpec(
            name="recall",
            direction="max",
            description="Classification recall",
        ),
        fn = recall,
    ),
    "f1": Metric(
        spec=MetricSpec(
            name="f1",
            direction="max",
            description="Classification F1-score",
        ),
        fn = f1_score,
    ),
}