import torch
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
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

def roc_auc(*, preds, targets, **_):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return float(roc_auc_score(y_score=preds, y_true=targets))

def pr_auc(*, preds, targets, **_):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return float(average_precision_score(y_score=preds, y_true=targets))

def log_loss(*, preds, targets, **_):
    preds = torch.clamp(preds, EPS, 1 - EPS)
    targets = targets.detach().cpu().numpy()
    return log_loss(preds=preds, targets=targets)

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
    "roc-auc": Metric(
        spec=MetricSpec(
            name="roc-auc",
            direction="max",
            input="probability",
            description="Receiver Operating Characteristic Area Under Curve",
        ),
        fn = roc_auc,
    ),
    "pr-auc": Metric(
        spec=MetricSpec(
            name="pr-auc",
            direction="max",
            input="probability",
            description="Precision-Recall Area Under Curve"
        ),
        fn=pr_auc
    ),
    "log-loss": Metric(
        spec=MetricSpec(
            name="log-loss",
            direction="min",
            input="probability",
            description="Logarithmic loss",
        ),
        fn=log_loss
    )
}