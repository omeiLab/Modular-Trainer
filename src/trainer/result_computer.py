from typing import Dict, List, Optional
import numpy as np
import torch

from src.metrics.database import MetricDB

class EpochResultComputer:
    """
    Epoch-level metric computation engine.

    This class is responsible for:

    1. Collecting step-level raw outputs (predictions, targets).
    2. Collecting step-level scalar losses.
    3. Computing all registered metrics at epoch end.
    4. Remaining agnostic to metric definitions (delegated to MetricDB).

    Design principles:
    ------------------
    - Accumulate raw tensors during steps (no step-level aggregation).
    - Perform metric computation only at epoch end.
    - Avoid hard-coded metric names (losses are treated generically).
    - Avoid O(n^2) tensor concatenation by accumulating into lists.
    - Remain stateless across epochs via explicit reset().

    Lifecycle:
    ----------
    epoch start:
        computer.reset()

    each step:
        computer.record_step(preds, targets)
        computer.record_loss("train_loss", value)

    epoch end:
        results = computer.compute_all(metric_names)
    """

    def __init__(self, metric_db: MetricDB, task_type: str = "regression"):
        """
        Initialize the EpochResultComputer.

        Args:
            metric_db (MetricDB):
                A registry that maps metric names to callable metric functions.
                Each metric function must accept:
                    fn(preds: torch.Tensor, targets: torch.Tensor) -> float
        """
        self.metric_db = metric_db
        self.task_type = task_type
        self.reset()

    def reset(self) -> None:
        """
        Reset all internal buffers.

        This must be called at the beginning of each epoch.
        """
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._losses: Dict[str, List[float]] = {}
        self._buffer: Dict[str, torch.Tensor] = {}

    def record_step(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Record step-level predictions and targets.

        Args:
            preds (torch.Tensor):
                Model predictions for the current step.
            targets (torch.Tensor):
                Ground-truth targets for the current step.

        Raises:
            AssertionError:
                If preds and targets have mismatched shapes.
        """
        assert preds.shape == targets.shape, (
            f"Preds and targets must have same shape, "
            f"got {preds.shape} and {targets.shape}"
        )

        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def record_loss(self, name: str, value: float) -> None:
        """
        Record a scalar loss value for the current step.

        This method is generic and supports multiple loss names
        (e.g., 'train_loss', 'val_loss', 'aux_loss').

        Args:
            name (str):
                Name of the loss metric.
            value (float):
                Scalar loss value for the current step.
        """
        self._losses.setdefault(name, []).append(float(value))

    def compute_all(self, metric_names: List[str]) -> Dict[str, float]:
        """
        Compute all requested metrics for the current epoch.

        This method performs tensor concatenation only once and
        applies the appropriate metric function from MetricDB.

        Args:
            metric_names (List[str]):
                List of metric names to compute.

        Returns:
            Dict[str, float]:
                Mapping from metric name to computed epoch-level value.
        """
        results: Dict[str, float] = {}

        # Concatenate predictions and targets once
        preds: Optional[torch.Tensor] = (
            torch.cat(self._preds) if self._preds else None
        )
        targets: Optional[torch.Tensor] = (
            torch.cat(self._targets) if self._targets else None
        )

        for name in metric_names:

            # Loss metrics
            if name in self._losses:
                results[name] = float(np.mean(self._losses[name]))
                continue

            # Other metrics from MetricDB
            metric_entry = self.metric_db.get(name)
            metric_input = metric_entry.spec.input
            metric_fn = metric_entry.fn

            if preds is None or targets is None:
                raise RuntimeError(
                    f"Metric '{name}' requires predictions and targets, "
                    "but no step data was recorded."
                )
            
            # transform preds to the required input format for the metric function
            if metric_input == "probability":
                preds = self.logit2proba(preds, task=self.task_type)
            elif metric_input == "hard_label":
                preds = self.logit2label(preds, task=self.task_type)

            results[name] = metric_fn(preds=preds, targets=targets)

        return results
    
    def logit2proba(self, logits: torch.Tensor, task: str = "binary") -> torch.Tensor:
        """
        Convert logits to probabilities using sigmoid or softmax.

        This is a utility function that can be used by metric functions
        that require probability inputs.

        Args:
            logits (torch.Tensor):
                Logits tensor of shape (batch_size, num_classes).

        Returns:
            torch.Tensor:
                Probability tensor of shape (batch_size, num_classes).
        """
        if self._buffer.get("probability") is not None:
            return self._buffer["probability"]
        
        proba = torch.sigmoid(logits) if task == "binary" else torch.softmax(logits, dim=-1)
        self._buffer["probability"] = proba
        return proba
    
    def logit2label(self, logits: torch.Tensor, task: str = "binary", threshold: float = 0.5) -> torch.Tensor:
        """
        Convert logits to hard labels. It must be converted to probability first.

        For binary classification, applies a threshold of 0.5.
        For multiclass classification, takes the argmax.

        Args:
            probs (torch.Tensor):
                Probability tensor of shape (batch_size, num_classes).

        Returns:
            torch.Tensor:
                Hard label tensor of shape (batch_size,).
        """
        if self._buffer.get("hard_label") is not None:
            return self._buffer["hard_label"]
        
        proba = self.logit2prob(logits)
        labels = (proba >= threshold).long() if task == "binary" else torch.argmax(proba, dim=-1)
        self._buffer["hard_label"] = labels
        return labels