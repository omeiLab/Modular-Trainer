from src.hooks.after_epoch.base import AfterEpochHook
import numpy as np
from dataclasses import dataclass
from typing import Mapping
import torch.nn as nn

@dataclass
class CheckpointConfig:
    metric: str = "val_loss"
    maximize: bool = False
    min_delta: float = 0.0

class AfterEpochCheckpointHook(AfterEpochHook):
    """
    Hook that saves a checkpoint when a monitored metric improves after an epoch.

    This hook compares a specified metric in the epoch results against the
    best score seen so far. If the metric improves according to the given
    criteria, a user-provided save function is invoked.

    Typical use cases include:
    - Saving the best model checkpoint based on validation loss or accuracy
    - Tracking and persisting the best-performing state during training

    This hook is designed to be executed after each epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        metric: str,
        maximize: bool = True,
        min_delta: float = 0.0,
    ):
        """
        Initialize the checkpoint hook.

        Args:
            metric (str): Name of the metric to monitor (e.g., "val_loss").
            save_fn (Callable): Function to call when a new best score is achieved.
                The function should accept (epoch, results) as arguments.
            maximize (bool): Whether a higher metric value indicates improvement.
                Set to False for metrics like loss.
            min_delta (float): Minimum change in the monitored metric to qualify
                as an improvement.
        """
        self.model = model
        self.metric = metric
        self.maximize = maximize
        self.min_delta = min_delta
        self.best_score = -np.inf if maximize else np.inf
        
    # dummy implementation, overwrite for real implementation
    def save_fn(self, epoch: int) -> None:
        print(f"[Checkpoint] Saving checkpoint at epoch {epoch}")
        
    def execute(self, epoch: int, results: Mapping[str, float]) -> None:
        """
        Execute the checkpoint logic after an epoch.

        Args:
            epoch (int): Current epoch index.
            results (Dict[str, float]): Dictionary containing epoch-level metrics.

        Raises:
            ValueError: If the monitored metric is not found in results.
        """
        if self.metric not in results:
            raise ValueError(
                f"[{self.__class__.__name__}] "
                f"Monitor '{self.metric}' not found in results. "
                f"Available metrics: {list(results.keys())}"
            )

        current = results[self.metric]

        if self.maximize:
            improve = current > self.best_score + self.min_delta
        else:
            improve = current < self.best_score - self.min_delta

        if improve:
            self.best_score = current
            self.save_fn(epoch)
            
    
