from src.hooks.after_epoch.base import AfterEpochHook
from src.control.controller import Controller
from typing import Mapping, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class EarlyStopConfig:
    patience: int = 0
    metric: str = "val_loss"
    maximize: bool = False
    min_delta: float = 0.0
    
    def enabled(self) -> bool:
        return self.patience > 0

class AfterEpochEarlyStopHook(AfterEpochHook):
    """
    Early stopping hook that monitors a specified metric at the end of each epoch
    and signals the controller to stop training if no improvement is seen for
    a given number of consecutive epochs (patience).

    Attributes:
        controller (Controller): The controller object to signal stopping.
        patience (int): Number of consecutive epochs without improvement before stopping.
        metric (str): The key in results dict to monitor (e.g., 'val_loss').
        maximize (bool): Whether to maximize the metric. If False, smaller values are better.
        min_delta (float): Minimum change in the monitored metric to qualify as improvement.
        best_score (float): Best observed metric value.
        no_improved_count (int): Counter of consecutive epochs without improvement.
    """
    def __init__(
        self, controller: Controller, 
        patience: int = 0, 
        metric: str = "loss", 
        maximize: bool = False, 
        min_delta: float = 0.0
    ):
        self.controller = controller
        self.patience = patience
        self.metric = metric
        self.maximize = maximize
        self.min_delta = min_delta
        self.best_score = -np.inf if maximize else np.inf
        self.no_improved_count = 0
        
    def execute(self, epoch: int, results: Mapping[str, Any]) -> None:
        """
        Execute the early stopping check at the end of an epoch.

        Args:
            epoch (int): Current epoch number.
            results (Dict[str, int | float]): Dictionary containing epoch metrics.
        
        Behavior:
            - Updates the best_score if improvement is detected.
            - Increments no_improved_count if no improvement.
            - Calls controller.stop() if no improvement for `patience` epochs.
        """
        if self.metric not in results:
            raise ValueError(f"Metric {self.metric} not found in results")
        score = results[self.metric]
        improved = False
        if self.maximize:
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.no_improved_count = 0
            print(f"Best score updated to: {self.best_score} at epoch {epoch}")
        else:
            self.no_improved_count += 1
            if self.no_improved_count >= self.patience:
                self.controller.stop()
                print(f"[EarlyStop] Stopping at epoch {epoch}")