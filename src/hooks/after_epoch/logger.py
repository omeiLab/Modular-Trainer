from src.hooks.after_epoch.base import AfterEpochHook
from typing import Mapping, Any
from dataclasses import dataclass

@dataclass
class LoggerConfig:
    metric: str = "val_loss"    # support only 1 metric for now
    verbose: int = 1
    
    def enabled(self) -> bool:
        return self.verbose > 0

class AfterEpochLoggerHook(AfterEpochHook):
    """
    A concrete AfterEpochHook that logs information about each step.

    This hook prints the current epoch and step, and optionally the loss 
    if it exists in the step_result dictionary. It does not modify the 
    training loop or step results.
    """
    def __init___(self, metric: str = "val_loss"):
        self.metric = metric
        
    def execute(self, epoch: int, results: Mapping[str, Any]) -> None:
        if self.metric in results:
            print(f"[LOG] {self.metric}: {results[self.metric]}")
        else:
            raise ValueError(f"Metric {self.metric} not found in results")