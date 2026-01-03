from src.hooks.after_epoch.base import AfterEpochHook
from typing import Dict

class AfterEpochLoggerHook(AfterEpochHook):
    """
    A concrete AfterEpochHook that logs information about each step.

    This hook prints the current epoch and step, and optionally the loss 
    if it exists in the step_result dictionary. It does not modify the 
    training loop or step results.
    """
    def execute(self, epoch: int, results: Dict[str, int | float]) -> None:
        if "loss" in results:
            print(f"[LOG] loss: {results['loss']}")
        else:
            print("No loss value found in results")