from base import AfterStepHook

class AfterStepLoggerHook(AfterStepHook):
    """
    A concrete AfterStepHook that logs information about each step.

    This hook prints the current epoch and step, and optionally the loss 
    if it exists in the step_result dictionary. It does not modify the 
    training loop or step results.
    """
    def execute(self, epoch: int, step: int, results: dict):
        print(f"[LOG] epoch {epoch} step {step}")
        if "loss" in results:
            print(f"[LOG] loss: {results['loss']}")
        else:
            print("No loss value found in results")