class AfterStepHook:
    """
    Hook that is called after each training step (batch).

    Subclasses should override the `execute` method to implement 
    behavior that depends on the step results, such as logging, 
    checkpointing, or early stopping.
    """
    def __init__(self):
        pass
    
    def execute(self, epoch: int, step: int, results: dict) -> None:
        """Execute the hook for the given step and step result."""
        print(f"Epoch {epoch}: after_step hook triggered")