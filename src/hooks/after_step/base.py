from typing import List

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
        return None
        
class CompositeAfterStepHook(AfterStepHook):
    def __init__(self, hooks: List[AfterStepHook]):
        """Initialize the composite hook with a list of hooks."""
        self.hooks = hooks
        
    def execute(self, epoch: int, step: int, results: dict) -> None:
        """Execute all hooks in the list iteratively."""
        for hook in self.hooks:
            hook.execute(epoch, step, results)