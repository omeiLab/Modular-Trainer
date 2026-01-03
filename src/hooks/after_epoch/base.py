from typing import List
from typing import Dict

class AfterEpochHook:
    """
    Hook that is called after the completion of each epoch.

    Subclasses should override the `execute` method to implement 
    behavior that should run at the end of an epoch, such as 
    aggregating metrics, adjusting learning rate, or saving checkpoints.
    """
    def __init__(self):
        pass
    
    def execute(self, epoch: int, results: Dict[str, int | float]) -> None:
        """Execute the hook for the given epoch."""
        return None
        
class CompositeAfterEpochHook(AfterEpochHook): 
    def __init__(self, hooks: List[AfterEpochHook]):
        """Initialize the composite hook with a list of hooks."""
        self.hooks = hooks
        
    def execute(self, epoch: int, results: Dict[str, int | float]) -> None:
        """Execute all hooks in the list iteratively."""
        for hook in self.hooks:
            hook.execute(epoch, results)