from typing import List

class BeforeEpochHook:
    """
    Hook that is called before the start of each epoch.

    Subclasses should override the `execute` method to implement 
    any behavior that needs to run at the beginning of an epoch, 
    such as resetting metrics or logging.
    """
    def __init__(self):
        pass
    
    def execute(self, epoch: int) -> None:
        """Execute the hook for the given epoch."""
        return None
        
class CompositeBeforeEpochHook(BeforeEpochHook):
    def __init__(self, hooks: List[BeforeEpochHook]):
        """Initialize the composite hook with a list of hooks."""
        self.hooks = hooks
        
    def execute(self, epoch: int) -> None:
        """Execute all hooks in the list iteratively."""
        for hook in self.hooks:
            hook.execute(epoch)
    