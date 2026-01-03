class AfterEpochHook:
    """
    Hook that is called after the completion of each epoch.

    Subclasses should override the `execute` method to implement 
    behavior that should run at the end of an epoch, such as 
    aggregating metrics, adjusting learning rate, or saving checkpoints.
    """
    def __init__(self):
        pass
    
    def execute(self, epoch: int) -> None:
        """Execute the hook for the given epoch."""
        print(f"Epoch {epoch}: after_epoch hook triggered")