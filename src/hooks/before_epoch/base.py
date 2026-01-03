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
        print(f"Epoch {epoch}: before_epoch hook triggered")