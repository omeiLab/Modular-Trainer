class Controller:
    """
    A simple controller that can be queried by the training loop 
    to determine whether to continue training.

    Subclasses or extensions may add logic for early stopping, 
    learning rate scheduling, or checkpoint-based termination.
    """
    def __init__(self):
        pass
    
    def should_continue(self) -> bool:
        """Return True if the training loop should continue, False to stop."""
        return True