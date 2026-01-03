class Controller:
    """
    A simple controller that can be queried by the training loop 
    to determine whether to continue training.

    Subclasses or extensions may add logic for early stopping, 
    learning rate scheduling, or checkpoint-based termination.
    """
    def __init__(self):
        self.stop_learning = False
        
    def stop(self):
        """Hook can call this to request stopping the loop."""
        self.stop_learning = True
    
    def should_continue(self) -> bool:
        """Loop calls this each step/epoch to decide whether to continue."""
        return not self.stop_learning