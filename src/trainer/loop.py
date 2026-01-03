from src.hooks.before_epoch.base import CompositeBeforeEpochHook
from src.hooks.after_step.base import CompositeAfterStepHook
from src.hooks.after_epoch.base import CompositeAfterEpochHook
from src.control.controller import Controller

class TrainerLoop:
    """
    Modular training loop that orchestrates the training process 
    using hooks and a controller.

    The loop delegates all side-effects and logging to hooks, 
    and consults the controller to determine whether to continue.
    """
    def __init__(
        self, 
        before_epoch_hook: CompositeBeforeEpochHook, 
        after_step_hook: CompositeAfterStepHook, 
        after_epoch_hook: CompositeAfterEpochHook, 
        control: Controller
    ):
        """
        Initialize the TrainerLoop with hooks and a controller.

        Args:
            before_epoch_hook (CompositeBeforeEpochHook): Hooks to execute before each epoch.
            after_step_hook (CompositeAfterStepHook): Hooks to execute after each step.
            after_epoch_hook (CompositeAfterEpochHook): Hooks to execute after each epoch.
            control (Controller): Controller to query for stopping criteria.
        """
        self.before_epoch_hook = before_epoch_hook
        self.after_step_hook = after_step_hook
        self.after_epoch_hook = after_epoch_hook
        self.control = control
    
    def run(self, num_epochs: int, steps_per_epoch: int) -> None:
        """
        Run the training loop for the specified number of epochs and steps.

        Args:
            num_epochs (int): Number of epochs to run.
            steps_per_epoch (int): Number of steps per epoch.
        """
        for epoch in range(num_epochs):
            self.before_epoch_hook.execute(epoch)
            for step in range(steps_per_epoch):
                self.after_step_hook.execute(epoch, step)
            results = self._run_step(epoch)
            self.after_epoch_hook.execute(epoch, results)
            if not self.control.should_continue():
                return
            
    def _run_step(self, epoch: int) -> dict:
        """
        Execute a single training step and return the step results.

        This is a dummy implementation for testing purposes. 
        Subclasses should override this method with actual training logic.

        Args:
            epoch (int): Current epoch number.

        Returns:
            dict: Step results containing metrics such as loss.
        """
        return {
            "epoch": epoch,
        }