from hooks.before_epoch.base import BeforeEpochHook
from hooks.after_step.base import AfterStepHook
from hooks.after_epoch.base import AfterEpochHook
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
        before_epoch_hook: BeforeEpochHook, 
        after_step_hook: AfterStepHook, 
        after_epoch_hook: AfterEpochHook, 
        control: Controller
    ):
        """
        Initialize the TrainerLoop with hooks and a controller.

        Args:
            before_epoch_hook (BeforeEpochHook): Hook to execute before each epoch.
            after_step_hook (AfterStepHook): Hook to execute after each step.
            after_epoch_hook (AfterEpochHook): Hook to execute after each epoch.
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
                results = self._run_step(epoch, step)
                self.after_step_hook.execute(epoch, step, results)
                if not self.control.should_continue():
                    return
            self.after_epoch_hook.execute(epoch)
            
    def _run_step(self, epoch: int, step: int) -> dict:
        """
        Execute a single training step and return the step results.

        This is a dummy implementation for testing purposes. 
        Subclasses should override this method with actual training logic.

        Args:
            epoch (int): Current epoch number.
            step (int): Current step number.

        Returns:
            dict: Step results containing metrics such as loss.
        """
        return {
            "epoch": epoch,
            "step": step,
        }