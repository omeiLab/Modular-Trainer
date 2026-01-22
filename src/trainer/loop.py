from src.hooks.before_epoch.base import CompositeBeforeEpochHook
from src.hooks.after_step.base import CompositeAfterStepHook
from src.hooks.after_epoch.base import CompositeAfterEpochHook
from src.control.controller import Controller
from src.trainer.result_builder import EpochResultBuilder 
from src.trainer.runner import Runner
from typing import Mapping, Any

class TrainerLoop:
    """
    Modular training loop that orchestrates multi-epoch training
    using hooks, a controller, and a Runner for epoch execution.

    Responsibilities:
        - Execute before-epoch hooks
        - Delegate training/validation for one epoch to Runner
        - Execute after-epoch hooks with aggregated results
        - Check Controller to determine if training should stop
    """

    def __init__(
        self, 
        before_epoch_hook: CompositeBeforeEpochHook, 
        after_step_hook: CompositeAfterStepHook, 
        after_epoch_hook: CompositeAfterEpochHook, 
        control: Controller,
        runner: Runner,
        result_builder: EpochResultBuilder
    ):
        """
        Initialize the TrainerLoop with hooks, a controller, a Runner, and a result builder.

        Args:
            before_epoch_hook (CompositeBeforeEpochHook): Hooks to execute before each epoch.
            after_step_hook (CompositeAfterStepHook): Hooks to execute after each step (currently not used).
            after_epoch_hook (CompositeAfterEpochHook): Hooks to execute after each epoch with aggregated results.
            control (Controller): Controller object to query for early stopping or other stop criteria.
            runner (Runner): Runner object responsible for executing one epoch of train + validation.
            result_builder (EpochResultBuilder): Builder to record and aggregate metrics per epoch.
        """
        self.before_epoch_hook = before_epoch_hook
        self.after_step_hook = after_step_hook
        self.after_epoch_hook = after_epoch_hook
        self.control = control
        self.runner = runner
        self.result_builder = result_builder
    
    def run(self, num_epochs: int) -> None:
        """
        Run the training loop for multiple epochs.

        Args:
            num_epochs (int): Number of epochs to run.
            steps_per_epoch (int): Number of steps per epoch. 
                                   (Currently unused since Runner handles batching.)
        """
        for epoch in range(num_epochs):
            self.result_builder.reset()
            self.before_epoch_hook.execute(epoch)
            results = self._run_epoch()
            self.after_epoch_hook.execute(epoch, results)
            if not self.control.should_continue():
                return
            
    def _run_epoch(self) -> Mapping[str, Any]:
        """
        Execute a single epoch using the Runner and collect epoch-level metrics.

        Returns:
            dict: Aggregated epoch metrics (train + validation) from the Runner.
        """
        epoch_results = self.runner.run_one_epoch()
        return epoch_results
