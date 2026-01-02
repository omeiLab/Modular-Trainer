from src.hooks.before_epoch import BeforeEpochHook
from src.hooks.after_step import AfterStepHook
from src.hooks.after_epoch import AfterEpochHook
from src.control.controller import Controller

class TrainerLoop:
    def __init__(self, before_epoch_hook: BeforeEpochHook, 
                after_step_hook: AfterStepHook, 
                after_epoch_hook: AfterEpochHook, 
                control: Controller):
        self.before_epoch_hook = before_epoch_hook
        self.after_step_hook = after_step_hook
        self.after_epoch_hook = after_epoch_hook
        self.control = control
    
    def run(self, num_epochs: int, steps_per_epoch: int) -> None:
        for epoch in range(num_epochs):
            self.before_epoch_hook.execute(epoch)
            for step in range(steps_per_epoch):
                results = self._run_step(epoch, step)
                self.after_step_hook.execute(epoch, step, results)
                if not self.control.should_continue():
                    return
            self.after_epoch_hook.execute(epoch)
            
    def _run_step(self, epoch: int, step: int) -> dict:
        # dummy implementation for results
        return {
            "epoch": epoch,
            "step": step,
        }