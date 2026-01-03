from typing import Dict
import numpy as np

from src.trainer.loop import TrainerLoop
from src.control.controller import Controller
from src.hooks.before_epoch.base import CompositeBeforeEpochHook
from src.hooks.after_epoch.base import CompositeAfterEpochHook, AfterEpochHook
from src.hooks.after_epoch.early_stop import AfterEpochEarlyStopHook
from src.hooks.after_epoch.logger import AfterEpochLoggerHook
from src.hooks.after_step.base import CompositeAfterStepHook

# ----------------------------
# Logger hook
# ----------------------------
class LoggerAfterEpochHook(AfterEpochHook):
    """Simple logger that prints metrics at the end of each epoch."""
    def execute(self, epoch: int, results: Dict[str, int | float]) -> None:
        print(f"[Logger] Epoch {epoch} metrics: {results}")

# ----------------------------
# Dummy TrainerLoop with aggregated epoch results
# ----------------------------
class DummyTrainerLoop(TrainerLoop):
    def _run_step(self, epoch: int) -> Dict[str, float]:
        # dummy step returns random loss for demonstration
        return {"loss": np.random.rand() * (1.0 - 0.05 * epoch)}  # decreasing roughly

    def aggregate_epoch_results(self, epoch: int, steps_per_epoch: int) -> Dict[str, float]:
        # simulate validation loss decreasing each epoch
        val_loss = 1.0 - 0.1 * epoch
        return {"val_loss": val_loss}

# ----------------------------
# Setup dummy run
# ----------------------------
# Controller
controller = Controller()

# Hooks
logger_hook = AfterEpochLoggerHook()
early_stop_hook = AfterEpochEarlyStopHook(
    controller=controller,
    patience=2,  # stop if val_loss doesn't improve for 2 epochs
    metric="loss",
    maximize=False,
    min_delta=0.01
)

# CompositeAfterEpochHook
after_epoch_hooks = CompositeAfterEpochHook([logger_hook, early_stop_hook])

# CompositeBeforeEpochHook (empty for now)
before_epoch_hooks = CompositeBeforeEpochHook([])

# CompositeAfterStepHook (empty for now)
after_step_hooks = CompositeAfterStepHook([])

# Dummy trainer loop
trainer_loop = DummyTrainerLoop(
    before_epoch_hook=before_epoch_hooks,
    after_step_hook=after_step_hooks,  
    after_epoch_hook=after_epoch_hooks,
    control=controller
)

# Run dummy training
num_epochs = 10
steps_per_epoch = 5

# Run training loop
trainer_loop.run(num_epochs=num_epochs, steps_per_epoch=steps_per_epoch)
