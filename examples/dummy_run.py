from src.trainer.loop import TrainerLoop
from hooks.before_epoch.base import BeforeEpochHook
from hooks.after_step.base import AfterStepHook
from hooks.after_epoch.base import AfterEpochHook
from src.control.controller import Controller

# initialize hooks and controller
before_epoch_hook = BeforeEpochHook()
after_step_hook = AfterStepHook()
after_epoch_hook = AfterEpochHook()
controller = Controller()

# initialize trainer loop
trainer_loop = TrainerLoop(before_epoch_hook, after_step_hook, after_epoch_hook, controller)

# dummy hyperparameters
epoch = 2
step = 3

# run the trainer loop
trainer_loop.run(epoch, step)