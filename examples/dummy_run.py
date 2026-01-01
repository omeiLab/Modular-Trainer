from src.trainer.loop import TrainerLoop
from src.hooks.before_epoch import BeforeEpochHook
from src.hooks.after_step import AfterStepHook
from src.hooks.after_epoch import AfterEpochHook
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