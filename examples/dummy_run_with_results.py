from src.trainer.loop import TrainerLoop
from src.hooks.before_epoch import BeforeEpochHook
from src.hooks.after_step import AfterStepHook
from src.hooks.after_epoch import AfterEpochHook
from src.control.controller import Controller


class DummyAfterStepHook(AfterStepHook):
    def __call__(self, epoch, step, step_result):
        print(
            f"[AfterStepHook] epoch={epoch}, step={step}, "
            f"step_result={step_result}"
        )


class DummyTrainerLoop(TrainerLoop):
    def _run_step(self, epoch, step):
        # dummy step result (opaque to loop)
        return {
            "loss": 0.1 * step,
            "epoch": epoch,
            "step": step,
        }


# initialize hooks and controller
before_epoch_hook = BeforeEpochHook()
after_step_hook = DummyAfterStepHook()
after_epoch_hook = AfterEpochHook()
controller = Controller()

# initialize trainer loop
trainer_loop = DummyTrainerLoop(
    before_epoch_hook,
    after_step_hook,
    after_epoch_hook,
    controller,
)

# dummy hyperparameters
epochs = 2
steps_per_epoch = 3

# run
trainer_loop.run(epochs, steps_per_epoch)
