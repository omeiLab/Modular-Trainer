from src.trainer.loop import TrainerLoop
from hooks.before_epoch.base import BeforeEpochHook
from hooks.after_step.base import AfterStepHook
from hooks.after_epoch.base import AfterEpochHook
from src.control.controller import Controller


class CaptureAfterStepHook(AfterStepHook):
    def __init__(self):
        self.records = []

    def execute(self, epoch, step, results):
        self.records.append((epoch, step, results))


class DummyTrainerLoop(TrainerLoop):
    def _run_step(self, epoch, step):
        return {"dummy": True, "epoch": epoch, "step": step}


def test_step_result_is_passed_to_after_step_hook():
    before_epoch_hook = BeforeEpochHook()
    after_step_hook = CaptureAfterStepHook()
    after_epoch_hook = AfterEpochHook()
    controller = Controller()

    trainer_loop = DummyTrainerLoop(
        before_epoch_hook,
        after_step_hook,
        after_epoch_hook,
        controller,
    )

    trainer_loop.run(num_epochs=1, steps_per_epoch=2)

    assert len(after_step_hook.records) == 2

    epoch, step, step_result = after_step_hook.records[0]
    assert step_result["dummy"] is True
    assert step_result["epoch"] == epoch
    assert step_result["step"] == step
