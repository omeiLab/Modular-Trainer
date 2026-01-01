class AfterStepHook:
    def __init__(self):
        pass
    
    def execute(self, epoch: int, step: int) -> None:
        print(f"Epoch {epoch}: after_step hook triggered")