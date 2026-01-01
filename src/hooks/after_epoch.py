class AfterEpochHook:
    def __init__(self):
        pass
    
    def execute(self, epoch: int) -> None:
        print(f"Epoch {epoch}: after_epoch hook triggered")