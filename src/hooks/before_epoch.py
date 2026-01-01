class BeforeEpochHook:
    def __init__(self):
        pass
    
    def execute(self, epoch: int) -> None:
        print(f"Epoch {epoch}: before_epoch hook triggered")