import pytest

from src.hooks.after_epoch.logger import AfterEpochLoggerHook

results = {
    "val_loss": 0.03,
    "accuracy": 0.7,
    "f1": 0.62
}

def test_logger():
    logger = AfterEpochLoggerHook(metric="f1")
    logger.execute(epoch=1, results=results)
    
def test_metric_not_exist():
    logger = AfterEpochLoggerHook(metric="mse")
    with pytest.raises(ValueError):
        logger.execute(epoch=1, results=results)
    