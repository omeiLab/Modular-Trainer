import pytest
import yaml

from src.trainer.config import TrainerConfig

def test_empty_config():
    config = TrainerConfig().from_yaml("test/config_test/yml/empty_config.yaml")

    assert config.num_epochs == 5
    assert config.task == "regression"
    assert config.metrics == []
    assert config.log.verbose == 1
    assert config.early_stop.patience == 0
    assert config.checkpoint.metric == "val_loss"

def test_miss_section_config():
    config = TrainerConfig().from_yaml("test/config_test/yml/miss_section_config.yaml")

    assert config.early_stop.patience == 0
    assert config.early_stop.metric == "val_loss"
    assert config.early_stop.min_delta == 0.0
    assert config.early_stop.enabled() == False

def test_miss_hyper_config():
    config = TrainerConfig().from_yaml("test/config_test/yml/miss_hyper_config.yaml")

    assert config.early_stop.min_delta == 0.0
    assert config.checkpoint.min_delta == 0.0
