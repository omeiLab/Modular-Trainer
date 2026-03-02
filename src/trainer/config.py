from dataclasses import dataclass, field
from typing import List, Literal
import yaml

from src.hooks.after_epoch.early_stop import EarlyStopConfig
from src.hooks.after_epoch.checkpoint import CheckpointConfig
from src.hooks.after_epoch.logger import LoggerConfig

@dataclass
class TrainerConfig:
    num_epochs: int = 5
    task: Literal["regression", "binary"] = "regression"    # add multiclass later
    metrics: List[str] = field(default_factory=list)
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log: LoggerConfig = field(default_factory=LoggerConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainerConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # helper: avoid KeyError
        def load_section(section_cls, data_dict, key):
            section_data = data_dict.get(key, {})
            return section_cls(**section_data) if section_data else section_cls()

        return cls(
            num_epochs = data.get("trainer", {}).get("num_epochs", 5),
            task = data.get("trainer", {}).get("task", "regression"),
            metrics = data.get("metrics", []),
            early_stop = load_section(EarlyStopConfig, data,  "early_stop"),
            checkpoint = load_section(CheckpointConfig, data, "checkpoint"),
            log = load_section(LoggerConfig, data, "log"),
        )
