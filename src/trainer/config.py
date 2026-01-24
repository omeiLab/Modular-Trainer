from dataclasses import dataclass
import yaml

@dataclass
class TrainerConfig:
    """
    Configuration for the Trainer class and associated hooks.

    This class centralizes hyperparameters and hook settings for a
    training run. Using a dataclass makes it easy to load from YAML,
    JSON, or other sources, and ensures type safety.

    Attributes:
        num_epochs (int): Number of epochs to train.
            Default: 5
            
        verbose (int): Logger verbosity. 0 disables logging.
            Default: 1
            
        early_stop_patience (Optional[int]): Number of epochs without
            improvement before early stopping. 0 disables ES.
            Default: 0 (disabled)
            
        early_stop_min_delta (float): Minimum change to qualify as
            improvement for early stopping.
            Default: 1e-4
            
        checkpoint_metric (str): Metric name used for saving checkpoints.
            Default: val_loss (validation loss)
            
        checkpoint_min_delta (float): Minimum change to trigger checkpoint.
            Default: 1e-4
    """
    num_epochs: int = 5
    verbose: int = 1
    early_stop_patience: int = 0
    early_stop_metric: str = "val_loss"
    early_stop_min_delta: float = 1e-4
    checkpoint_metric: str = "val_loss"
    checkpoint_min_delta: float = 0.0
    
    @classmethod
    def from_yaml(cls, config_path: str):
        cfg = cls()
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise ValueError(f"Unknown config parameter: {k}")
        return cfg
