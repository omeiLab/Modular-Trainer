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
            
        early_stop_metric (str): Metric name used for monitoring.
            Default: val_loss (validation loss)
            
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
    # trainer
    num_epochs: int = 5
    
    # log
    verbose: int = 1

    # early stopping
    early_stop_patience: int = 0
    early_stop_metric: str = "val_loss"
    early_stop_min_delta: float = 1e-4

    # checkpoint
    checkpoint_metric: str = "val_loss"
    checkpoint_min_delta: float = 0.0

    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainerConfig":
        """
        Load TrainerConfig from a hierarchical YAML file.

        The YAML file may contain multiple sections (trainer, early_stop,
        checkpoint, etc.). Only relevant fields are extracted.
        """
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()

        # ---- trainer ----
        trainer_cfg = data.get("trainer", {})
        cfg.num_epochs = trainer_cfg.get("num_epochs", cfg.num_epochs)
        
        # ---- log ----
        log_cfg = data.get("log", {})
        cfg.verbose = log_cfg.get("verbose", cfg.verbose)

        # ---- early stop ----
        es_cfg = data.get("early_stop", {})

        cfg.early_stop_patience = es_cfg.get(
            "patience", cfg.early_stop_patience
        )
        cfg.early_stop_metric = es_cfg.get(
            "metric", cfg.early_stop_metric
        )
        cfg.early_stop_min_delta = es_cfg.get(
            "min_delta", cfg.early_stop_min_delta
        )

        # ---- checkpoint ----
        ckpt_cfg = data.get("checkpoint", {})

        cfg.checkpoint_metric = ckpt_cfg.get(
            "metric", cfg.checkpoint_metric
        )
        cfg.checkpoint_min_delta = ckpt_cfg.get(
            "min_delta", cfg.checkpoint_min_delta
        )

        return cfg
