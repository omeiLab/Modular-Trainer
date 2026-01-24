from typing import Callable, Optional
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.optimizer as optim
import yaml

from src.trainer.loop import TrainerLoop
from src.trainer.runner import Runner
from src.trainer.result_builder import EpochResultBuilder
from src.trainer.config import TrainerConfig
from src.control.controller import Controller
from src.hooks.before_epoch.base import CompositeBeforeEpochHook
from src.hooks.after_epoch.base import CompositeAfterEpochHook
from src.hooks.after_epoch.early_stop import AfterEpochEarlyStopHook, EarlyStopConfig
from src.hooks.after_epoch.logger import AfterEpochLoggerHook, LoggerConfig
from src.hooks.after_epoch.checkpoint import AfterEpochCheckpointHook, CheckpointConfig
from src.hooks.after_step.base import CompositeAfterStepHook

class Trainer:
    """
    Modular Trainer that wraps model, optimizer, dataloaders, hooks,
    and result aggregation into a single interface.
    
    Provides methods to build the training loop and run training.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config_path: Optional[str] = None
    ):
        """
        Initialize the Trainer with core components.

        Args:
            model (nn.Module): PyTorch model to train.
            optimizer (optim.Optimizer): Optimizer for model updates.
            loss_fn (Callable): Loss function.
            train_loader (DataLoader): Training dataset loader.
            val_loader (DataLoader): Validation dataset loader.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config_path = config_path
        self.generate_config()

    def generate_config(self) -> None:
        """
        Load hyperparameters and hook settings from a YAML/JSON config.

        This method can be used to set:
        - num_epochs
        - hook parameters (early stopping, checkpointing, etc.)
        - metrics to track
        - any other training hyperparameters
        """
        config = TrainerConfig()
        if self.config_path:
            config = config.from_yaml(self.config_path)
            
        # LoggerConfig
        self.log_config = LoggerConfig(verbose=config.verbose)
        
        # EarlyStopConfig
        self.es_config = EarlyStopConfig(patience=config.early_stop_patience, metric=config.early_stop_metric, min_delta=config.early_stop_min_delta)
        
        # CheckpointConfig
        self.ckpt_config = CheckpointConfig(metric=config.checkpoint_metric, min_delta=config.checkpoint_min_delta)
        
        # Other hyperparameters
        self.num_epochs = config.num_epochs
                
    def build(self) -> None:
        """
        Construct hooks, result builder, runner, and training loop.

        After calling this method, `self.trainer_loop` is ready to run.
        """
        # Controller
        controller = Controller()

        # Before-epoch hooks
        before_epoch_hooks_lst = []
        before_epoch_hooks = CompositeBeforeEpochHook(before_epoch_hooks_lst)

        # After-step hooks
        after_step_hooks_lst = []
        after_step_hooks = CompositeAfterStepHook(after_step_hooks_lst)

        # After-epoch hooks
        after_epoch_hooks_lst = []
        if self.log_config.enabled():
            logger_hook = AfterEpochLoggerHook()
            after_epoch_hooks_lst.append(logger_hook)
        
        if self.es_config.enabled():
            early_stop_hook = AfterEpochEarlyStopHook(
                controller=controller,
                patience=self.es_config.patience,
                metric=self.es_config.metric,
                maximize=self.es_config.maximize,
                min_delta=self.es_config.min_delta
            )
            after_epoch_hooks_lst.append(early_stop_hook)
        
        checkpoint_hook = AfterEpochCheckpointHook(
            model=self.model,
            metric=self.ckpt_config.metric,
            maximize=self.ckpt_config.maximize,
            min_delta=self.ckpt_config.min_delta
        )
        after_epoch_hooks_lst.append(checkpoint_hook)
        after_epoch_hooks = CompositeAfterEpochHook(after_epoch_hooks_lst)

        # EpochResultBuilder
        result_builder = EpochResultBuilder()

        # Runner
        runner = Runner(
            self.model, self.optimizer, self.loss_fn,
            self.train_loader, self.val_loader, result_builder
        )

        # Trainer loop
        self.trainer_loop = TrainerLoop(
            before_epoch_hooks,
            after_step_hooks,
            after_epoch_hooks,
            controller,
            runner,
            result_builder
        )

    def run(self) -> nn.Module:
        """
        Execute the training loop for the specified number of epochs.

        Returns:
            nn.Module: Trained PyTorch model.
        """
        self.trainer_loop.run(self.num_epochs)
        return self.model
