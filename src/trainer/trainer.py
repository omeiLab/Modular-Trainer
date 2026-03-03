from typing import Callable, Optional
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.optimizer as optim
import yaml

from src.trainer.loop import TrainerLoop
from src.trainer.runner import Runner
from src.trainer.result_builder import EpochResultBuilder
from src.trainer.result_computer import EpochResultComputer
from src.trainer.config import TrainerConfig
from src.control.controller import Controller
from src.hooks.before_epoch.base import CompositeBeforeEpochHook
from src.hooks.after_epoch.base import CompositeAfterEpochHook
from src.hooks.after_epoch.early_stop import AfterEpochEarlyStopHook
from src.hooks.after_epoch.logger import AfterEpochLoggerHook
from src.hooks.after_epoch.checkpoint import AfterEpochCheckpointHook
from src.hooks.after_step.base import CompositeAfterStepHook
from src.metrics.database import MetricDB
from src.metrics.builtin import BUILTIN_METRICS

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
        self.metric_db = MetricDB(BUILTIN_METRICS)
        self.generate_config()
        self.build()

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
            
        # metrics
        self.metrics = config.metrics
            
        # LoggerConfig
        self.log_config = config.log
        
        # EarlyStopConfig
        self.es_config = config.early_stop
        
        # CheckpointConfig
        self.ckpt_config = config.checkpoint
        
        # Other hyperparameters
        self.num_epochs = config.num_epochs
        self.task_type = config.task
        
        # validate metrics
        self.validate_metrics()
        
    def validate_metrics(self):
        """
        Check all the metrics are existed in MetricDB.
        """
        all_metrics = self.metrics + [self.es_config.metric, self.ckpt_config.metric]
        for metric in all_metrics:
            if not self.metric_db.has(metric) and "loss" not in metric:
                raise KeyError(f"The metric {metric} does not exist in MetricDB.")
        
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
                maximize=self.metric_db.get_direction(self.es_config.metric) == "max",
                min_delta=self.es_config.min_delta
            )
            after_epoch_hooks_lst.append(early_stop_hook)
        
        checkpoint_hook = AfterEpochCheckpointHook(
            model=self.model,
            metric=self.ckpt_config.metric,
            maximize=self.metric_db.get_direction(self.ckpt_config.metric) == "max",
            min_delta=self.ckpt_config.min_delta
        )
        after_epoch_hooks_lst.append(checkpoint_hook)
        after_epoch_hooks = CompositeAfterEpochHook(after_epoch_hooks_lst)

        # EpochResultComputer
        result_computer = EpochResultComputer(self.metric_db)

        # EpochResultBuilder
        result_builder = EpochResultBuilder(result_computer)
        result_builder.register("train_loss")
        result_builder.register("val_loss")
        for metric in self.metrics:
            result_builder.register(metric)

        # Runner
        runner = Runner(
            self.model, self.optimizer, self.loss_fn, self.train_loader, self.val_loader, 
            self.metrics, result_builder, result_computer, self.metric_db
        )

        # Trainer loop
        self.trainer_loop = TrainerLoop(
            before_epoch_hooks,
            after_step_hooks,
            after_epoch_hooks,
            controller,
            runner,
        )

    def run(self) -> nn.Module:
        """
        Execute the training loop for the specified number of epochs.

        Returns:
            nn.Module: Trained PyTorch model.
        """
        self.trainer_loop.run(self.num_epochs)
        return self.model
