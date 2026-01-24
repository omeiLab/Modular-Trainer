from typing import Callable
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.optimizer as optim

from src.trainer.loop import TrainerLoop
from src.trainer.runner import Runner
from src.trainer.result_builder import EpochResultBuilder
from src.control.controller import Controller
from src.hooks.before_epoch.base import CompositeBeforeEpochHook
from src.hooks.after_epoch.base import CompositeAfterEpochHook
from src.hooks.after_epoch.early_stop import AfterEpochEarlyStopHook
from src.hooks.after_epoch.logger import AfterEpochLoggerHook
from src.hooks.after_epoch.checkpoint import AfterEpochCheckpointHook
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

    def from_config(self, config_path: str) -> None:
        """
        Load hyperparameters and hook settings from a YAML/JSON config.

        This method can be used to set:
        - num_epochs
        - hook parameters (early stopping, checkpointing, etc.)
        - metrics to track
        - any other training hyperparameters

        Args:
            config_path (str): Path to config file.
        """
        # TODO: parse config and store internally
        pass

    def build(self) -> None:
        """
        Construct hooks, result builder, runner, and training loop.

        After calling this method, `self.trainer_loop` is ready to run.
        """
        # Controller
        controller = Controller()

        # Before-epoch hooks
        before_epoch_hooks = CompositeBeforeEpochHook([])

        # After-step hooks
        after_step_hooks = CompositeAfterStepHook([])

        # After-epoch hooks
        logger_hook = AfterEpochLoggerHook()
        early_stop_hook = AfterEpochEarlyStopHook(
            controller=controller,
            patience=2,  # stop if val_loss doesn't improve for 2 epochs
            metric="val_loss",
            maximize=False,
            min_delta=0.0001
        )
        checkpoint_hook = AfterEpochCheckpointHook(
            model=self.model,
            metric="val_loss",
            maximize=False,
            min_delta=0.0001
        )
        after_epoch_hooks = CompositeAfterEpochHook([logger_hook, early_stop_hook, checkpoint_hook])

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
        self.trainer_loop.run(num_epochs=10)
        return self.model
