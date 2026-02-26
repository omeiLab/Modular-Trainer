import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, List

from src.trainer.result_builder import EpochResultBuilder
from src.trainer.result_computer import EpochResultComputer
from src.metrics.database import MetricDB

class Runner:
    """
    A simple Runner to handle one epoch of training and validation.

    Responsibilities:
    - Move model to the correct device at initialization.
    - Run training and validation loops for one epoch.
    - Record step/epoch metrics into an EpochResultBuilder.
    - Return aggregated epoch-level metrics.

    Designed for general regression, classification, and NLP tasks.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metrics: List[str],
        result_builder: EpochResultBuilder,
        result_computer: EpochResultComputer,
        metric_db: MetricDB
    ):
        """
        Initialize the Runner.

        Args:
            model (torch.nn.Module): The model to train/validate.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            loss_fn (callable): Loss function.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader): Validation data loader.
            result_builder (EpochResultBuilder): Builder to record metrics.
            result_computer (EpochResultComputer): Computer to save step results and calculate metrics.
            metric_db (MetricDB): Database of metrics to record.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics
        self.result_builder = result_builder
        self.result_computer = result_computer
        self.metric_db = metric_db

        # Move model to device
        self.device = next(model.parameters()).device
        self.model.to(self.device)

    def _train_one_epoch(self) -> None:
        """
        Run one epoch of training.

        Returns:
            dict: Training metrics (currently 'train_loss').
        """
        self.model.train()
        for batch in self.train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.result_computer.record_loss("train_loss", loss.item())

    def _validate_one_epoch(self) -> None:
        """
        Run one epoch of validation.

        Returns:
            dict: Validation metrics (currently 'val_loss').
        """
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)
                
                # record metrics
                self.result_computer.record_loss("val_loss", loss.item())
                self.result_computer.record_step(preds=preds, targets=targets)

    def run_one_epoch(self):
        """
        Run one epoch of train + validation, recording metrics to the result builder.

        Returns:
            dict: Aggregated epoch-level metrics (train + val).
        """
        self.result_computer.reset()  # Clear previous epoch's step data
        train_metrics = self._train_one_epoch()
        val_metrics = self._validate_one_epoch()
        return self.result_builder.build()
