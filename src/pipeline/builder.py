from typing import Dict, Mapping, Any
import numpy as np

from src.trainer.loop import TrainerLoop
from src.trainer.runner import Runner
from src.trainer.result_builder import EpochResultBuilder
from src.control.controller import Controller
from src.hooks.before_epoch.base import CompositeBeforeEpochHook
from src.hooks.after_epoch.base import CompositeAfterEpochHook, AfterEpochHook
from src.hooks.after_epoch.early_stop import AfterEpochEarlyStopHook
from src.hooks.after_epoch.logger import AfterEpochLoggerHook
from src.hooks.after_epoch.checkpoint import AfterEpochCheckpointHook
from src.hooks.after_step.base import CompositeAfterStepHook

def build_trainer(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    num_epochs,
):
    """
    Build and run a training pipeline with predefined training loop, hooks,
    and control flow, then return the trained model.

    This function serves as a high-level entry point that wires together
    the core training components (runner, loop, hooks, controller) with
    reasonable default behaviors:
      - Logging after each epoch
      - Early stopping based on validation metrics
      - Checkpoint saving via a user-replaceable callback

    The function will immediately execute the training loop for the specified
    number of epochs. Early stopping may terminate training before reaching
    `num_epochs`.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained. It is assumed to be fully initialized
        (and moved to the desired device, if applicable) before calling
        this function.

    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.

    loss_fn : Callable
        Loss function used during training and validation steps.

    train_loader : DataLoader
        DataLoader providing training batches.

    val_loader : DataLoader
        DataLoader providing validation batches. Used by validation
        steps, logging, early stopping, and checkpointing.

    num_epochs : int
        Maximum number of epochs to run the training loop. Actual training
        may stop earlier if an early stopping condition is met.

    Returns
    -------
    model : torch.nn.Module
        The trained model after completion of the training loop
        (or early stopping).

    Notes
    -----
    - This function is intentionally opinionated and designed as a
      "batteries-included" entry point.
    - Custom behaviors (e.g., metrics, checkpoint saving logic, hooks)
      are expected to be extended or replaced in future iterations.
    - Checkpoint saving currently uses a dummy `save_fn` placeholder.
    """

    # dummy save_fn for checkpoint
    # modify later
    def save_fn(epoch, results):
        print(f"[Checkpoint] Saving checkpoint at epoch {epoch}")
    
    # controller
    controller = Controller()
    
    # before-epoch hooks
    before_epoch_hooks = CompositeBeforeEpochHook([])
    
    # after-step hooks
    after_step_hooks = CompositeAfterStepHook([])
    
    # after-epoch hooks
    logger_hook = AfterEpochLoggerHook()
    early_stop_hook = AfterEpochEarlyStopHook(
        controller=controller,
        patience=2,  # stop if val_loss doesn't improve for 2 epochs
        metric="val_loss",
        maximize=False,
        min_delta=0.0001
    )
    checkpoint_hook = AfterEpochCheckpointHook(
        metric="val_loss",
        save_fn=save_fn,
        maximize=False,
        min_delta=0.0001
    )
    after_epoch_hooks = CompositeAfterEpochHook([logger_hook, early_stop_hook, checkpoint_hook])
    
    # EpochResultBuilder
    result_builder = EpochResultBuilder()
    
    # Runner
    runner = Runner(model, optimizer, loss_fn, train_loader, val_loader, result_builder)
    
    # trainer loop & run
    trainer_loop = TrainerLoop(before_epoch_hooks, after_step_hooks, after_epoch_hooks, controller, runner, result_builder)
    trainer_loop.run(num_epochs)
    
    return model