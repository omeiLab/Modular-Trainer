from typing import Dict, List

from src.trainer.result_computer import EpochResultComputer


class EpochResultBuilder:
    """
    Epoch-level result manager.

    This class coordinates metric computation and maintains
    epoch-level history across the entire training process.

    Responsibilities:
    -----------------
    1. Register metric names defined by Trainer configuration.
    2. Request metric computation from EpochResultComputer.
    3. Store computed epoch-level results into history.
    4. Return the current epoch result to the Runner.

    Design principles:
    ------------------
    - Builder does NOT compute metrics itself.
    - Builder does NOT manage step-level data.
    - Metric computation is delegated to EpochResultComputer.
    - History storage is encapsulated inside this class.

    Lifecycle:
    ----------
    During Trainer setup:
        builder.register("train_loss")
        builder.register("accuracy")
        ...

    During training:
        epoch_result = builder.build()

    Future extensions:
    ------------------
    This class may later provide:
        - best(metric_name)
        - last(metric_name)
        - history(metric_name)
        - export utilities (CSV / JSON)
    """

    def __init__(self, computer: EpochResultComputer):
        """
        Initialize the EpochResultBuilder.

        Args:
            computer (EpochResultComputer):
                The computation engine responsible for
                epoch-level metric calculation.
        """
        self.computer = computer
        self._metrics: List[str] = []
        self._history: List[Dict[str, float]] = []

    def register(self, name: str) -> None:
        """
        Register a metric name to be computed at epoch end.

        Duplicate registrations are ignored.

        Args:
            name (str):
                Name of the metric (must exist in MetricDB
                or be a recorded loss name).
        """
        if name not in self._metrics:
            self._metrics.append(name)

    def _record(self, epoch_result: Dict[str, float]) -> None:
        """
        Record the computed epoch result into history.

        Args:
            epoch_result (Dict[str, float]):
                Mapping from metric name to epoch-level value.
        """
        self._history.append(epoch_result)

    def build(self) -> Dict[str, float]:
        """
        Compute and record the current epoch metrics.

        This method:
        1. Requests metric computation from EpochResultComputer.
        2. Stores the result internally.
        3. Returns the epoch result to the caller (e.g., Runner).

        Returns:
            Dict[str, float]:
                Mapping from metric name to computed epoch-level value.
        """
        epoch_result = self.computer.compute_all(self._metrics)
        self._record(epoch_result)
        return epoch_result