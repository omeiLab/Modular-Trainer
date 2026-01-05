from typing import Dict, Mapping, Any
import numpy as np

class EpochResultBuilder:
    """
    Builder for collecting and aggregating step-level metrics
    into epoch-level results.

    This class owns all epoch-level metric state. It is responsible for:
    - Recording step-level metric dictionaries.
    - Maintaining per-metric buffers across steps.
    - Resetting buffers at the beginning of each epoch.
    - Producing an epoch-level snapshot for hooks (e.g. logging,
      early stopping, checkpointing).

    Design principles:
    - Metrics are dynamically discovered on first record().
    - TrainerLoop should not know about metric storage or aggregation.
    - This class encapsulates all metric-related state and logic.
    """

    def __init__(self):
        """
        Initialize an empty EpochResultBuilder.

        Internal storage maps metric names to a list of recorded values
        within the current epoch.
        """
        self._storage = {}
        self._reducers = {}
        self._valid_reducers = ['avg', 'min', 'max', 'last', 'sum']

    def reset(self) -> None:
        """
        Reset all stored metric values for a new epoch.

        This clears the per-metric value lists while preserving
        the metric keys themselves. Metric schema is therefore
        stable across epochs.
        """
        for v in self._storage.values():
            v.clear()

    def register(self, name: str, reduce: str) -> None:
        """
        Register a metric name explicitly.

        Registration is optional. Metrics are automatically registered
        on first record() if not present. This method exists to support
        future use cases such as strict schema validation or custom
        aggregation policies.

        Args:
            name (str): Name of the metric to register.
            reduce (str): Method of aggregation for this metric, should be one of 'avg', 'last', 'min', 'max', 'sum'
        """
        if reduce not in self._valid_reducers:
            raise ValueError(f"Unsupported aggregation method: {reduce}, valid methods are {self._valid_reducers}")
        if name not in self._storage:
            self._storage[name] = []
        self._reducers[name] = reduce

    def record(self, step_result: Dict[str, Any]) -> None:
        """
        Record step-level metrics.

        Each call appends the provided metric values to the internal
        per-metric buffers. Metrics are dynamically created on first use.

        Args:
            step_result (Dict[str, float]): Mapping from metric name
                to numeric value for a single training step.

        Raises:
            ValueError: If a metric name is not a string or a value
                is not numeric.
        """
        # validation & registration
        for k, v in step_result.items():
            if not isinstance(k, str):
                raise ValueError("Metric name must be str")
            if not isinstance(v, (int, float)):
                raise ValueError(f"Metric {k} must be numeric")

            if k not in self._storage:
                self._storage[k] = []
                self._reducers[k] = 'avg'

        # commit values
        for k, v in step_result.items():
            self._storage[k].append(float(v))
            
    def _aggregation(self, key: str) -> float:
        """Helper function to apply aggregation method to a list of values"""
        lst = self._storage[key]
        method = self._reducers[key]
        
        if method == 'avg':
            return np.mean(lst)
        if method == 'last':
            return lst[-1]
        if method =='min':
            return np.min(lst)
        if method =='max':
            return np.max(lst)
        if method =='sum':
            return np.sum(lst)
        
        raise ValueError(f"Unsupported aggregation method: {method}")

    def build(self) -> Mapping[str, Any]:
        """
        Build and return the epoch-level results.

        Returns:
            Dict[str, Any]: A snapshot of collected metrics for the
            current epoch. Each metric maps to its list of recorded
            values. Aggregation (e.g. mean, last) can be applied
            by downstream hooks or extended in this builder.
        """
        epoch_results = {}
        for metric, _ in self._reducers.items():
            epoch_results[metric] = self._aggregation(metric)
        return epoch_results