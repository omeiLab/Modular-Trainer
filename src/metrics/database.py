from dataclasses import dataclass
from typing import Literal, Optional, Callable, Dict

OptimizeDirection = Literal["min", "max"]
Reducer = Literal['avg', 'min', 'max', 'last', 'sum']

@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: OptimizeDirection
    description: Optional[str] = None

@dataclass(frozen=True)
class Metric:
    spec: MetricSpec
    fn: Callable[..., float]
    reduce: Reducer
    
class MetricDB:
    def __init__(self, metrics: Dict[str, Metric]):
        self._metrics = metrics

    def get(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError:
            raise KeyError(f"Unknown metric: {name}")

    def has(self, name: str) -> bool:
        return name in self._metrics

    def list(self) -> list[MetricSpec]:
        return [m.spec for m in self._metrics.values()]
