import pytest
from src.trainer.result_builder import EpochResultBuilder
from src.trainer.result_computer import EpochResultComputer
from src.metrics.database import MetricDB

EPS = 1e-8

# dummy metricDB
DUMMY_METRICS = {}
db = MetricDB(DUMMY_METRICS)

# dummy computer
counter = 0.0
computer = EpochResultComputer(db)

# override
def compute(metric_names):
    global counter
    counter += 0.1
    return {name: counter for name in metric_names}
computer.compute_all = compute

@pytest.fixture
def generate_builder():
    builder = EpochResultBuilder(computer)
    builder.register("train_loss")
    builder.register("accuracy")
    builder.register("train_loss")
    return builder

def test_register(generate_builder):
    builder = generate_builder
    assert builder._metrics == ["train_loss", "accuracy"]

def test_build(generate_builder):
    builder = generate_builder
    result = builder.build()
    result = builder.build()
    result = builder.build()
    assert result['train_loss'] == pytest.approx(0.3, abs=EPS)
    assert result['accuracy'] == pytest.approx(0.3, abs=EPS)
    assert len(builder._history) == 3
    assert builder._history[0]['train_loss'] == pytest.approx(0.1, abs=EPS)
    assert builder._history[1]['accuracy'] == pytest.approx(0.2, abs=EPS)