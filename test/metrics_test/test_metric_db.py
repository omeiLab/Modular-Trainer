import pytest
import torch
import random

from src.metrics.database import Metric, MetricSpec, MetricDB

def positive():
    '''
    Generate a random positive floating number
    '''
    return random.random()

def negative():
    '''
    Generate a random negative floating number
    '''
    return -random.random()

DUMMY_METRICS = {
    "pos": Metric(
        spec=MetricSpec(
            name="pos",
            direction="min",
            input="logits",
            description="Generate random positive floating number",
        ),
        fn = positive,
    ),
    "neg": Metric(
        spec=MetricSpec(
            name="neg",
            direction="max",
            input="probability",
            description="Generate a random negative floating number",
        ),
        fn = negative,
    ),
}

@pytest.fixture
def metric_db():
    return MetricDB(DUMMY_METRICS)

def test_get(metric_db):
    pos_metric = metric_db.get("pos")
    neg_metric = metric_db.get("neg")
    assert pos_metric.spec.name == "pos"
    assert pos_metric.spec.direction == "min"
    assert pos_metric.spec.input == "logits"
    assert pos_metric.spec.description == "Generate random positive floating number"
    for _ in range(10):
        assert pos_metric.fn() >= 0
        assert neg_metric.fn() <= 0
    
def test_get_not_exist(metric_db):
    with pytest.raises(KeyError):
        metric_db.get("unknown_metric")
        
def test_has(metric_db):
    assert metric_db.has("pos") == True
    assert metric_db.has("neg") == True
    assert metric_db.has("zero") == False
    
def test_list(metric_db):
    lst = metric_db.list()
    names = [spec.name for spec in lst]
    directions = [spec.direction for spec in lst]
    assert "pos" in names
    assert "neg" in names
    assert "min" in directions
    assert "max" in directions
    
def test_empty():
    empty_db = MetricDB({})
    assert empty_db.list() == []
    assert empty_db.has("any") is False
    with pytest.raises(KeyError):
        empty_db.get("any")