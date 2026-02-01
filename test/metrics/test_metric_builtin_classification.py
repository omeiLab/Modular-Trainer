import pytest
import torch
import math

from src.metrics.builtin import accuracy, precision, recall, f1_score

@pytest.fixture
def generate_data():
    true  = torch.tensor([1, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    preds = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    return true, preds

def test_accuracy(generate_data):
    true_tensor, pred_tensor = generate_data
    val = accuracy(preds=pred_tensor, targets=true_tensor)
    
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert 0.0 <= val <= 1.0

def test_precision(generate_data):
    true_tensor, pred_tensor = generate_data
    val = precision(preds=pred_tensor, targets=true_tensor)
    
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert 0.0 <= val <= 1.0
    
def test_precision_zero_division(generate_data):
    true  = torch.tensor([1, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    preds = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    val = precision(preds=preds, targets=true)
    
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert 0.0 <= val <= 1.0

def test_recall(generate_data):
    true_tensor, pred_tensor = generate_data
    val = recall(preds=pred_tensor, targets=true_tensor)
    
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert 0.0 <= val <= 1.0
    
def test_recall_zero_division(generate_data):
    true  = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    preds = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    val = recall(preds=preds, targets=true)
    
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert 0.0 <= val <= 1.0
    
def test_f1_score(generate_data):
    true_tensor, pred_tensor = generate_data
    val = f1_score(preds=pred_tensor, targets=true_tensor)
    
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert 0.0 <= val <= 1.0
