import pytest
import torch
import math
from src.metrics.builtin import mean_squared_error, root_mean_squared_error, mean_absolute_error, r_squared 

@pytest.fixture
def generate_data():
    true_tensor = torch.tensor([ 1.0, 2.0, 3.0, 4.0, 5.0])
    pred_tensor = torch.tensor([-1.0, 2.0, 3.0, 6.0, 6.0])
    return true_tensor, pred_tensor

@pytest.fixture
def generate_identity_data():
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    return tensor, tensor

def test_mean_squared_error_positive(generate_data):
    true_tensor, pred_tensor = generate_data
    val = mean_squared_error(preds=pred_tensor, targets=true_tensor)

    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert val >= 0.0


def test_mean_squared_error_identity(generate_identity_data):
    true_tensor, pred_tensor = generate_identity_data
    val = mean_squared_error(preds=pred_tensor, targets=true_tensor)

    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert val == 0.0
    
def test_root_mean_squared_error_positive(generate_data):
    true_tensor, pred_tensor = generate_data
    val = root_mean_squared_error(preds=pred_tensor, targets=true_tensor)

    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert val >= 0.0


def test_root_mean_squared_error_identity(generate_identity_data):
    true_tensor, pred_tensor = generate_identity_data
    val = root_mean_squared_error(preds=pred_tensor, targets=true_tensor)

    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert val == 0.0

def test_mean_absolute_error_positive(generate_data):
    true_tensor, pred_tensor = generate_data
    val = mean_absolute_error(preds=pred_tensor, targets=true_tensor)

    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert val >= 0.0


def test_mean_absolute_error_identity(generate_identity_data):
    true_tensor, pred_tensor = generate_identity_data
    val = mean_absolute_error(preds=pred_tensor, targets=true_tensor)

    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert val == 0.0
    
def test_r2(generate_data):
    true_tensor, pred_tensor = generate_data
    val = r_squared(preds=pred_tensor, targets=true_tensor)
    
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert val <= 1.0
    
def test_r2_identity(generate_identity_data):
    true_tensor, pred_tensor = generate_identity_data
    val = r_squared(preds=pred_tensor, targets=true_tensor)
    
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert not math.isinf(val)
    assert val == 1.0