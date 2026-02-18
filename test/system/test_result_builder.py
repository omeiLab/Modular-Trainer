import pytest
from src.trainer.result_builder import EpochResultBuilder

@pytest.fixture
def create_builder():
    builder = EpochResultBuilder()
    builder.register('loss', 'min')
    builder.register('accuracy', 'last')
    builder.record({'loss': 0.1, 'accuracy': 0.9})
    builder.record({'loss': 0.2, 'accuracy': 0.8})
    builder.record({'loss': 0.3, 'accuracy': 0.7})
    return builder
    
def test_build(create_builder):   
    builder = create_builder
    result = builder.build()
    
    assert result['loss'] == 0.1
    assert result['accuracy'] == 0.7
    
def test_reset(create_builder):
    builder = create_builder
    builder.reset()
    builder.record({'loss': 1.0, 'accuracy': 0.5})
    result = builder.build()

    assert result['loss'] == 1.0
    assert result['accuracy'] == 0.5

    
def test_invalid_reduce_method():
    builder = EpochResultBuilder()
    with pytest.raises(ValueError):
        builder.register('invalid_metric', 'invalid_reduce_method')