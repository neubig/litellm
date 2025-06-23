"""
Test for float timestamp handling in ModelResponse and ModelResponseStream.

This test ensures that float timestamps (like those returned by SambaNova API)
are properly converted to integers as required by the OpenAI API specification.
"""

import pytest
import time
from litellm.types.utils import ModelResponse, ModelResponseStream


def test_model_response_float_timestamp():
    """Test that ModelResponse converts float timestamps to integers."""
    
    # Test with float timestamp (like SambaNova returns)
    float_timestamp = 1750647137.9003417
    response = ModelResponse(
        id="test-float",
        created=float_timestamp,
        model="test-model",
        object="chat.completion",
        choices=[]
    )
    
    assert isinstance(response.created, int), f"Expected int, got {type(response.created)}"
    assert response.created == 1750647137, f"Expected 1750647137, got {response.created}"


def test_model_response_stream_float_timestamp():
    """Test that ModelResponseStream converts float timestamps to integers."""
    
    # Test with float timestamp (like SambaNova returns)
    float_timestamp = 1750647137.9003417
    response = ModelResponseStream(
        id="test-stream-float",
        created=float_timestamp,
        choices=[]
    )
    
    assert isinstance(response.created, int), f"Expected int, got {type(response.created)}"
    assert response.created == 1750647137, f"Expected 1750647137, got {response.created}"


def test_model_response_integer_timestamp():
    """Test that ModelResponse still works with integer timestamps."""
    
    # Test with integer timestamp (normal case)
    int_timestamp = 1750647137
    response = ModelResponse(
        id="test-int",
        created=int_timestamp,
        model="test-model",
        object="chat.completion",
        choices=[]
    )
    
    assert isinstance(response.created, int), f"Expected int, got {type(response.created)}"
    assert response.created == 1750647137, f"Expected 1750647137, got {response.created}"


def test_model_response_none_timestamp():
    """Test that ModelResponse generates timestamp when None is provided."""
    
    # Test with None timestamp (should generate current time)
    response = ModelResponse(
        id="test-none",
        created=None,
        model="test-model",
        object="chat.completion",
        choices=[]
    )
    
    assert isinstance(response.created, int), f"Expected int, got {type(response.created)}"
    # Should be close to current time (within 10 seconds)
    current_time = int(time.time())
    assert abs(response.created - current_time) < 10, f"Generated timestamp {response.created} too far from current time {current_time}"


def test_model_response_float_no_fractional():
    """Test that ModelResponse handles float timestamps with no fractional part."""
    
    # Test with float timestamp that has no fractional part
    float_timestamp = 1750647137.0
    response = ModelResponse(
        id="test-float-no-frac",
        created=float_timestamp,
        model="test-model",
        object="chat.completion",
        choices=[]
    )
    
    assert isinstance(response.created, int), f"Expected int, got {type(response.created)}"
    assert response.created == 1750647137, f"Expected 1750647137, got {response.created}"


def test_real_time_float_timestamp():
    """Test with actual float timestamp from time.time()."""
    
    # Test with real float timestamp
    float_timestamp = time.time()
    expected_int = int(float_timestamp)
    
    response = ModelResponse(
        id="test-real-time",
        created=float_timestamp,
        model="test-model",
        object="chat.completion",
        choices=[]
    )
    
    assert isinstance(response.created, int), f"Expected int, got {type(response.created)}"
    assert response.created == expected_int, f"Expected {expected_int}, got {response.created}"