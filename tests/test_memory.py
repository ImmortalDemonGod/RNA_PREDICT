"""
Simple test file to verify memory profiling.
"""

import numpy as np
import gc

def test_memory_allocation():
    """Test that allocates memory to verify profiling."""
    # Force garbage collection before test
    gc.collect()
    
    # Create a large array to see memory usage
    arr = np.zeros((5000, 5000), dtype=np.float64)
    assert arr.shape == (5000, 5000)
    
    # Keep the array in memory
    arr[0, 0] = 1.0

def test_small_allocation():
    """Test with minimal memory allocation for comparison."""
    # Force garbage collection before test
    gc.collect()
    
    arr = np.zeros((10, 10))
    assert arr.shape == (10, 10) 