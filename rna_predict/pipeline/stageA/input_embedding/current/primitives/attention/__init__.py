"""
Attention modules package for neural network operations.

This package contains modules for attention mechanisms, including:
- Trunk operations (dense_trunk.py)
- Mask operations (mask_operations.py)
"""

from .dense_trunk import (
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
)
from .mask_operations import (
    MaskCreationConfig,
    MaskSliceInfo,
    _create_masks,
)

__all__ = [
    # Dense trunk functions
    "rearrange_qk_to_dense_trunk",
    "rearrange_to_dense_trunk",
    
    # Mask operations
    "MaskCreationConfig",
    "MaskSliceInfo",
    "_create_masks",
] 