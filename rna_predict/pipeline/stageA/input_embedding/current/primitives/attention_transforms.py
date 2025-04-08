"""
Attention-specific transformation module for neural network operations.

This module contains functions for transforming and manipulating tensors
specifically for attention mechanisms, including dense trunk rearrangement.

This is a compatibility layer that imports and re-exports the functions from
the new attention submodule structure.
"""

# Re-export functionality from the attention submodule
from .attention import (
    MaskCreationConfig,
    MaskSliceInfo,
    _create_masks,
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
)

# Backward compatibility: Make all imports available at the module level
__all__ = [
    "rearrange_qk_to_dense_trunk",
    "rearrange_to_dense_trunk",
    "MaskCreationConfig",
    "MaskSliceInfo",
    "_create_masks",
]
