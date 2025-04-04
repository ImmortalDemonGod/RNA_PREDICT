"""
Primitives module for neural network operations.

This package contains fundamental building blocks and operations for neural networks,
particularly focusing on attention mechanisms and tensor transformations used in the
RNA prediction pipeline.

Note: This module has been refactored from a single file into multiple submodules
for better maintainability and code organization.
"""

# Direct import from external dependency (needed by dependents)
# Using PyTorch's native LayerNorm for macOS compatibility
from torch.nn import LayerNorm

# From atom_pair_transforms.py
from .atom_pair_transforms import (
    broadcast_token_to_local_atom_pair,
    gather_pair_embedding_in_dense_trunk,
)

# From attention.dense_trunk.py
from .attention.dense_trunk import (
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
)

# From attention_base.py
from .attention_base import (
    AdaptiveLayerNorm,
    Attention,
    _attention,
)

# From attention_utils.py
from .attention_utils import (
    _local_attention,
    create_local_attn_bias,
    optimized_concat_split,
)

# From linear_primitives.py
from .linear_primitives import (
    BiasInitLinear,
    LinearNoBias,
    Transition,
)

__all__ = [
    # External imports
    "LayerNorm",
    # Linear primitives
    "LinearNoBias",
    "BiasInitLinear",
    "Transition",
    # Attention base
    "AdaptiveLayerNorm",
    "Attention",
    "_attention",
    # Attention utils
    "create_local_attn_bias",
    "optimized_concat_split",
    "_local_attention",
    # Data transforms
    "rearrange_qk_to_dense_trunk",
    "rearrange_to_dense_trunk",
    "gather_pair_embedding_in_dense_trunk",
    "broadcast_token_to_local_atom_pair",
    "DenseTrunkConfig",
]
