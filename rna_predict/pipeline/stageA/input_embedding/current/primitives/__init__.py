"""
Primitives module for neural network operations.

This package contains fundamental building blocks and operations for neural networks,
particularly focusing on attention mechanisms and tensor transformations used in the
RNA prediction pipeline.

Note: This module has been refactored from a single file into multiple submodules
for better maintainability and code organization.
"""

# Direct import from external dependency (needed by dependents)
from protenix.openfold_local.model.primitives import LayerNorm

# From linear_primitives.py
from .linear_primitives import (
    LinearNoBias,
    BiasInitLinear,
    Transition,
)

# From attention_base.py
from .attention_base import (
    AdaptiveLayerNorm,
    Attention,
    _attention,
)

# From attention_utils.py
from .attention_utils import (
    create_local_attn_bias,
    optimized_concat_split,
    _local_attention,
)

# From data_transforms.py
from .data_transforms import (
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
    gather_pair_embedding_in_dense_trunk,
    broadcast_token_to_local_atom_pair,
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
] 