"""
Primitives module for neural network operations.

This module has been refactored into multiple submodules for better maintainability.
This file is now a thin wrapper that imports from the submodules to maintain backward
compatibility with existing code.

The module structure is:
- primitives/linear_primitives.py: Linear layer implementations and transformations
- primitives/attention_base.py: Core attention mechanisms
- primitives/attention_utils.py: Specialized attention utilities
- primitives/data_transforms.py: Data transformation functions
"""

# Direct import from external dependency
from protenix.openfold_local.model.primitives import LayerNorm

from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import (
    AdaptiveLayerNorm,
    Attention,
    _attention,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils import (
    _local_attention,
    create_local_attn_bias,
    optimized_concat_split,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives.data_transforms import (
    broadcast_token_to_local_atom_pair,
    gather_pair_embedding_in_dense_trunk,
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives.linear_primitives import (
    BiasInitLinear,
    LinearNoBias,
    Transition,
)

# Make all imported names available in this module
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
