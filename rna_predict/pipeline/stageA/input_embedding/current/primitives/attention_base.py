"""
Attention base module for neural network operations.

This module contains the core attention mechanism implementations, including the
AdaptiveLayerNorm, base attention function, and the main Attention class.

This file is now a facade that re-exports the components from their respective modules.
"""

# Import from new modular files
from .adaptive_layer_norm import AdaptiveLayerNorm
from .attention_weights import validate_attention_shapes
from .attention_weights import handle_dimension_mismatch
from .attention_weights import compute_attention_weights
from .attention_core import (
    AttentionInputs,
    ProcessQueryInputs,
    ProcessDifferentQueryInputs,
    ForwardInputs,
    AttentionConfig,
    attention,
)
from .attention_module import Attention

