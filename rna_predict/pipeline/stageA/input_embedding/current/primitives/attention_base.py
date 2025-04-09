"""
Attention base module for neural network operations.

This module contains the core attention mechanism implementations, including the
AdaptiveLayerNorm, base attention function, and the main Attention class.

This file is now a facade that re-exports the components from their respective modules.
"""

from .attention_core import (
    attention,
    AttentionInputs,
    AttentionConfig,
    ForwardInputs,
)
from .adaptive_layer_norm import AdaptiveLayerNorm
from .attention_module import Attention

__all__ = [
    'attention',
    'AttentionInputs',
    'AttentionConfig',
    'ForwardInputs',
    'AdaptiveLayerNorm',
    'Attention',
]

