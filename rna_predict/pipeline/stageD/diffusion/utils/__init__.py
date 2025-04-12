"""
Utility functions for Stage D diffusion.

This package provides utility functions for the Stage D diffusion process.
"""

from .config_utils import get_embedding_dimension, create_fallback_input_features
from .embedding_utils import ensure_s_inputs, ensure_z_trunk
from .tensor_utils import normalize_tensor_dimensions
from .config_types import DiffusionConfig

__all__ = [
    "get_embedding_dimension",
    "create_fallback_input_features",
    "ensure_s_inputs",
    "ensure_z_trunk",
    "normalize_tensor_dimensions",
    "DiffusionConfig",
]
