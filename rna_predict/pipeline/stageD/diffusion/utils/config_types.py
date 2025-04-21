"""
Configuration type definitions for Stage D diffusion.

This module provides type definitions for configuration objects used in the Stage D diffusion process.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class DiffusionConfig:
    """Configuration for Stage D diffusion."""

    # Required parameters
    partial_coords: torch.Tensor
    trunk_embeddings: Dict[str, torch.Tensor]
    diffusion_config: Dict[str, Any]

    # Optional parameters with defaults
    mode: str = "inference"
    device: str = "cpu"
    input_features: Optional[Dict[str, Any]] = None
    debug_logging: bool = False
    sequence: Optional[str] = None

    # Internal state (not set by user)
    trunk_embeddings_internal: Dict[str, torch.Tensor] = field(default_factory=dict, init=False)
