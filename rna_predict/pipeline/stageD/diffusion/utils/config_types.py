"""
Configuration type definitions for Stage D diffusion.

This module provides type definitions for configuration objects used in the Stage D diffusion process.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from omegaconf import DictConfig

import torch


@dataclass
class DiffusionConfig:
    """Configuration for Stage D diffusion."""

    # Required parameters
    partial_coords: torch.Tensor
    trunk_embeddings: Dict[str, torch.Tensor]
    diffusion_config: DictConfig

    # Optional parameters with defaults
    mode: str = "inference"
    device: str = "cpu"
    input_features: Optional[Dict[str, Any]] = None
    debug_logging: bool = False
    sequence: Optional[str] = None
    test_residues_per_batch: int = 25  # Added parameter for inference mode
    cfg: Optional[DictConfig] = None
    atom_metadata: Optional[Dict[str, Any]] = None  # Added parameter for atom metadata

    # Feature parameters required by _validate_feature_config
    ref_element_size: int = 128
    ref_atom_name_chars_size: int = 256
    profile_size: int = 32

    # Internal state (not set by user)
    trunk_embeddings_internal: Dict[str, torch.Tensor] = field(default_factory=dict, init=False)
