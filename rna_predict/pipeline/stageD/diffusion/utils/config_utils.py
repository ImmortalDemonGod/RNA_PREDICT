"""
Configuration utilities for Stage D diffusion.

This module provides functions for handling configuration in the Stage D diffusion process.
"""

import torch
from typing import Any, Dict, Optional


def get_embedding_dimension(
    diffusion_config: Dict[str, Any], key: str, default_value: int
) -> int:
    """
    Get embedding dimension from diffusion config with fallback.

    Args:
        diffusion_config: Configuration dictionary
        key: Key to look for in config
        default_value: Default value if key not found

    Returns:
        Embedding dimension
    """
    conditioning_config = diffusion_config.get("conditioning", {})
    return diffusion_config.get(key, conditioning_config.get(key, default_value))


def create_fallback_input_features(
    partial_coords: torch.Tensor, diffusion_config: Dict[str, Any], device: str
) -> Dict[str, Any]:
    """
    Create fallback input features when none are provided.

    Args:
        partial_coords: Partial coordinates tensor [B, N_atom, 3]
        diffusion_config: Configuration for diffusion components
        device: Device to run on

    Returns:
        Dict of fallback input features
    """
    N = partial_coords.shape[1]
    return {
        "atom_to_token_idx": torch.arange(N, device=device).unsqueeze(0),
        "ref_pos": partial_coords.to(device),
        "ref_space_uid": torch.arange(N, device=device).unsqueeze(0),
        "ref_charge": torch.zeros(1, N, 1, device=device),
        "ref_element": torch.zeros(1, N, 128, device=device),
        "ref_atom_name_chars": torch.zeros(1, N, 256, device=device),
        "ref_mask": torch.ones(1, N, 1, device=device),
        "restype": torch.zeros(1, N, 32, device=device),
        "profile": torch.zeros(1, N, 32, device=device),
        "deletion_mean": torch.zeros(1, N, 1, device=device),
        "sing": torch.zeros(
            1, N, diffusion_config.get("c_s_inputs", 449), device=device
        ),
    }
