"""
Configuration utilities for Stage D diffusion.

This module provides functions for handling configuration in the Stage D diffusion process.
"""

import torch
from typing import Any, Dict
from .config_types import DiffusionConfig


def get_embedding_dimension(
    diffusion_config: DiffusionConfig, key: str, default_value: int
) -> int:
    """
    Get embedding dimension from diffusion config with fallback.

    Args:
        diffusion_config: DiffusionConfig structured config
        key: Key to look for in config.diffusion_config
        default_value: Default value if key not found

    Returns:
        Embedding dimension
    """
    conditioning_config = diffusion_config.diffusion_config.get("conditioning", {})
    return diffusion_config.diffusion_config.get(key, conditioning_config.get(key, default_value))


def create_fallback_input_features(
    partial_coords: torch.Tensor, diffusion_config: DiffusionConfig, device: str
) -> Dict[str, Any]:
    """
    Create fallback input features when none are provided.

    Args:
        partial_coords: Partial coordinates tensor [B, N_atom, 3]
        diffusion_config: DiffusionConfig structured config
        device: Device to run on

    Returns:
        Dict of fallback input features
    """
    N = partial_coords.shape[1]
    # Get the dimension for s_inputs
    c_s_inputs_dim = diffusion_config.diffusion_config.get("c_s_inputs", None)
    if c_s_inputs_dim is None:
        raise ValueError("Missing config value for 'c_s_inputs'")

    # Get the dimension for ref_element
    ref_element_dim = diffusion_config.diffusion_config.get("ref_element_dim", None)
    if ref_element_dim is None:
        raise ValueError("Missing config value for 'ref_element_dim'")

    # Get the dimension for ref_atom_name_chars
    ref_atom_name_chars_dim = diffusion_config.diffusion_config.get("ref_atom_name_chars_dim", None)
    if ref_atom_name_chars_dim is None:
        raise ValueError("Missing config value for 'ref_atom_name_chars_dim'")

    # Get the dimension for restype and profile
    restype_dim = diffusion_config.diffusion_config.get("restype_dim", None)
    if restype_dim is None:
        raise ValueError("Missing config value for 'restype_dim'")

    return {
        "atom_to_token_idx": torch.arange(N, device=device).unsqueeze(0),
        "ref_pos": partial_coords.to(device),
        "ref_space_uid": torch.arange(N, device=device).unsqueeze(0),
        "ref_charge": torch.zeros(1, N, 1, device=device),
        "ref_element": torch.zeros(1, N, ref_element_dim, device=device),
        "ref_atom_name_chars": torch.zeros(1, N, ref_atom_name_chars_dim, device=device),
        "ref_mask": torch.ones(1, N, 1, device=device),
        "restype": torch.zeros(1, N, restype_dim, device=device),
        "profile": torch.zeros(1, N, restype_dim, device=device),
        "deletion_mean": torch.zeros(1, N, 1, device=device),
        "sing": torch.zeros(1, N, c_s_inputs_dim, device=device),
        # Add s_inputs as well to ensure it's available
        "s_inputs": torch.zeros(1, N, c_s_inputs_dim, device=device),
    }
