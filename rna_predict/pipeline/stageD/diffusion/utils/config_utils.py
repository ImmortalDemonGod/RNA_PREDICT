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
    Retrieves the embedding dimension from a diffusion configuration.
    
    This function looks for the specified key in the provided configuration dictionary.
    It first checks the top level and then within the nested "conditioning" dictionary.
    If the key is missing in both, the given default value is returned.
    
    Args:
        diffusion_config: A dictionary of diffusion configuration settings.
        key: The key to locate the embedding dimension.
        default_value: The fallback value if the key is not present.
    
    Returns:
        The embedding dimension as an integer.
    """
    conditioning_config = diffusion_config.get("conditioning", {})
    return diffusion_config.get(key, conditioning_config.get(key, default_value))


def create_fallback_input_features(
    partial_coords: torch.Tensor, diffusion_config: Dict[str, Any], device: str
) -> Dict[str, Any]:
    """
    Create a dictionary of fallback input features for the diffusion process.
    
    This function generates a collection of tensors to serve as fallback input features when
    no explicit features are provided. The tensors are configured based on the number of atoms,
    derived from the second dimension of the provided partial coordinates tensor. It creates
    tensors for atom-to-token indexing, reference positions, unique identifiers, charges, element
    representations, atom name characters, masks, residue types, profiles, deletion means, and a
    'sing' feature whose last dimension is determined by the "c_s_inputs" key in the diffusion
    configuration (defaulting to 449 if not specified). All tensors are allocated on the given device.
    
    Args:
        partial_coords: Tensor with shape [B, N_atom, 3] representing partial atomic coordinates.
        diffusion_config: Dictionary containing configuration settings for diffusion components.
        device: The device (e.g., 'cpu' or 'cuda') on which tensors are allocated.
    
    Returns:
        A dictionary mapping feature names to tensors initialized based on the partial coordinates and
        diffusion configuration.
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
