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
    Retrieves the embedding dimension from the diffusion configuration.
    
    This function returns the embedding dimension by looking up the specified key in the
    given configuration dictionary. It first checks the top-level of the dictionary and,
    if the key is not found, then attempts to retrieve it from the nested "conditioning"
    sub-dictionary. If the key remains absent, the provided default value is returned.
    
    Args:
        diffusion_config: A dictionary containing diffusion configuration settings.
        key: The configuration key corresponding to the embedding dimension.
        default_value: The value to return if the key is not found in the configuration.
        
    Returns:
        The embedding dimension as an integer.
    """
    conditioning_config = diffusion_config.get("conditioning", {})
    return diffusion_config.get(key, conditioning_config.get(key, default_value))


def create_fallback_input_features(
    partial_coords: torch.Tensor, diffusion_config: Dict[str, Any], device: str
) -> Dict[str, Any]:
    """
    Creates fallback input features for the diffusion process when no features are provided.
    
    This function generates a dictionary of tensors representing default input features required 
    by the diffusion process. The number of atoms is determined from the second dimension of 
    the provided `partial_coords` tensor. Each feature tensor—such as indices, positions, charges, 
    and masks—is initialized with appropriate dimensions and moved to the specified device. The 
    feature tensor for 'sing' uses a dimension defined by the "c_s_inputs" key in `diffusion_config`, 
    defaulting to 449 if the key is absent.
    
    Args:
        partial_coords: A tensor of shape [B, N_atom, 3] containing partial atomic coordinates.
        diffusion_config: A dictionary with diffusion configuration settings. The "c_s_inputs" key 
            (if present) sets the dimensionality for the 'sing' feature.
        device: A string specifying the device (e.g., "cpu" or "cuda") on which to allocate the tensors.
    
    Returns:
        A dictionary mapping feature names (e.g., "atom_to_token_idx", "ref_pos", "ref_charge") to their 
        corresponding tensors, serving as fallback input features for the diffusion process.
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
