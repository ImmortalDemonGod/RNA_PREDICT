# Copied from feature_processing.py
"""
Feature processing utilities for the AtomAttentionEncoder.
"""

import warnings
from typing import Optional

import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
)


def _process_feature(
    input_feature_dict: InputFeatureDict, feature_name: str, expected_dim: int
) -> Optional[torch.Tensor]:
    """
    Process a feature from the input dictionary.

    Args:
        input_feature_dict: Dictionary containing input features
        feature_name: Name of the feature to extract
        expected_dim: Expected dimension of the feature

    Returns:
        Processed feature tensor or None if feature is invalid
    """
    if feature_name not in input_feature_dict:
        warnings.warn(f"Feature {feature_name} missing from input dictionary.", stacklevel=2)
        # Create a default tensor for the missing feature
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 1
        n_atoms = 1

        # Try to get batch_size and n_atoms from input_feature_dict
        if "ref_pos" in input_feature_dict and isinstance(input_feature_dict["ref_pos"], torch.Tensor):
            ref_pos = input_feature_dict["ref_pos"]
            if ref_pos.dim() == 2:               # [N,3]
                batch_size, n_atoms = 1, ref_pos.shape[0]
            elif ref_pos.dim() >= 3:             # [B,N,3]
                batch_size = ref_pos.shape[0]
                n_atoms = ref_pos.shape[1]
        elif "atom_to_token_idx" in input_feature_dict and isinstance(input_feature_dict["atom_to_token_idx"], torch.Tensor):
            atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
            if atom_to_token_idx.dim() == 2:     # [N,1]
                batch_size, n_atoms = 1, atom_to_token_idx.shape[0]
            elif atom_to_token_idx.dim() >= 3:   # [B,N,1]
                batch_size = atom_to_token_idx.shape[0]
                n_atoms = atom_to_token_idx.shape[1]

        # Create default tensor based on feature name
        if feature_name == "ref_pos":
            default_tensor = torch.zeros((batch_size, n_atoms, 3), device=default_device)
        elif feature_name == "ref_charge":
            default_tensor = torch.zeros((batch_size, n_atoms, 1), device=default_device)
        elif feature_name == "ref_mask":
            default_tensor = torch.ones((batch_size, n_atoms, 1), device=default_device)
        elif feature_name == "ref_element":
            default_tensor = torch.zeros((batch_size, n_atoms, expected_dim), device=default_device)
        elif feature_name == "ref_atom_name_chars":
            default_tensor = torch.zeros((batch_size, n_atoms, expected_dim), device=default_device)
        else:
            default_tensor = torch.zeros((batch_size, n_atoms, expected_dim), device=default_device)

        # Add the default tensor to the input_feature_dict
        # Use setitem method for TypedDict to avoid mypy errors
        if feature_name == "atom_to_token_idx":
            input_feature_dict["atom_to_token_idx"] = default_tensor
        elif feature_name == "ref_pos":
            input_feature_dict["ref_pos"] = default_tensor
        elif feature_name == "ref_charge":
            input_feature_dict["ref_charge"] = default_tensor
        elif feature_name == "ref_mask":
            input_feature_dict["ref_mask"] = default_tensor
        elif feature_name == "ref_element":
            input_feature_dict["ref_element"] = default_tensor
        elif feature_name == "ref_atom_name_chars":
            input_feature_dict["ref_atom_name_chars"] = default_tensor
        elif feature_name == "ref_space_uid":
            input_feature_dict["ref_space_uid"] = default_tensor
        elif feature_name == "restype":
            input_feature_dict["restype"] = default_tensor
        elif feature_name == "profile":
            input_feature_dict["profile"] = default_tensor
        elif feature_name == "deletion_mean":
            input_feature_dict["deletion_mean"] = default_tensor
        return default_tensor

    # Access the feature from the dictionary
    feature = input_feature_dict.get(feature_name)
    if feature is None:
        warnings.warn(f"Feature {feature_name} not found in input_feature_dict.", stacklevel=2)
        return None

    # Ensure feature is a tensor
    if not isinstance(feature, torch.Tensor):
        warnings.warn(f"Feature {feature_name} is not a tensor.", stacklevel=2)
        return None

    # Check if shape is already correct
    if feature.dim() > 0 and feature.shape[-1] == expected_dim:
        return feature

    # Handle specific case: expected dim is 1, but feature is missing the last dimension
    # e.g., expected [B, N, 1], got [B, N]
    if expected_dim == 1 and feature.dim() > 0 and feature.shape[-1] != 1:
        # Check if adding a dimension of size 1 makes it match expected rank implicitly
        # This is heuristic, assumes the feature provided is missing the final singleton dim
        try:
            # Check if the feature has rank >= 2 (e.g., [B, N]) before unsqueezing
            if feature.dim() >= 2:
                reshaped_feature = feature.unsqueeze(-1)
                warnings.warn(
                    f"Feature {feature_name} has shape {feature.shape}, "
                    f"but expected last dim 1. Unsqueezing to {reshaped_feature.shape}.",
                    stacklevel=2
                )
                return reshaped_feature
            else:
                warnings.warn(
                    f"Feature {feature_name} has shape {feature.shape} (rank {feature.dim()}) "
                    f"expected last dim 1, but cannot unsqueeze. Skipping this feature.",
                    stacklevel=2
                )
                return None
        except Exception as e:
            warnings.warn(
                f"Could not unsqueeze feature {feature_name} (shape {feature.shape}) "
                f"to match expected dim {expected_dim}. Error: {e}. Skipping.",
                stacklevel=2
            )
            return None

    # General reshape attempt (less likely to be correct than unsqueeze for dim 1)
    # This path should ideally not be hit if unsqueeze handles the common case
    try:
        # This might fail if numel doesn't match
        reshaped_feature = feature.view(*feature.shape[:-1], expected_dim)
        warnings.warn(
            f"Attempting general reshape for feature {feature_name} from {feature.shape} "
            f"to {reshaped_feature.shape} (expected dim {expected_dim}).",
            stacklevel=2
        )
        return reshaped_feature
    except RuntimeError:
        warnings.warn(
            f"Feature {feature_name} has wrong shape {feature.shape}, "
            f"expected last dim to be {expected_dim}. Could not reshape. Skipping.",
            stacklevel=2
        )
        return None


def adapt_tensor_dimensions(tensor: torch.Tensor, expected_dim: int) -> torch.Tensor:
    """
    Adapt the input tensor to have the expected dimension as the last dimension.
    If the tensor is missing the last singleton dimension, it will be unsqueezed.
    If the tensor has the wrong last dimension, it will be reshaped if possible.
    """
    if tensor.dim() > 0 and tensor.shape[-1] == expected_dim:
        return tensor
    if expected_dim == 1 and tensor.dim() > 0 and tensor.shape[-1] != 1:
        if tensor.dim() >= 2:
            return tensor.unsqueeze(-1)
    try:
        return tensor.view(*tensor.shape[:-1], expected_dim)
    except Exception:
        return tensor


def extract_atom_features(
    encoder: torch.nn.Module, input_feature_dict: InputFeatureDict, debug_logging: bool = False
) -> torch.Tensor:
    """
    Extract atom features from input dictionary.

    Args:
        encoder: The encoder module instance (to access input_feature and linear_no_bias_f)
        input_feature_dict: Dictionary containing atom features
        debug_logging: Whether to print debug logs

    Returns:
        Tensor of atom features
    """
    if debug_logging:
        print(f"[DEBUG][extract_atom_features] encoder.input_feature config: {encoder.input_feature}")
    features = []

    # Ensure encoder.input_feature is a dictionary before iterating
    if not hasattr(encoder, "input_feature") or not isinstance(
        encoder.input_feature, dict
    ):
        raise TypeError(
            "encoder.input_feature is not a dictionary or does not exist."
        )

    # Process each feature individually
    for feature_name, feature_dim in encoder.input_feature.items(): # type: ignore[union-attr] # Ignore previous error after check
        if debug_logging:
            print(f"[DEBUG][extract_atom_features] Processing feature: {feature_name}, expected_dim: {feature_dim}")
        processed_feature = _process_feature(
            input_feature_dict, feature_name, feature_dim
        )
        if processed_feature is not None:
            if debug_logging:
                print(f"[DEBUG][extract_atom_features] Processed {feature_name} shape: {processed_feature.shape}")
            features.append(processed_feature)

    # Check if we have any valid features
    if not features:
        if debug_logging:
            print("[DEBUG][extract_atom_features] No valid features found, creating defaults.")
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create default tensors for required features
        batch_size = 1
        n_atoms = 1

        # Try to get batch_size and n_atoms from input_feature_dict
        if "atom_to_token_idx" in input_feature_dict and isinstance(input_feature_dict["atom_to_token_idx"], torch.Tensor):
            atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
            if atom_to_token_idx.dim() == 2:     # [N,1]
                batch_size, n_atoms = 1, atom_to_token_idx.shape[0]
            elif atom_to_token_idx.dim() >= 3:   # [B,N,1]
                batch_size = atom_to_token_idx.shape[0]
                n_atoms = atom_to_token_idx.shape[1]

        # Create default features with appropriate shapes
        for feature_name, feature_dim in encoder.input_feature.items():
            if debug_logging:
                print(f"[DEBUG][extract_atom_features] Creating default for {feature_name} with dim {feature_dim}")
            if feature_name == "ref_pos":
                # Position tensor with shape [batch_size, n_atoms, 3]
                default_tensor = torch.zeros((batch_size, n_atoms, 3), device=default_device)
                features.append(default_tensor)

    # Concatenate features if available, otherwise return empty tensor
    if features:
        return torch.cat(features, dim=-1)
    else:
        # Return empty tensor with appropriate device
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros((1, 1, 1), device=default_device)
