"""
Feature processing utilities for the AtomAttentionEncoder.
"""
import warnings
from typing import Optional

import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
    safe_tensor_access,
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
        warnings.warn(f"Feature {feature_name} missing from input dictionary.")
        return None

    # Using string literal for TypedDict key access
    feature = input_feature_dict[feature_name]  # type: ignore

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
                    f"but expected last dim 1. Unsqueezing to {reshaped_feature.shape}."
                )
                return reshaped_feature
            else:
                warnings.warn(
                    f"Feature {feature_name} has shape {feature.shape}, expected last dim 1, "
                    f"but cannot unsqueeze rank {feature.dim()} tensor. Skipping."
                )
                return None
        except Exception as e:
            warnings.warn(
                f"Could not unsqueeze feature {feature_name} (shape {feature.shape}) "
                f"to match expected dim {expected_dim}. Error: {e}. Skipping."
            )
            return None

    # General reshape attempt (less likely to be correct than unsqueeze for dim 1)
    # This path should ideally not be hit if unsqueeze handles the common case
    try:
        # This might fail if numel doesn't match
        reshaped_feature = feature.view(*feature.shape[:-1], expected_dim)
        warnings.warn(
            f"Attempting general reshape for feature {feature_name} from {feature.shape} "
            f"to {reshaped_feature.shape} (expected dim {expected_dim})."
        )
        return reshaped_feature
    except RuntimeError:
        warnings.warn(
            f"Feature {feature_name} has wrong shape {feature.shape}, "
            f"expected last dim to be {expected_dim}. Could not reshape. Skipping."
        )
        return None


def extract_atom_features(
    encoder: torch.nn.Module, input_feature_dict: InputFeatureDict
) -> torch.Tensor:
    """
    Extract atom features from input dictionary.

    Args:
        encoder: The encoder module instance (to access input_feature and linear_no_bias_f)
        input_feature_dict: Dictionary containing atom features

    Returns:
        Tensor of atom features
    """
    features = []

    # Process each feature individually
    for feature_name, feature_dim in encoder.input_feature.items():
        processed_feature = _process_feature(
            input_feature_dict, feature_name, feature_dim
        )
        if processed_feature is not None:
            features.append(processed_feature)

    # Check if we have any valid features
    if not features:
        raise ValueError("No valid features found in input dictionary.")

    # Concatenate features along last dimension
    cat_features = torch.cat(features, dim=-1)

    # Pass through atom encoder
    return encoder.linear_no_bias_f(cat_features)


def ensure_space_uid(input_feature_dict: InputFeatureDict) -> None:
    """
    Ensure ref_space_uid exists and has correct shape.

    Args:
        input_feature_dict: Dictionary of input features
    """
    ref_space_uid = safe_tensor_access(input_feature_dict, "ref_space_uid")

    # Check shape
    if ref_space_uid.shape[-1] != 3:
        warnings.warn(
            f"ref_space_uid has wrong shape {ref_space_uid.shape}, expected [..., 3]. "
            f"Setting to zeros."
        )
        input_feature_dict["ref_space_uid"] = torch.zeros(
            (*ref_space_uid.shape[:-1], 3), device=ref_space_uid.device
        )


def adapt_tensor_dimensions(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Adapt tensor dimensions to match a target size along the last dimension.

    Args:
        tensor: Input tensor to adapt
        target_dim: The desired size of the last dimension

    Returns:
        Adapted tensor with compatible dimensions
    """
    if tensor.size(-1) == target_dim:
        return tensor
    elif tensor.size(-1) == 1:
        # Expand singleton dimension
        return tensor.expand(*tensor.shape[:-1], target_dim)
    else:
        # Create compatible tensor with padding/truncation
        compatible_tensor = torch.zeros(
            *tensor.shape[:-1],
            target_dim,
            device=tensor.device,
            dtype=tensor.dtype,  # Match dtype
        )
        # Copy values where dimensions overlap
        copy_dim = min(tensor.size(-1), target_dim)
        compatible_tensor[..., :copy_dim] = tensor[..., :copy_dim]
        warnings.warn(
            f"Adapted tensor last dimension from {tensor.size(-1)} to {target_dim} "
            f"using padding/truncation."
        )
        return compatible_tensor