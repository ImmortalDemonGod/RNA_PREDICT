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
        warnings.warn(f"Feature {feature_name} missing from input dictionary.")
        # Create a default tensor for the missing feature
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 1
        n_atoms = 1

        # Try to get batch_size and n_atoms from input_feature_dict
        if "ref_pos" in input_feature_dict and isinstance(input_feature_dict["ref_pos"], torch.Tensor):
            ref_pos = input_feature_dict["ref_pos"]
            if ref_pos.dim() >= 2:
                batch_size = ref_pos.shape[0]
                n_atoms = ref_pos.shape[1]
        elif "atom_to_token_idx" in input_feature_dict and isinstance(input_feature_dict["atom_to_token_idx"], torch.Tensor):
            atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
            if atom_to_token_idx.dim() >= 2:
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
        warnings.warn(f"Feature {feature_name} not found in input_feature_dict.")
        return None

    # Ensure feature is a tensor
    if not isinstance(feature, torch.Tensor):
        warnings.warn(f"Feature {feature_name} is not a tensor.")
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
        print(f"[DEBUG][extract_atom_features] Processing feature: {feature_name}, expected_dim: {feature_dim}")
        processed_feature = _process_feature(
            input_feature_dict, feature_name, feature_dim
        )
        if processed_feature is not None:
            print(f"[DEBUG][extract_atom_features] Processed {feature_name} shape: {processed_feature.shape}")
            features.append(processed_feature)

    # Check if we have any valid features
    if not features:
        print("[DEBUG][extract_atom_features] No valid features found, creating defaults.")
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create default tensors for required features
        batch_size = 1
        n_atoms = 1

        # Try to get batch_size and n_atoms from input_feature_dict
        if "atom_to_token_idx" in input_feature_dict and isinstance(input_feature_dict["atom_to_token_idx"], torch.Tensor):
            atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
            if atom_to_token_idx.dim() >= 2:
                batch_size = atom_to_token_idx.shape[0]
                n_atoms = atom_to_token_idx.shape[1]

        # Create default features with appropriate shapes
        for feature_name, feature_dim in encoder.input_feature.items():
            print(f"[DEBUG][extract_atom_features] Creating default for {feature_name} with dim {feature_dim}")
            if feature_name == "ref_pos":
                # Position tensor with shape [batch_size, n_atoms, 3]
                default_tensor = torch.zeros((batch_size, n_atoms, 3), device=default_device)
                # Use explicit key for TypedDict
                input_feature_dict["ref_pos"] = default_tensor
                features.append(default_tensor)
            elif feature_name == "ref_charge":
                # Charge tensor with shape [batch_size, n_atoms, 1]
                default_tensor = torch.zeros((batch_size, n_atoms, 1), device=default_device)
                # Use explicit key for TypedDict
                input_feature_dict["ref_charge"] = default_tensor
                features.append(default_tensor)
            elif feature_name == "ref_mask":
                # Mask tensor with shape [batch_size, n_atoms, 1]
                default_tensor = torch.ones((batch_size, n_atoms, 1), device=default_device)
                # Use explicit key for TypedDict
                input_feature_dict["ref_mask"] = default_tensor
                features.append(default_tensor)
            elif feature_name == "ref_element":
                # Element tensor with shape [batch_size, n_atoms, feature_dim]
                default_tensor = torch.zeros((batch_size, n_atoms, feature_dim), device=default_device)
                # Use explicit key for TypedDict
                input_feature_dict["ref_element"] = default_tensor
                features.append(default_tensor)
            elif feature_name == "ref_atom_name_chars":
                # Atom name tensor with shape [batch_size, n_atoms, feature_dim]
                default_tensor = torch.zeros((batch_size, n_atoms, feature_dim), device=default_device)
                # Use explicit key for TypedDict
                input_feature_dict["ref_atom_name_chars"] = default_tensor
                features.append(default_tensor)

        # If we still have no features, raise an error
        if not features:
            raise ValueError("No valid features found in input dictionary and could not create defaults.")

    # --- Start Dimension Alignment Fix ---
    # Check if any feature is missing the batch dimension (2D instead of 3D)
    # This happens in the test_residue_index_squeeze_fix_memory_efficient test
    has_2d_features = any(f.ndim == 2 for f in features)

    # If we have 2D features, add a batch dimension to all features
    if has_2d_features:
        print("[DEBUG][extract_atom_features] Detected 2D features without batch dimension. Adding batch dimension.")
        features_with_batch = []
        for f in features:
            if f.ndim == 2:  # [N_atom, feature_dim]
                # Add batch dimension -> [1, N_atom, feature_dim]
                f = f.unsqueeze(0)
                print(f"[DEBUG][extract_atom_features] Added batch dimension: {f.shape}")
            features_with_batch.append(f)
        features = features_with_batch

    # Find the maximum number of dimensions among the features
    max_dims = 0
    for f in features:
        max_dims = max(max_dims, f.ndim)

    # Determine the target sample dimension size (usually from tensors already having max_dims)
    target_sample_dim = 1
    # Check dims >= 3 assuming shape [B, S, N...] where S=target_sample_dim is at index 1
    if max_dims >= 3:
        for f in features:
            if f.ndim == max_dims:
                target_sample_dim = max(target_sample_dim, f.shape[1])

    aligned_features = []
    for f in features:
        temp_f = f
        # 1. Align number of dimensions by adding singleton sample dim at index 1 if needed
        # Handles common case like [B, N, C] needing to become [B, S, N, C]
        if temp_f.ndim == max_dims - 1 and max_dims >= 3:
             temp_f = temp_f.unsqueeze(1) # Add sample dim -> [B, 1, N, C]
        elif temp_f.ndim < max_dims:
             # Fallback for other potential dimension mismatches (e.g., missing batch dim)
             warnings.warn(f"Unexpected dimension mismatch for feature {f.shape}, target ndim {max_dims}. Attempting leading unsqueeze.")
             while temp_f.ndim < max_dims:
                 temp_f = temp_f.unsqueeze(0) # Add leading dims

        # 2. Expand the sample dimension (dim 1) if it's a singleton and needs to match target_sample_dim
        # Ensure the dimension exists before checking its size
        if temp_f.ndim == max_dims and max_dims >= 3 and temp_f.shape[1] == 1 and target_sample_dim > 1:
             try:
                 # Create target shape for expansion, only changing dim 1
                 expand_shape = list(temp_f.shape)
                 expand_shape[1] = target_sample_dim
                 temp_f = temp_f.expand(expand_shape)
             except RuntimeError as e:
                 raise RuntimeError(f"Failed to expand sample dimension for feature from {f.shape} to match target sample dim {target_sample_dim}. Current shape: {temp_f.shape}. Error: {e}")

        aligned_features.append(temp_f)
        print(f"[DEBUG][extract_atom_features] Aligned {f.shape} to {temp_f.shape}")
    # --- End Dimension Alignment Fix ---


    # Concatenate aligned features along last dimension
    # Add check to ensure shapes are compatible before concatenation
    if len(aligned_features) > 1:
        first_shape_prefix = aligned_features[0].shape[:-1]
        for i, t in enumerate(aligned_features[1:], 1):
            if t.shape[:-1] != first_shape_prefix:
                 raise RuntimeError(f"Shape mismatch before final concatenation in extract_atom_features. "
                                  f"Tensor 0 shape prefix: {first_shape_prefix}, "
                                  f"Tensor {i} shape prefix: {t.shape[:-1]}. "
                                  f"Original feature name likely: {list(encoder.input_feature.keys())[i] if hasattr(encoder, 'input_feature') else 'unknown'}") # type: ignore
    print("[DEBUG][extract_atom_features] Feature names and shapes before cat:")
    for i, f in enumerate(aligned_features):
        print(f"  Feature {i}: shape {f.shape}")
    cat_features = torch.cat(aligned_features, dim=-1) # Use aligned_features
    print(f"[DEBUG][extract_atom_features] Concatenated feature shape: {cat_features.shape}")
    expected_in_features = encoder.linear_no_bias_f.in_features if hasattr(encoder.linear_no_bias_f, 'in_features') else None
    print(f"[DEBUG][extract_atom_features] Expected in_features for linear_no_bias_f: {expected_in_features}")
    assert cat_features.shape[-1] == expected_in_features, (
        f"UNIQUE ERROR: Concatenated feature dim {cat_features.shape[-1]} does not match expected in_features {expected_in_features}")
    # Ensure encoder.linear_no_bias_f is callable before calling
    if not hasattr(encoder, "linear_no_bias_f") or not callable(
        encoder.linear_no_bias_f
    ):
        raise TypeError(
            "encoder.linear_no_bias_f is not callable or does not exist."
        )

    # Pass through atom encoder
    return encoder.linear_no_bias_f(cat_features) # type: ignore[operator] # Ignore previous error after check


def ensure_space_uid(input_feature_dict: InputFeatureDict) -> None:
    """
    Ensure ref_space_uid exists and has correct shape.

    Args:
        input_feature_dict: Dictionary of input features
    """
    # Create a default tensor if ref_space_uid is not found
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_shape = (1, 3)  # Default shape for ref_space_uid
    default_tensor = torch.zeros(default_shape, device=default_device)

    # Check if ref_space_uid exists in the input_feature_dict
    if "ref_space_uid" not in input_feature_dict:
        # Add default tensor if missing
        input_feature_dict["ref_space_uid"] = default_tensor
        return

    # Get the existing ref_space_uid
    ref_space_uid = input_feature_dict["ref_space_uid"]

    # If it's not a tensor, replace with default
    if not isinstance(ref_space_uid, torch.Tensor):
        input_feature_dict["ref_space_uid"] = default_tensor
        return

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
