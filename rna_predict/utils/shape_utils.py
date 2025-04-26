"""
Utility functions for handling tensor shape adjustments.

This module provides functions to adjust tensor shapes to match expected dimensions,
which helps prevent shape mismatch errors in operations like LayerNorm and attention bias.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


def adjust_tensor_feature_dim(
    tensor: torch.Tensor, expected_feature_dim: int, tensor_name: str = "tensor"
) -> torch.Tensor:
    """
    Adjusts the last dimension (features) of a tensor to match the expected size.

    Pads with zeros if the actual dimension is smaller.
    Slices if the actual dimension is larger.

    Args:
        tensor: The input tensor.
        expected_feature_dim: The target size for the last dimension.
        tensor_name: A descriptive name for logging purposes.

    Returns:
        The adjusted tensor.
    """
    actual_feature_dim = tensor.shape[-1]

    if actual_feature_dim == expected_feature_dim:
        return tensor
    elif actual_feature_dim < expected_feature_dim:
        padding_size = expected_feature_dim - actual_feature_dim
        # Pad only the last dimension: (0, padding_size) means no padding for other dims
        adjusted_tensor = F.pad(tensor, (0, padding_size), mode='constant', value=0)
        logger.debug(
            f"Padded '{tensor_name}' from {actual_feature_dim} to {expected_feature_dim} features. "
            f"Original shape: {tensor.shape}, New shape: {adjusted_tensor.shape}"
        )
        return adjusted_tensor
    else:  # actual_feature_dim > expected_feature_dim
        # Slice the last dimension
        slices = [slice(None)] * tensor.dim()
        slices[-1] = slice(0, expected_feature_dim)
        adjusted_tensor = tensor[tuple(slices)]
        logger.debug(
            f"Sliced '{tensor_name}' from {actual_feature_dim} to {expected_feature_dim} features. "
            f"Original shape: {tensor.shape}, New shape: {adjusted_tensor.shape}"
        )
        return adjusted_tensor


def adjust_attention_bias(
    bias: torch.Tensor,
    scores_shape: Tuple[int, ...],
    tensor_name: str = "attention_bias"
) -> torch.Tensor:
    """
    Adjusts attention bias tensor to match the shape of attention scores.

    This handles the common case where bias has shape [B, H, N_q, N_k] or [1, 1, N_q, N_k]
    and needs to match scores with shape [..., N_q, N_k].

    Args:
        bias: The attention bias tensor.
        scores_shape: The shape of the attention scores tensor.
        tensor_name: A descriptive name for logging purposes.

    Returns:
        The adjusted bias tensor that can be safely added to scores.
    """
    # Extract the query and key dimensions from scores
    n_queries, n_keys = scores_shape[-2], scores_shape[-1]

    # Check if bias already has the right shape for the last two dimensions
    if bias.shape[-2:] == (n_queries, n_keys):
        # If bias has more leading dimensions than needed, we can broadcast
        return bias

    # If bias has wrong dimensions, we need to adjust it
    if bias.shape[-2] != n_queries or bias.shape[-1] != n_keys:
        # First ensure bias has at least 2D
        if bias.dim() < 2:
            bias = bias.view(1, 1)

        # Adjust query dimension (second to last)
        if bias.shape[-2] < n_queries:
            # Pad with zeros
            pad_queries = n_queries - bias.shape[-2]
            # Create padding tuple for F.pad: (left, right, top, bottom, ...)
            # We want to pad the second-to-last dimension on the bottom
            padding = (0, 0, 0, pad_queries)
            bias = F.pad(bias, padding, mode='constant', value=0)
        elif bias.shape[-2] > n_queries:
            # Slice to match
            slices = [slice(None)] * bias.dim()
            slices[-2] = slice(0, n_queries)
            bias = bias[tuple(slices)]

        # Adjust key dimension (last)
        if bias.shape[-1] < n_keys:
            # Pad with zeros
            pad_keys = n_keys - bias.shape[-1]
            # We want to pad the last dimension on the right
            # Create padding tuple for F.pad: (left_last, right_last, left_second_last, right_second_last)
            padding = (0, pad_keys, 0, 0)
            bias = F.pad(bias, padding, mode='constant', value=0)
        elif bias.shape[-1] > n_keys:
            # Slice to match
            slices = [slice(None)] * bias.dim()
            slices[-1] = slice(0, n_keys)
            bias = bias[tuple(slices)]

        logger.debug(
            f"Adjusted '{tensor_name}' to match attention scores. "
            f"Original shape: {bias.shape}, New shape: {bias.shape}, "
            f"Target dimensions: ({n_queries}, {n_keys})"
        )

    return bias


def expand_tensor_for_samples(
    tensor: Optional[torch.Tensor],
    num_samples: int = 1,
    sample_dim: int = 1,
    tensor_name: str = "tensor"
) -> Optional[torch.Tensor]:
    """
    Expands a tensor to include a sample dimension for multi-sample diffusion.

    This handles the case where a tensor needs an extra dimension for samples,
    typically inserted after the batch dimension.

    Args:
        tensor: The input tensor to expand.
        num_samples: Number of samples to expand to.
        sample_dim: The dimension index where the sample dimension should be inserted.
        tensor_name: A descriptive name for logging purposes.

    Returns:
        The expanded tensor with an extra sample dimension.
    """
    if tensor is None:
        return None

    # If tensor already has the right number of dimensions and the sample dimension is correct
    if tensor.dim() > sample_dim and tensor.shape[sample_dim] == num_samples:
        return tensor

    # Insert a new dimension at the specified position
    shape = list(tensor.shape)
    shape.insert(sample_dim, num_samples)

    # Expand the tensor to the new shape
    expanded = tensor.unsqueeze(sample_dim).expand(shape)

    logger.debug(
        f"Expanded '{tensor_name}' to include sample dimension. "
        f"Original shape: {tensor.shape}, New shape: {expanded.shape}"
    )

    return expanded


def ensure_consistent_sample_dimensions(
    trunk_embeddings: Dict[str, torch.Tensor],
    input_features: Optional[Dict[str, Any]] = None,
    num_samples: int = 1,
    sample_dim: int = 1
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    """
    Ensures all tensors in trunk_embeddings and input_features have consistent sample dimensions.

    This is particularly important for diffusion models where some tensors might have an extra
    sample dimension while others don't.

    Args:
        trunk_embeddings: Dictionary of trunk embeddings (s_trunk, pair, etc.)
        input_features: Optional dictionary of input features
        num_samples: Number of samples to ensure
        sample_dim: The dimension index where the sample dimension should be

    Returns:
        Tuple of (updated trunk_embeddings, updated input_features)
    """
    # Process trunk_embeddings
    updated_trunk_embeddings = {}
    for key, tensor in trunk_embeddings.items():
        if not isinstance(tensor, torch.Tensor):
            updated_trunk_embeddings[key] = tensor
            continue

        # Check if this tensor needs expansion
        if tensor.dim() <= sample_dim or tensor.shape[sample_dim] != num_samples:
            updated_trunk_embeddings[key] = expand_tensor_for_samples(
                tensor, num_samples, sample_dim, f"trunk_embeddings[{key}]"
            )
        else:
            updated_trunk_embeddings[key] = tensor

    # Process input_features if provided
    if input_features is not None:
        updated_input_features = {}
        # Whitelist of per-atom features that should NOT be broadcast along sample dimension
        per_atom_keys = {
            "atom_to_token_idx", "ref_pos", "ref_space_uid", "ref_charge", "ref_element", "ref_atom_name_chars", "ref_mask"
        }
        for key, value in input_features.items():
            if not isinstance(value, torch.Tensor):
                updated_input_features[key] = value
                continue

            # Skip broadcasting for per-atom metadata features
            if key in per_atom_keys:
                updated_input_features[key] = value
                continue

            # Process all other tensor input features to ensure consistent sample dimensions
            if value.dim() <= sample_dim or value.shape[sample_dim] != num_samples:
                updated_input_features[key] = expand_tensor_for_samples(
                    value, num_samples, sample_dim, f"input_features[{key}]"
                )
            else:
                updated_input_features[key] = value

        return updated_trunk_embeddings, updated_input_features

    return updated_trunk_embeddings, input_features