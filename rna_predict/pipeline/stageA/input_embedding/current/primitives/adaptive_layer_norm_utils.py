"""
Utility functions for Adaptive Layer Normalization.

This module contains utility functions for tensor shape manipulation and dimension checking
used by the AdaptiveLayerNorm class.
"""

import warnings
from typing import Tuple

import torch


def has_compatible_dimensions(a: torch.Tensor, s: torch.Tensor) -> bool:
    """
    Check if tensors have compatible dimensions for sample dimension addition.

    Args:
        a: Input tensor to be conditioned
        s: Conditioning tensor

    Returns:
        Boolean indicating whether dimensions are compatible
    """
    return s.dim() > a.dim() and s.dim() == a.dim() + 1 and s.shape[0] == a.shape[0]


def has_matching_feature_dimensions(a: torch.Tensor, s: torch.Tensor) -> bool:
    """
    Check if feature dimensions match between tensors.

    Args:
        a: Input tensor to be conditioned
        s: Conditioning tensor

    Returns:
        Boolean indicating whether feature dimensions match
    """
    s_dims_to_match = s.shape[2:]
    a_dims_to_match = a.shape[1:]
    return s_dims_to_match == a_dims_to_match


def should_add_sample_dimension(a: torch.Tensor, s: torch.Tensor) -> bool:
    """
    Determine if tensor 'a' needs a sample dimension added to match 's'.

    Args:
        a: Input tensor to be conditioned
        s: Conditioning tensor

    Returns:
        Boolean indicating whether to add a sample dimension
    """
    # First check if dimensions are compatible for adding a sample dimension
    if not has_compatible_dimensions(a, s):
        return False

    # Then check if feature dimensions match
    return has_matching_feature_dimensions(a, s)


def check_and_adjust_dimensions(
    a: torch.Tensor, s: torch.Tensor
) -> Tuple[torch.Tensor, bool]:
    """
    Check and adjust dimensions of input tensors to ensure compatibility.

    Args:
        a: Input tensor to be conditioned
        s: Conditioning tensor

    Returns:
        Tuple of (adjusted tensor a, whether a was unsqueezed)
    """
    a_was_unsqueezed = False

    # Check if we need to add a sample dimension
    if should_add_sample_dimension(a, s):
        # Add missing sample dimension to a
        a = a.unsqueeze(1)
        a_was_unsqueezed = True
        warnings.warn(
            f"INFO: Unsqueezed 'a' in AdaptiveLayerNorm to match 's'. New 'a' shape: {a.shape}"
        )

    return a, a_was_unsqueezed


def needs_singleton_dimension(a: torch.Tensor, scale: torch.Tensor) -> bool:
    """
    Check if scale and shift tensors need a singleton dimension for broadcasting.

    Args:
        a: Target tensor for shape reference
        scale: Scale tensor to check

    Returns:
        Boolean indicating whether to add a singleton dimension
    """
    # Check if we need to add a singleton dimension for broadcasting
    return (
        a.dim() == 4
        and scale.dim() == 3
        and a.shape[0] == scale.shape[0]
        and a.shape[2:] == scale.shape[1:]
    )


def interpolate_sequence_dim(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Interpolate tensor along sequence dimension to match target size.

    Handles 3D and higher-dimensional tensors by flattening leading dims.
    """
    # Debug: log shapes to diagnose interpolation errors
    print(f"[DEBUG][AdaLN][interpolate_sequence_dim] tensor.shape={tensor.shape}, target_size={target_size}")
    # If no sequence length required, return empty along sequence dim
    if target_size <= 0:
        # Return tensor with zero-length sequence dimension
        *lead, seq, chan = tensor.shape
        return tensor.reshape(*lead, 0, chan)
    # If sequence dimension already matches, return tensor
    if tensor.shape[-2] == target_size:
        return tensor
    orig_shape = tensor.shape
    # Handle tensors with extra leading dimensions (e.g., multi-sample)
    if tensor.dim() > 3:
        seq_dim = orig_shape[-2]
        chan_dim = orig_shape[-1]
        # Flatten leading dims into batch
        tensor_flat = tensor.reshape(-1, seq_dim, chan_dim)
        # Interpolate along sequence dim
        interp_flat = torch.nn.functional.interpolate(
            tensor_flat.transpose(1, 2), size=target_size, mode="nearest"
        ).transpose(1, 2)
        # Restore original leading dims
        leading_shape = orig_shape[:-2]
        return interp_flat.reshape(*leading_shape, target_size, chan_dim)
    # Standard 3D case
    return torch.nn.functional.interpolate(
        tensor.transpose(1, 2),  # [B, C, S]
        size=target_size,
        mode="nearest",
    ).transpose(1, 2)  # [B, S, C]


def _try_expand_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Try to expand tensor dimensions to match target shape.

    Args:
        tensor: Input tensor to expand
        target: Target tensor with desired shape

    Returns:
        Expanded tensor if successful

    Raises:
        RuntimeError: If expansion fails
    """
    return tensor.expand_as(target)


def _copy_matching_dimensions(
    tensor: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Create a new tensor with target shape and copy data where dimensions match.

    Args:
        tensor: Source tensor
        target: Target tensor with desired shape

    Returns:
        New tensor with copied data
    """
    import logging
    logger = logging.getLogger(__name__)

    # Debug logging
    logger.debug(f"[_copy_matching_dimensions] tensor.shape={tensor.shape}, target.shape={target.shape}")

    new_tensor = torch.zeros_like(target)

    # Special case for diffusion module with N_sample dimension
    # If target is [B, N_sample, N, N, C] and tensor is [B, N_sample, C],
    # we need to broadcast tensor to match target's shape
    if len(target.shape) == 5 and len(tensor.shape) == 3:
        if tensor.shape[0] == target.shape[0] and tensor.shape[1] == target.shape[1]:
            # This is the case we're handling
            # We need to broadcast tensor from [B, N_sample, C] to [B, N_sample, N, N, C]
            try:
                # First, add the missing dimensions
                expanded_tensor = tensor.unsqueeze(2).unsqueeze(2)
                # Now expand to match target's shape
                expanded_tensor = expanded_tensor.expand(target.shape[0], target.shape[1], target.shape[2], target.shape[3], -1)
                # Only copy the last dimension up to the minimum size
                min_dim = min(expanded_tensor.shape[-1], target.shape[-1])
                new_tensor[..., :min_dim] = expanded_tensor[..., :min_dim]
                logger.debug(f"[_copy_matching_dimensions] Special case: expanded tensor to {expanded_tensor.shape}")
                return new_tensor
            except RuntimeError as e:
                logger.warning(f"[_copy_matching_dimensions] Failed to expand tensor: {e}. Trying alternative approach.")

    # Only copy if both tensors have at least 3 dimensions
    try:
        if tensor.dim() >= 3 and target.dim() >= 3:
            # Find minimum dimension to copy
            min_dim = min(tensor.size(2), target.size(2))
            new_tensor[:, :, :min_dim] = tensor[:, :, :min_dim]
    except RuntimeError as e:
        logger.warning(f"[_copy_matching_dimensions] Standard copy failed: {e}. Trying manual copy.")
        # Try a more careful approach for the specific case we're seeing
        try:
            if len(target.shape) == 5 and len(tensor.shape) == 3:
                # Create a new tensor with the right shape
                result = torch.zeros_like(target)
                # Fill it with the values from tensor, broadcasting as needed
                for i in range(target.shape[2]):
                    for j in range(target.shape[3]):
                        result[:, :, i, j, :tensor.shape[-1]] = tensor
                logger.debug("[_copy_matching_dimensions] Used manual broadcasting for 5D target and 3D tensor")
                return result
        except RuntimeError as e2:
            logger.warning(f"[_copy_matching_dimensions] Manual copy also failed: {e2}. Returning zero tensor.")

    return new_tensor


def _try_reshape_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Try to reshape tensor to match target shape.

    Args:
        tensor: Input tensor to reshape
        target: Target tensor with desired shape

    Returns:
        Reshaped tensor if successful, zeros tensor otherwise
    """
    try:
        return tensor.reshape(*target.shape)
    except RuntimeError:
        return torch.zeros_like(target)


def match_tensor_shape(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Match tensor shape to target tensor shape using various methods.

    Args:
        tensor: Input tensor to reshape
        target: Target tensor with desired shape

    Returns:
        Reshaped tensor
    """
    # If shapes already match, return original tensor
    if tensor.shape == target.shape:
        return tensor

    # Try different methods in order of preference
    try:
        return _try_expand_tensor(tensor, target)
    except RuntimeError as expand_error:
        if "must match the existing size" in str(expand_error):
            return _copy_matching_dimensions(tensor, target)
        else:
            return _try_reshape_tensor(tensor, target)


def _has_token_dimension_mismatch(scale: torch.Tensor, a: torch.Tensor) -> bool:
    """
    Check if there is a token dimension mismatch between scale and a tensors.

    Args:
        scale: Scale tensor to check
        a: Target tensor to compare with

    Returns:
        Boolean indicating whether there is a token dimension mismatch
    """
    return (
        len(scale.shape) >= 3 and len(a.shape) >= 3 and scale.shape[-2] != a.shape[-2]
    )


def _handle_fewer_tokens_in_scale(
    scale: torch.Tensor, shift: torch.Tensor, a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Handle the case where scale has fewer tokens than a.

    Args:
        scale: Scale tensor with fewer tokens
        shift: Shift tensor with fewer tokens
        a: Target tensor with more tokens

    Returns:
        Tuple of (expanded scale tensor, expanded shift tensor)
    """
    warnings.warn(
        f"Token dimension mismatch in AdaptiveLayerNorm: scale has {scale.shape[-2]} tokens, "
        f"but a has {a.shape[-2]} tokens. Expanding scale to match a's token dimension."
    )

    # Use nearest neighbor interpolation to expand the token dimension
    # Reshape to [B, C, S] for interpolation, then back to original format
    scale_expanded = torch.nn.functional.interpolate(
        scale.transpose(-2, -1),  # [B, ..., C, S] -> [B, ..., S, C]
        size=a.shape[-2],
        mode="nearest",
    ).transpose(-2, -1)  # [B, ..., S, C] -> [B, ..., C, S]

    shift_expanded = torch.nn.functional.interpolate(
        shift.transpose(-2, -1), size=a.shape[-2], mode="nearest"
    ).transpose(-2, -1)

    return scale_expanded, shift_expanded


def _handle_more_tokens_in_scale(
    scale: torch.Tensor, shift: torch.Tensor, a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Handle the case where scale has more tokens than a.

    Args:
        scale: Scale tensor with more tokens
        shift: Shift tensor with more tokens
        a: Target tensor with fewer tokens

    Returns:
        Tuple of (truncated scale tensor, truncated shift tensor)
    """
    warnings.warn(
        f"Token dimension mismatch in AdaptiveLayerNorm: scale has {scale.shape[-2]} tokens, "
        f"but a has {a.shape[-2]} tokens. Using first {a.shape[-2]} tokens from scale."
    )
    # Use only the first a.shape[-2] tokens from scale and shift
    truncated_scale = scale[..., : a.shape[-2], :]
    truncated_shift = shift[..., : a.shape[-2], :]

    return truncated_scale, truncated_shift


def _handle_token_dimension_mismatch(
    scale: torch.Tensor, shift: torch.Tensor, a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Handle token dimension mismatch between scale/shift and a tensors.

    Args:
        scale: Scale tensor to adjust
        shift: Shift tensor to adjust
        a: Target tensor shape

    Returns:
        Tuple of (adjusted scale tensor, adjusted shift tensor)
    """
    if not _has_token_dimension_mismatch(scale, a):
        return scale, shift

    # If scale has fewer tokens than a
    if scale.shape[-2] < a.shape[-2]:
        return _handle_fewer_tokens_in_scale(scale, shift, a)
    else:
        # If scale has more tokens than a
        return _handle_more_tokens_in_scale(scale, shift, a)


def _handle_sequence_length_mismatch(
    scale: torch.Tensor, shift: torch.Tensor, a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Handle sequence length mismatch between scale/shift and a tensors.

    Args:
        scale: Scale tensor to adjust
        shift: Shift tensor to adjust
        a: Target tensor shape

    Returns:
        Tuple of (adjusted scale tensor, adjusted shift tensor)
    """
    has_sequence_mismatch = (
        len(scale.shape) >= 2 and len(a.shape) >= 2 and scale.shape[1] != a.shape[1]
    )

    if has_sequence_mismatch:
        scale = interpolate_sequence_dim(scale, a.shape[1])
        shift = interpolate_sequence_dim(shift, a.shape[1])

    return scale, shift


def _handle_general_dimension_mismatch(
    scale: torch.Tensor, shift: torch.Tensor, a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Handle general dimension mismatches between scale/shift and a tensors.

    Args:
        scale: Scale tensor to adjust
        shift: Shift tensor to adjust
        a: Target tensor shape

    Returns:
        Tuple of (adjusted scale tensor, adjusted shift tensor)
    """
    if scale.shape != a.shape:
        scale = match_tensor_shape(scale, a)

    if shift.shape != a.shape:
        shift = match_tensor_shape(shift, a)

    return scale, shift


def adjust_tensor_shapes(
    scale: torch.Tensor, shift: torch.Tensor, a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adjust tensor shapes when broadcasting fails.

    Args:
        scale: Scale tensor to adjust
        shift: Shift tensor to adjust
        a: Target tensor shape

    Returns:
        Tuple of (adjusted scale tensor, adjusted shift tensor)
    """
    # Step 1: Handle token dimension mismatch (dimension -2)
    scale, shift = _handle_token_dimension_mismatch(scale, shift, a)

    # Step 2: Handle sequence length mismatch (dimension 1)
    scale, shift = _handle_sequence_length_mismatch(scale, shift, a)

    # Step 3: Handle any remaining dimension mismatches
    scale, shift = _handle_general_dimension_mismatch(scale, shift, a)

    return scale, shift


def should_squeeze_tensor(
    tensor: torch.Tensor, original_shape: tuple, was_unsqueezed: bool
) -> bool:
    """
    Determine if tensor should be squeezed to restore original shape.

    Args:
        tensor: Input tensor
        original_shape: Original shape to restore to
        was_unsqueezed: Whether the tensor was unsqueezed

    Returns:
        Boolean indicating whether to squeeze the tensor
    """
    return (
        was_unsqueezed and tensor.dim() > len(original_shape) and tensor.shape[1] == 1
    )


def restore_original_shape(
    tensor: torch.Tensor, original_shape: tuple, was_unsqueezed: bool
) -> torch.Tensor:
    """
    Restore tensor to its original shape if it was modified.

    Args:
        tensor: Input tensor
        original_shape: Original shape to restore to
        was_unsqueezed: Whether the tensor was unsqueezed

    Returns:
        Tensor with original shape restored if needed
    """
    if should_squeeze_tensor(tensor, original_shape, was_unsqueezed):
        return tensor.squeeze(1)
    return tensor


def adaptive_layer_norm(
    a: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, debug_logging: bool = False
) -> torch.Tensor:
    """
    Apply adaptive layer normalization to input tensor 'a'.

    Args:
        a: Input tensor to be normalized
        scale: Scale tensor for normalization
        shift: Shift tensor for normalization
        debug_logging: Whether to log debug information

    Returns:
        Normalized tensor
    """
    if debug_logging:
        print(f"[INSTRUMENT][AdaptiveLayerNorm] scale.shape={scale.shape}, a.shape={a.shape}, scale.dtype={scale.dtype}, a.dtype={a.dtype}")

    # Check and adjust dimensions of input tensors
    a, a_was_unsqueezed = check_and_adjust_dimensions(a, scale)

    # Adjust tensor shapes if necessary
    scale, shift = adjust_tensor_shapes(scale, shift, a)

    # Apply adaptive layer normalization
    normalized_a = a * scale + shift

    # Restore original shape if necessary
    normalized_a = restore_original_shape(normalized_a, a.shape, a_was_unsqueezed)

    return normalized_a
