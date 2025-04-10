"""
Utility functions for Adaptive Layer Normalization.

This module contains utility functions for tensor shape manipulation and dimension checking
used by the AdaptiveLayerNorm class.
"""

import torch
import warnings


def has_compatible_dimensions(a: torch.Tensor, s: torch.Tensor) -> bool:
    """
    Check if tensors have compatible dimensions for sample dimension addition.

    Args:
        a: Input tensor to be conditioned
        s: Conditioning tensor

    Returns:
        Boolean indicating whether dimensions are compatible
    """
    return (s.dim() > a.dim() and
            s.dim() == a.dim() + 1 and
            s.shape[0] == a.shape[0])


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
) -> tuple[torch.Tensor, bool]:
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
        a.dim() == 4 and
        scale.dim() == 3 and
        a.shape[0] == scale.shape[0] and
        a.shape[2:] == scale.shape[1:]
    )


def interpolate_sequence_dim(
    tensor: torch.Tensor, target_size: int
) -> torch.Tensor:
    """
    Interpolate tensor along sequence dimension to match target size.

    Args:
        tensor: Input tensor
        target_size: Target sequence length

    Returns:
        Interpolated tensor
    """
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


def _copy_matching_dimensions(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Create a new tensor with target shape and copy data where dimensions match.

    Args:
        tensor: Source tensor
        target: Target tensor with desired shape

    Returns:
        New tensor with copied data
    """
    new_tensor = torch.zeros_like(target)

    # Only copy if both tensors have at least 3 dimensions
    if tensor.dim() >= 3 and target.dim() >= 3:
        # Find minimum dimension to copy
        min_dim = min(tensor.size(2), target.size(2))
        new_tensor[:, :, :min_dim] = tensor[:, :, :min_dim]

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


def adjust_tensor_shapes(
    scale: torch.Tensor, shift: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adjust tensor shapes when broadcasting fails.

    Args:
        scale: Scale tensor to adjust
        shift: Shift tensor to adjust
        a: Target tensor shape

    Returns:
        Tuple of (adjusted scale tensor, adjusted shift tensor)
    """
    # Handle sequence length mismatch with interpolation
    if scale.shape[1] != a.shape[1]:
        scale = interpolate_sequence_dim(scale, a.shape[1])
        shift = interpolate_sequence_dim(shift, a.shape[1])

    # Handle other dimension mismatches
    if scale.shape != a.shape:
        scale = match_tensor_shape(scale, a)

    if shift.shape != a.shape:
        shift = match_tensor_shape(shift, a)

    return scale, shift


def should_squeeze_tensor(tensor: torch.Tensor, original_shape: tuple, was_unsqueezed: bool) -> bool:
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
        was_unsqueezed and
        tensor.dim() > len(original_shape) and
        tensor.shape[1] == 1
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
