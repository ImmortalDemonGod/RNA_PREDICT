"""
Utility functions for Adaptive Layer Normalization.

This module contains utility functions for tensor shape manipulation and dimension checking
used by the AdaptiveLayerNorm class.
"""

import torch
import warnings


def has_compatible_dimensions(a: torch.Tensor, s: torch.Tensor) -> bool:
    """
    Determines if tensors are compatible for sample dimension addition.
    
    Checks that the conditioning tensor (s) has exactly one more dimension than the input tensor (a)
    and that the size of their first dimension matches, indicating that a sample dimension can be added
    to the input tensor.
    
    Args:
        a: The input tensor that may require a sample dimension.
        s: The conditioning tensor with an expected extra sample dimension.
    
    Returns:
        True if the tensors meet the compatibility criteria for sample dimension addition, otherwise False.
    """
    return (s.dim() > a.dim() and
            s.dim() == a.dim() + 1 and
            s.shape[0] == a.shape[0])


def has_matching_feature_dimensions(a: torch.Tensor, s: torch.Tensor) -> bool:
    """
    Checks if feature dimensions match for input and conditioning tensors.
    
    This function compares the feature dimensions of tensor `a` and tensor `s` by 
    using all dimensions starting from the second axis of `a` and the third axis of `s`.
    It returns True if these dimensions are identical, ensuring that the tensors are 
    compatible in terms of their feature layout.
    
    Args:
        a: The tensor to be conditioned.
        s: The conditioning tensor.
    
    Returns:
        True if the feature dimensions match, otherwise False.
    """
    s_dims_to_match = s.shape[2:]
    a_dims_to_match = a.shape[1:]
    return s_dims_to_match == a_dims_to_match


def should_add_sample_dimension(a: torch.Tensor, s: torch.Tensor) -> bool:
    """
    Determines if a sample dimension should be added to tensor `a` for compatibility with `s`.
    
    This function first verifies that the dimensions of `a` and `s` are suitable for adding a
    sample dimension and then checks that their feature dimensions match. It returns True if both
    conditions are met, indicating that `a` should be modified by adding a sample dimension.
    
    Args:
        a (torch.Tensor): The tensor that may require an additional sample dimension.
        s (torch.Tensor): The tensor used to determine the necessary shape for alignment.
    
    Returns:
        bool: True if tensor `a` requires a sample dimension to match `s`, otherwise False.
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
    Adjusts the input tensor's shape to be compatible with the conditioning tensor.
    
    If the input tensor is missing a sample dimension, this function unsqueezes it
    along dimension 1 and emits a warning. It returns the adjusted tensor along with
    a boolean flag indicating whether an unsqueeze operation was applied.
    
    Args:
        a: The input tensor to adjust.
        s: The conditioning tensor used to determine if a new sample dimension is needed.
    
    Returns:
        A tuple containing the adjusted tensor and a boolean flag that is True if the
        tensor was unsqueezed.
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
    Determine if the scale tensor requires a singleton dimension for broadcasting.
    
    Checks whether the target tensor `a` is 4D and the scale tensor is 3D with a matching
    batch dimension and spatial dimensions. Returns True if adding a singleton dimension to
    the scale tensor would align it with `a` for broadcasting.
        
    Args:
        a: The reference tensor providing the target shape (expected to be 4D).
        scale: The tensor to evaluate (expected to be 3D).
    
    Returns:
        True if the scale tensor should be unsqueezed to match the dimensions of `a`; otherwise, False.
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
    Interpolate the sequence dimension of a tensor to a desired target length.
    
    Assumes the input tensor has shape [B, S, C], where S is the sequence length, and
    uses nearest-neighbor interpolation to adjust this dimension while preserving the
    batch and feature dimensions.
    
    Args:
        tensor: Input tensor with shape [B, S, C].
        target_size: Desired size of the sequence dimension.
    
    Returns:
        Tensor with the sequence dimension resized to match target_size.
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
    Creates a new tensor with the target shape and copies matching third-dimension data.
    
    This function initializes a tensor filled with zeros matching the target tensor's shape.
    If both input tensors have at least three dimensions, it copies data along the third dimension
    (up to the minimum size between the source and target) from the source tensor into the new tensor.
    If either tensor has fewer than three dimensions, the function returns the zero-initialized tensor.
        
    Args:
        tensor: The source tensor from which to copy data.
        target: A tensor whose shape is used to initialize and define the output tensor.
        
    Returns:
        A new tensor with the same shape as the target tensor containing copied data for matching dimensions.
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
    Attempts to reshape the input tensor to match the target tensor's shape.
    
    If reshaping fails due to incompatible dimensions, returns a zeros tensor
    with the same shape as the target.
    
    Args:
        tensor: The tensor to be reshaped.
        target: A tensor whose shape is used as the desired output shape.
    
    Returns:
        A tensor reshaped to the target's dimensions, or a zeros tensor if the
        operation is unsuccessful.
    """
    try:
        return tensor.reshape(*target.shape)
    except RuntimeError:
        return torch.zeros_like(target)


def match_tensor_shape(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Adjusts the input tensor's shape to match the target tensor's dimensions.
    
    If the input tensor already has the same shape as the target, it is returned unchanged.
    Otherwise, the function applies a series of strategies—such as expanding dimensions,
    copying matching sections, or reshaping—to produce a tensor that aligns with the target.
        
    Args:
        tensor: Tensor to be reshaped.
        target: Tensor providing the desired shape.
        
    Returns:
        A tensor whose shape matches that of the target.
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
    Adjusts the scale and shift tensors to match the shape of a reference tensor.
    
    This function reconciles shape mismatches for broadcasting by first addressing 
    differences in the sequence dimension via interpolation, and then aligning the 
    remaining dimensions through reshaping. The adjustments ensure that both the scale 
    and shift tensors become compatible with the target tensor's shape.
    
    Args:
        scale: The tensor of scaling factors to be adjusted.
        shift: The tensor of shifting values to be adjusted.
        a: The reference tensor whose shape defines the target dimensions.
    
    Returns:
        A tuple containing the adjusted scale tensor and the adjusted shift tensor.
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
    Determine whether a tensor should be squeezed to restore its original shape.
    
    This function checks if an unsqueezed tensor has acquired an extra singleton
    dimension (typically in the second position) compared to its original shape.
    If the tensor was previously unsqueezed and now has more dimensions than
    indicated by the original shape, with the second dimension being 1, it returns True.
    
    Args:
        tensor: The tensor to evaluate.
        original_shape: The shape of the tensor before unsqueezing.
        was_unsqueezed: Flag indicating if the tensor was unsqueezed prior to processing.
    
    Returns:
        True if the tensor has an extra singleton dimension that should be removed;
        otherwise, False.
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
    Restore the tensor to its original shape if it was previously modified.
    
    This function checks whether the tensor requires a squeeze operation based on the
    original shape and a flag indicating if it was unsqueezed. If a squeeze is needed,
    the tensor is squeezed along the second dimension; otherwise, it is returned unchanged.
    
    Args:
        tensor: The tensor to restore.
        original_shape: The target shape that the tensor should match.
        was_unsqueezed: A flag indicating if the tensor was previously unsqueezed.
    
    Returns:
        The tensor with its shape restored to the original dimensions if applicable.
    """
    if should_squeeze_tensor(tensor, original_shape, was_unsqueezed):
        return tensor.squeeze(1)
    return tensor
