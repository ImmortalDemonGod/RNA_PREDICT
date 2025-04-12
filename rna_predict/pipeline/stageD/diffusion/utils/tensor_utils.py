"""
Tensor manipulation utilities for Stage D diffusion.

This module provides functions for manipulating tensors in the Stage D diffusion process.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def normalize_tensor_dimensions(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Normalize tensor dimensions and adjust the batch size.
    
    Modifies the input tensor by removing extra dimensions. For a 4D tensor, the second
    dimension is squeezed; for a 5D tensor, the first two dimensions are squeezed. After
    normalizing, if the tensor's first dimension does not match the target batch size,
    a warning is logged and the tensor is truncated accordingly.
    
    Args:
        tensor: The input tensor to normalize.
        batch_size: The desired batch size.
    
    Returns:
        The normalized tensor with dimensions adjusted to match the batch size.
    """
    # Handle dimension normalization (squeeze extra dimensions)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(1)
    elif tensor.dim() == 5:
        tensor = tensor.squeeze(0).squeeze(0)

    # Ensure batch size matches
    if tensor.shape[0] != batch_size:
        logger.warning(f"Adjusting batch size from {tensor.shape[0]} to {batch_size}")
        tensor = tensor[:batch_size]

    return tensor
