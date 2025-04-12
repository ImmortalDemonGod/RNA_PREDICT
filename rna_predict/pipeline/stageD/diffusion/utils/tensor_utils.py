"""
Tensor manipulation utilities for Stage D diffusion.

This module provides functions for manipulating tensors in the Stage D diffusion process.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def normalize_tensor_dimensions(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Normalize tensor dimensions by removing extra singleton dimensions and enforcing the target batch size.
    
    For a 4-D tensor, the function squeezes the second dimension; for a 5-D tensor, it squeezes the first two dimensions.
    If the tensor's first dimension (batch size) does not match the specified target, a warning is logged and the tensor
    is sliced to match the target batch size.
    
    Args:
        tensor: Input tensor to normalize, expected to have 4 or 5 dimensions.
        batch_size: Desired batch size for the output tensor.
    
    Returns:
        The normalized tensor with adjusted dimensions and batch size.
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
