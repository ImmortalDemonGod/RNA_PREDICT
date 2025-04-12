"""
Tensor manipulation utilities for Stage D diffusion.

This module provides functions for manipulating tensors in the Stage D diffusion process.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def normalize_tensor_dimensions(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Normalize tensor dimensions by squeezing extra dimensions and ensuring batch size matches.

    Args:
        tensor: Input tensor to normalize
        batch_size: Target batch size

    Returns:
        Normalized tensor
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
