"""
Tensor manipulation utilities for Stage D diffusion.

This module provides functions for manipulating tensors in the Stage D diffusion process.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def normalize_tensor_dimensions(tensor: torch.Tensor, batch_size: int, key: str = None) -> torch.Tensor:
    """
    Normalize tensor dimensions by squeezing extra dimensions and ensuring batch size matches.
    For pair embeddings (key == 'pair' or 'z_trunk'), preserves [B, N_res, N_res, C] shape even if N_res=1.
    Args:
        tensor: Input tensor to normalize
        batch_size: Target batch size
        key: Optional, type of embedding (affects squeezing behavior)
    Returns:
        Normalized tensor
    """
    # For pair embeddings, do not squeeze dim 1 if shape is [B, 1, 1, C]
    if key in {"pair", "z_trunk"}:
        if tensor.dim() == 4 and tensor.shape[1] == 1 and tensor.shape[2] == 1:
            # Do not squeeze; preserve [B, 1, 1, C]
            pass
        else:
            # Fallback to legacy behavior for higher N_res
            if tensor.dim() == 4:
                tensor = tensor.squeeze(1)
            elif tensor.dim() == 5:
                tensor = tensor.squeeze(0).squeeze(0)
    else:
        # Legacy behavior for single embeddings
        if tensor.dim() == 4:
            tensor = tensor.squeeze(1)
        elif tensor.dim() == 3 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        elif tensor.dim() == 5:
            tensor = tensor.squeeze(0).squeeze(0)

    # Ensure batch size matches
    if tensor.shape[0] != batch_size:
        logger.warning(f"Adjusting batch size from {tensor.shape[0]} to {batch_size}")
        tensor = tensor[:batch_size]

    # Unique error if shape was collapsed for pair embedding
    if key in {"pair", "z_trunk"} and tensor.dim() == 3 and batch_size > 0:
        raise RuntimeError(
            f"[UNIQUE-PAIR-SHAPE-ERROR] Pair embedding shape collapsed to {tensor.shape} for key={key}. "
            f"Expected [B, N_res, N_res, C] for pairwise embeddings."
        )

    return tensor
