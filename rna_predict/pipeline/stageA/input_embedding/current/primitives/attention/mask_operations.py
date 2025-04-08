"""
Mask operations module for attention mechanisms.

This module contains functions for creating and manipulating masks
for use in attention mechanisms.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class MaskCreationConfig:
    """Configuration for mask creation."""

    n_queries: int
    n_keys: int
    query_lists: List[torch.Tensor]
    key_lists: List[torch.Tensor]
    query_dims: List[int]
    key_dims: List[int]
    n_q_chunks: int
    n_k_chunks: int
    q_trunk_indices: List[int]
    n_q_per_chunk: int
    window_size: int
    original_query_length: int


@dataclass
class MaskSliceInfo:
    """Information about mask slices."""

    q_trunk_idx: int
    mask_slice: List[int]
    chunk_idx: int
    window_size: int


@dataclass
class TensorMasksConfig:
    """Configuration for tensor masks creation."""

    n_q_trunks: int
    n_k_trunks: int
    n_queries: int
    n_keys: int
    device: torch.device


def _init_mask_slices(config: MaskCreationConfig) -> List[List[int]]:
    """
    Initialize mask slices for each query chunk.

    Args:
        config (MaskCreationConfig): Configuration for mask creation

    Returns:
        List[List[int]]: Initialized mask slices
    """
    return [[] for _ in range(config.n_q_chunks)]


def _calculate_mask_slice(
    j: int, info: MaskSliceInfo, config: MaskCreationConfig
) -> List[int]:
    """
    Calculate a single mask slice.

    Args:
        j (int): Current query index
        info (MaskSliceInfo): Mask slice information
        config (MaskCreationConfig): Configuration for mask creation

    Returns:
        List[int]: Calculated mask slice
    """
    # Calculate index within the trunked tensor
    q_trunk_idx = info.q_trunk_idx + j

    # Skip if outside the original query length
    if q_trunk_idx >= config.original_query_length:
        return []

    # Start from the chunk index minus window size, clamped to 0
    start = max(0, info.chunk_idx - info.window_size)
    # End at the chunk index plus window size plus 1, clamped to num chunks
    end = min(config.n_k_chunks, info.chunk_idx + info.window_size + 1)

    # Return the mask slice with all key chunks in the window
    return list(range(start, end))


def _create_masks(
    config: MaskCreationConfig,
) -> List[List[int]]:
    """
    Create a list of masks for each chunk of trunked query.

    Args:
        config (MaskCreationConfig): Configuration for mask creation

    Returns:
        List[List[int]]: List of mask slices for each query chunk
    """
    # Initialize mask slices for each query chunk
    mask_slices = _init_mask_slices(config)

    # For each query chunk
    for i, q_trunk_idx in enumerate(config.q_trunk_indices):
        # For each query in the chunk
        for j in range(config.n_q_per_chunk):
            # Create info object for current mask slice
            mask_info = MaskSliceInfo(
                q_trunk_idx=q_trunk_idx,
                mask_slice=[],
                chunk_idx=i,
                window_size=config.window_size,
            )

            # Calculate mask slice
            current_slice = _calculate_mask_slice(j, mask_info, config)

            # Skip if empty (out of bounds)
            if not current_slice:
                continue

            # Assign to the appropriate position in mask_slices
            position = i * config.n_q_per_chunk + j
            if position < len(mask_slices):
                mask_slices[position] = current_slice

    return mask_slices


def create_tensor_masks(
    mask_slices: List[List[int]],
    config: TensorMasksConfig,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Create tensor masks from mask slices.

    Args:
        mask_slices: List of mask slices
        config: Configuration for tensor masks creation

    Returns:
        Tuple of query masks, key masks, and trunked mask
    """
    # Convert mask slices to tensors
    q_masks = []
    for i, mask_slice in enumerate(mask_slices):
        mask = torch.zeros(
            (config.n_q_trunks, config.n_queries),
            dtype=torch.bool,
            device=config.device,
        )
        if mask_slice:
            for idx in mask_slice:
                mask[i, idx] = True
        q_masks.append(mask)

    # Create k_masks similarly
    k_masks = []
    for i in range(config.n_k_trunks):
        mask = torch.zeros(
            (config.n_k_trunks, config.n_keys), dtype=torch.bool, device=config.device
        )
        mask[i, :] = True  # Mark all keys as valid
        k_masks.append(mask)

    # Create trunk validity mask
    mask_trunked = torch.zeros(
        (config.n_q_trunks, config.n_k_trunks), dtype=torch.bool, device=config.device
    )
    for i in range(config.n_q_trunks):
        mask_trunked[i, :] = True

    return q_masks, k_masks, mask_trunked
