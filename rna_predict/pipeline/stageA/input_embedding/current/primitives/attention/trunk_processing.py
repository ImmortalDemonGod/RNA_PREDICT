"""
Core logic for processing tensors into dense trunks.
"""

from typing import List

import torch

from ..core_transforms import _apply_trunk_slices


def _process_tensor_to_trunks(
    tensor_list: List[torch.Tensor],
    dim_list: List[int],
    n_trunks: int,
    items_per_trunk: int,
) -> List[torch.Tensor]:
    """
    Process tensors into trunks by slicing and concatenating.

    Args:
        tensor_list: List of tensors to process.
        dim_list: List of dimensions along which to process each tensor.
        n_trunks: Number of trunks to create.
        items_per_trunk: Number of items (e.g., queries or keys) per trunk.

    Returns:
        List of processed tensor trunks.
    """
    result = []

    # Process each trunk
    for trunk_idx in range(n_trunks):
        # Apply trunk slices to get parts of each tensor for the current trunk
        sliced_tensors = _apply_trunk_slices(
            tensor_list, dim_list, trunk_idx, items_per_trunk
        )
        # Concatenate slices along the specified dimension (assuming consistent dim for concat)
        # Note: Original code used dim_list[0]. This assumes all tensors in the list
        # should be concatenated along the same dimension after slicing.
        if sliced_tensors: # Ensure there are tensors to concatenate
             result.append(torch.cat(sliced_tensors, dim=dim_list[0]))
        # Handle cases where slicing might result in empty lists if logic allows?
        # Current assumption: _apply_trunk_slices returns tensors to be concatenated.

    return result