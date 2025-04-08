"""
Core logic for processing tensors into dense trunks.
"""

import math
from typing import List

import torch
import torch.nn.functional as F

# Removed import of _apply_trunk_slices as logic is now self-contained


def _process_tensor_to_trunks(
    tensor_list: List[torch.Tensor],
    dim_list: List[int],
    items_per_trunk: int,
) -> List[torch.Tensor]:
    """
    Process each tensor in a list into trunks independently.
    Handles padding and zero-length sequences.

    Args:
        tensor_list: List of tensors to process (e.g., [(B1, S1, D), (B2, S2, D)]).
                     Assumes tensors have compatible shapes for concatenation later
                     along the batch dimension if needed, except for the sequence dimension.
        dim_list: List of sequence dimensions along which to process each tensor.
        items_per_trunk: Number of items (e.g., queries or keys) per trunk.

    Returns:
        List of processed trunked tensors. Each tensor will have shape like
        (B, NumTrunks, ItemsPerTrunk, D) or similar depending on original dims.
        Returns appropriately shaped empty tensors for zero-length inputs.
    """
    processed_tensors = []

    if not tensor_list:
        return []

    for i, tensor in enumerate(tensor_list):
        if tensor.numel() == 0 and tensor.shape[0] == 0:
             # Handle completely empty tensor (e.g., shape (0, 0, 16)) - difficult to infer target shape
             # Let's try to infer target shape based on the first non-empty tensor if possible,
             # otherwise, we might need more context or make assumptions.
             # For now, append the empty tensor as is, concatenation might handle it or fail later.
             # A better approach might be needed depending on downstream requirements.
             # Alternative: skip? But that changes list length.
             # Let's try creating a zero-batch-size tensor with expected rank
             try:
                 example_tensor = next(t for t in tensor_list if t.numel() > 0)
                 example_dim = dim_list[tensor_list.index(example_tensor)]
                 if example_dim < 0: example_dim += example_tensor.ndim
                 d_head = example_tensor.shape[-1] # Assume last dim is feature dim
                 # Shape: (0, num_trunks=0, items_per_trunk, d_head) - num_trunks is 0 for seq_len 0
                 empty_trunked = torch.empty(
                     0, 0, items_per_trunk, d_head,
                     dtype=tensor.dtype, device=tensor.device
                 )
                 processed_tensors.append(empty_trunked)
                 continue
             except StopIteration:
                 # All tensors are empty, very ambiguous case.
                 # Return list of original empty tensors.
                 processed_tensors.append(tensor)
                 continue


        dim = dim_list[i]
        if dim < 0:
            dim += tensor.ndim

        # Ensure dim is valid
        if not (0 <= dim < tensor.ndim):
             raise IndexError(f"Dimension {dim_list[i]} out of range for tensor {i} with shape {tensor.shape}")

        seq_len = tensor.shape[dim]
        batch_size = tensor.shape[0] # Assume batch is always dim 0
        feature_dim = tensor.shape[-1] # Assume features are always last dim

        # --- Handle Zero Sequence Length ---
        if seq_len == 0:
            # Expected shape: (B, NumTrunks=0, ItemsPerTrunk, D)
            # Create empty tensor with correct rank and dimensions
            shape_before = list(tensor.shape[:dim])
            shape_after = list(tensor.shape[dim+1:])
            # Target shape: (*shape_before, num_trunks=0, items_per_trunk, *shape_after)
            # Simplified target: (B, 0, items_per_trunk, D) assuming B is dim 0, D is last dim
            empty_trunked_shape = [batch_size, 0, items_per_trunk, feature_dim]
            # Adjust if original tensor had more dims (e.g., heads)
            if tensor.ndim > 3:
                 # Example: (B, H, S, D) -> (B, H, 0, items_per_trunk, D)
                 other_dims = list(tensor.shape[1:dim]) # Dims between B and S
                 empty_trunked_shape = [batch_size] + other_dims + [0, items_per_trunk] + shape_after

            empty_trunked = torch.empty(empty_trunked_shape, dtype=tensor.dtype, device=tensor.device)
            processed_tensors.append(empty_trunked)
            continue

        # --- Handle Non-Zero Sequence Length ---
        num_trunks_tensor = math.ceil(seq_len / items_per_trunk)
        padded_len = num_trunks_tensor * items_per_trunk
        padding_needed = padded_len - seq_len

        padded_tensor = tensor
        if padding_needed > 0:
            # Create padding tuple dynamically
            # (pad_left, pad_right, pad_dim_before, pad_dim_after, ...)
            # We need to pad dimension 'dim' at the end.
            padding_tuple = [0] * (2 * tensor.ndim)
            # Target the specific dimension: index is 2 * (ndim - 1 - dim) + 1 for 'after' padding
            padding_tuple_idx_after = 2 * (tensor.ndim - 1 - dim) + 1
            padding_tuple[padding_tuple_idx_after] = padding_needed
            padded_tensor = F.pad(tensor, tuple(padding_tuple), "constant", 0)

        # Reshape into trunks: (B, ..., S, ...) -> (B, ..., NumTrunks, ItemsPerTrunk, ...)
        shape_before = list(padded_tensor.shape[:dim])
        shape_after = list(padded_tensor.shape[dim+1:])
        target_shape = shape_before + [num_trunks_tensor, items_per_trunk] + shape_after

        try:
            trunked_tensor = padded_tensor.reshape(*target_shape)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to reshape tensor with shape {padded_tensor.shape} "
                f"to target shape {target_shape}. Original shape {tensor.shape}, dim {dim}. Error: {e}"
            ) from e

        processed_tensors.append(trunked_tensor)

    return processed_tensors
