"""
Utility functions for creating masks and preparing padding information
for dense trunk attention.
"""

import math # Added import
from typing import List, Tuple, Union, Dict, Any # Added Dict, Any

import torch

# Assuming PaddingInfo is a dataclass/TypedDict defined elsewhere,
# but we are changing the return type to a plain dict now.
# from ..attention_utils import PaddingInfo # Keep if needed elsewhere, but not for return type
from .config_types import MaskConfigParams, PaddingInfoParams, TrunkDimensionsParams
from .mask_operations import (
    MaskCreationConfig,
    TensorMasksConfig,
    _create_masks,
    create_tensor_masks,
)


def _calculate_trunk_dimensions(
    params: TrunkDimensionsParams,
) -> Tuple[int, int, int, int]:
    """
    Calculate trunk dimensions for padding info preparation.

    Args:
        params: Parameters for trunk dimension calculation

    Returns:
        Tuple containing total_q, total_k, n_q_trunks, n_k_trunks based on the first tensor.
    """
    if params.trunk_info is None:
        # --- Modified Logic: Use first tensor's dimensions ---
        total_q = 0
        total_k = 0
        n_q_trunks = 0
        n_k_trunks = 0

        # Get dimensions from the first query tensor, if available
        if params.q_list:
            first_q = params.q_list[0]
            first_q_dim = params.dim_q_list[0]
            if first_q_dim < 0:
                first_q_dim += first_q.ndim
            if 0 <= first_q_dim < first_q.ndim:
                total_q = first_q.shape[first_q_dim]
                n_q_trunks = math.ceil(total_q / params.n_queries) if total_q > 0 else 0
            else:
                 print(f"Warning: Invalid dimension {params.dim_q_list[0]} for first query tensor with shape {first_q.shape}")


        # Get dimensions from the first key tensor, if available
        if params.k_list:
            first_k = params.k_list[0]
            first_k_dim = params.dim_k_list[0]
            if first_k_dim < 0:
                first_k_dim += first_k.ndim
            if 0 <= first_k_dim < first_k.ndim:
                total_k = first_k.shape[first_k_dim]
                n_k_trunks = math.ceil(total_k / params.n_keys) if total_k > 0 else 0
            else:
                 print(f"Warning: Invalid dimension {params.dim_k_list[0]} for first key tensor with shape {first_k.shape}")

        # --- End Modified Logic ---

    else: # Use provided trunk_info if available
        total_q = params.trunk_info.total_queries
        total_k = params.trunk_info.total_keys
        n_q_trunks = params.trunk_info.n_q_trunks
        n_k_trunks = params.trunk_info.n_k_trunks

    # Ensure non-negative trunk numbers
    n_q_trunks = max(0, n_q_trunks)
    n_k_trunks = max(0, n_k_trunks)

    # Note: total_q and total_k now represent the length of the *first* tensor,
    # which is used to calculate the representative trunk/padding values.
    return total_q, total_k, n_q_trunks, n_k_trunks


def _create_mask_config(
    params: MaskConfigParams,
) -> MaskCreationConfig:
    """
    Create a mask creation configuration.

    Args:
        params: Parameters for mask creation configuration

    Returns:
        MaskCreationConfig for creating masks
    """
    return MaskCreationConfig(
        n_queries=params.n_queries,
        n_keys=params.n_keys,
        query_lists=params.q_list,
        key_lists=params.k_list,
        query_dims=params.dim_q_list,
        key_dims=params.dim_k_list,
        n_q_chunks=params.n_q_trunks,
        n_k_chunks=params.n_k_trunks,
        q_trunk_indices=list(range(params.n_q_trunks)),
        n_q_per_chunk=params.n_queries,
        window_size=1,  # Default window size, consider making configurable if needed
        original_query_length=params.total_q,
    )


def _prepare_padding_info(
    params: PaddingInfoParams,
) -> Dict[str, Any]: # Updated return type hint
    """
    Prepare padding information for rearrangement, including masks if requested,
    and padding lengths / trunk numbers.

    Args:
        params: Parameters for padding info preparation

    Returns:
        Dict[str, Any]: Dictionary containing masks, padding lengths, and trunk numbers.
                        Keys: "q_mask", "k_mask", "mask_trunked", "q_padding",
                              "k_padding", "num_q_trunks", "num_k_trunks".
    """
    # --- Calculate representative values based on the FIRST tensor ---
    # These values are specifically for populating the returned dict correctly.
    rep_total_q = 0
    rep_total_k = 0
    rep_n_q_trunks = 0
    rep_n_k_trunks = 0

    if params.q_list:
        first_q = params.q_list[0]
        first_q_dim_idx = 0 # Index within dim_q_list
        first_q_dim = params.dim_q_list[first_q_dim_idx]
        if first_q_dim < 0: first_q_dim += first_q.ndim
        if 0 <= first_q_dim < first_q.ndim:
            rep_total_q = first_q.shape[first_q_dim]
            rep_n_q_trunks = math.ceil(rep_total_q / params.n_queries) if rep_total_q > 0 else 0

    if params.k_list:
        first_k = params.k_list[0]
        first_k_dim_idx = 0 # Index within dim_k_list
        first_k_dim = params.dim_k_list[first_k_dim_idx]
        if first_k_dim < 0: first_k_dim += first_k.ndim
        if 0 <= first_k_dim < first_k.ndim:
            rep_total_k = first_k.shape[first_k_dim]
            rep_n_k_trunks = math.ceil(rep_total_k / params.n_keys) if rep_total_k > 0 else 0

    rep_n_q_trunks = max(0, rep_n_q_trunks)
    rep_n_k_trunks = max(0, rep_n_k_trunks)

    # Calculate padding based on these representative values
    rep_q_padded_len = rep_n_q_trunks * params.n_queries
    rep_q_padding = max(0, rep_q_padded_len - rep_total_q)

    rep_k_padded_len = rep_n_k_trunks * params.n_keys
    rep_k_padding = max(0, rep_k_padded_len - rep_total_k)
    # --- End representative value calculation ---


    # Initialize mask values
    q_mask_value: Union[List[torch.Tensor], None] = None
    k_mask_value: Union[List[torch.Tensor], None] = None
    mask_trunked_value: Union[torch.Tensor, None] = None

    # Compute masks if requested (use actual trunk info if available and needed for mask logic)
    # Use the potentially different trunk numbers from params.trunk_info if provided,
    # as mask creation might depend on the overall structure.
    mask_n_q_trunks = params.trunk_info.n_q_trunks if params.trunk_info else rep_n_q_trunks
    mask_n_k_trunks = params.trunk_info.n_k_trunks if params.trunk_info else rep_n_k_trunks
    mask_total_q = params.trunk_info.total_queries if params.trunk_info else rep_total_q

    # Avoid mask creation for fully empty inputs based on representative lengths
    if params.compute_mask and (rep_total_q > 0 or rep_total_k > 0):
        # Create mask configuration parameters using potentially different trunk counts for masking logic
        mask_config_params = MaskConfigParams(
            n_queries=params.n_queries,
            n_keys=params.n_keys,
            q_list=params.q_list,
            k_list=params.k_list,
            dim_q_list=params.dim_q_list,
            dim_k_list=params.dim_k_list,
            n_q_trunks=mask_n_q_trunks, # Use potentially summed value for mask logic
            n_k_trunks=mask_n_k_trunks, # Use potentially summed value for mask logic
            total_q=mask_total_q,       # Use potentially summed value for mask logic
        )
        mask_config = _create_mask_config(mask_config_params)
        mask_slices = _create_masks(mask_config)

        if mask_slices:
            device = torch.device("cpu")
            first_valid_tensor = next((t for t in params.q_list + params.k_list if t.numel() > 0), None)
            if first_valid_tensor is not None:
                device = first_valid_tensor.device

            tensor_masks_config = TensorMasksConfig(
                n_q_trunks=mask_n_q_trunks, # Use potentially summed value for mask logic
                n_k_trunks=mask_n_k_trunks, # Use potentially summed value for mask logic
                n_queries=params.n_queries,
                n_keys=params.n_keys,
                device=device,
            )
            q_mask_value, k_mask_value, mask_trunked_value = create_tensor_masks(
                mask_slices, tensor_masks_config
            )

    # Create and return the dictionary using the REPRESENTATIVE values for padding/trunks
    return {
        "q_mask": q_mask_value,
        "k_mask": k_mask_value,
        "mask_trunked": mask_trunked_value,
        "q_padding": rep_q_padding,         # Use representative value
        "k_padding": rep_k_padding,         # Use representative value
        "num_q_trunks": rep_n_q_trunks,     # Use representative value
        "num_k_trunks": rep_n_k_trunks,     # Use representative value
    }
