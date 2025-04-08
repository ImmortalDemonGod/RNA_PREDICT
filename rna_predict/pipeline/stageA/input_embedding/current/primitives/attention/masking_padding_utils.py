"""
Utility functions for creating masks and preparing padding information
for dense trunk attention.
"""

from typing import List, Tuple, Union

import torch

from ..attention_utils import PaddingInfo
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
        Tuple containing total_q, total_k, n_q_trunks, n_k_trunks
    """
    if params.trunk_info is None:
        # Calculate total sizes
        total_q = sum(
            q.shape[params.dim_q_list[i]] for i, q in enumerate(params.q_list)
        )
        total_k = sum(
            k.shape[params.dim_k_list[i]] for i, k in enumerate(params.k_list)
        )

        # Number of trunks needed
        n_q_trunks = (total_q + params.n_queries - 1) // params.n_queries
        n_k_trunks = (total_k + params.n_keys - 1) // params.n_keys
    else:
        total_q = params.trunk_info.total_queries
        total_k = params.trunk_info.total_keys
        n_q_trunks = params.trunk_info.n_q_trunks
        n_k_trunks = params.trunk_info.n_k_trunks

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
) -> PaddingInfo:
    """
    Prepare padding information for rearrangement, including masks if requested.

    Args:
        params: Parameters for padding info preparation

    Returns:
        PaddingInfo: Padding information containing masks.
    """
    # Calculate trunk dimensions
    dim_params = TrunkDimensionsParams(
        q_list=params.q_list,
        k_list=params.k_list,
        dim_q_list=params.dim_q_list,
        dim_k_list=params.dim_k_list,
        n_queries=params.n_queries,
        n_keys=params.n_keys,
        trunk_info=params.trunk_info,
    )
    total_q, total_k, n_q_trunks, n_k_trunks = _calculate_trunk_dimensions(dim_params)

    # Initialize padding info values
    q_mask_value: Union[List[torch.Tensor], None] = None
    k_mask_value: Union[List[torch.Tensor], None] = None
    mask_trunked_value: Union[torch.Tensor, None] = None

    # Compute masks if requested
    if params.compute_mask:
        # Create mask configuration parameters
        mask_config_params = MaskConfigParams(
            n_queries=params.n_queries,
            n_keys=params.n_keys,
            q_list=params.q_list,
            k_list=params.k_list,
            dim_q_list=params.dim_q_list,
            dim_k_list=params.dim_k_list,
            n_q_trunks=n_q_trunks,
            n_k_trunks=n_k_trunks,
            total_q=total_q,
        )

        # Create mask configuration
        mask_config = _create_mask_config(mask_config_params)

        # Create masks
        mask_slices = _create_masks(mask_config)

        # If masks were created, convert them to tensors
        if mask_slices:
            # Create tensor masks config
            tensor_masks_config = TensorMasksConfig(
                n_q_trunks=n_q_trunks,
                n_k_trunks=n_k_trunks,
                n_queries=params.n_queries,
                n_keys=params.n_keys,
                # Ensure device is correctly inferred or passed
                device=params.q_list[0].device
                if params.q_list
                else (
                    params.k_list[0].device if params.k_list else torch.device("cpu")
                ),
            )

            q_mask_value, k_mask_value, mask_trunked_value = create_tensor_masks(
                mask_slices, tensor_masks_config
            )

    # Create and return the properly typed PaddingInfo
    return PaddingInfo(
        q_mask=q_mask_value, k_mask=k_mask_value, mask_trunked=mask_trunked_value
    )
