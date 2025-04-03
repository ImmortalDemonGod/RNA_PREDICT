"""
Dense trunk operations module for attention mechanisms.

This module contains functions for rearranging tensors into dense trunks
for efficient attention operations. It serves as the main interface,
utilizing helper modules for specific functionalities like configuration,
chunking, padding, masking, and core processing.
"""

from typing import List, Optional, Tuple, Union

import torch

# Imports from local modules within the same package level
from ..attention_utils import PaddingInfo
from ..core_transforms import (
    TrunkInfo,
    _calculate_trunk_info,
    _create_zero_tensor,
    _handle_return_types,
    _prepare_tensor_lists,
    _validate_dimensions,
    _validate_input_types,
)

# Imports from the newly created sub-modules
from .chunk_utils import _process_keys_values_chunks
from .config_types import (
    AttentionBiasConfig,
    DenseTrunkConfig,
    KeysValuesChunkParams,
    PaddingInfoParams,
    RearrangementConfig,
)
from .masking_padding_utils import _prepare_padding_info
from .padding_reshape_utils import (
    _process_attention_bias,
    _reshape_query_to_trunk,
)
from .trunk_processing import _process_tensor_to_trunks


def rearrange_qk_to_dense_trunk(
    q: Union[torch.Tensor, List[torch.Tensor]],
    k: Union[torch.Tensor, List[torch.Tensor]],
    dim_q: Union[int, List[int]],
    dim_k: Union[int, List[int]],
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
) -> Tuple[
    Union[torch.Tensor, List[torch.Tensor]],
    Union[torch.Tensor, List[torch.Tensor]],
    PaddingInfo,
]:
    """
    Rearrange query and key tensors into dense trunks.

    Args:
        q: Query tensor or list of tensors.
        k: Key tensor or list of tensors.
        dim_q: Query dimension(s) to rearrange.
        dim_k: Key dimension(s) to rearrange.
        n_queries: Number of queries per trunk. Defaults to 32.
        n_keys: Number of keys per trunk. Defaults to 128.
        compute_mask: Whether to compute masks. Defaults to True.

    Returns:
        Rearranged query tensor(s), rearranged key tensor(s), and padding information.
    """
    # Step 1: Validate and prepare input tensors
    q_is_list, k_is_list = _validate_input_types(q, k)
    q_list, k_list = _prepare_tensor_lists(q, k, q_is_list, k_is_list)

    # Step 2: Handle dimension lists
    dim_q_list = [dim_q] if isinstance(dim_q, int) else dim_q
    dim_k_list = [dim_k] if isinstance(dim_k, int) else dim_k

    # Step 3: Validate dimensions
    _validate_dimensions(dim_q_list, dim_k_list, q_list, k_list)

    # Step 4: Calculate trunk information
    trunk_info = _calculate_trunk_info(
        q_list, k_list, dim_q_list, dim_k_list, n_queries, n_keys
    )

    # Step 5: Process queries and keys using the imported function
    q_new = _process_tensor_to_trunks(
        trunk_info.q_list, trunk_info.dim_q_list, trunk_info.n_q_trunks, n_queries
    )
    k_new = _process_tensor_to_trunks(
        trunk_info.k_list, trunk_info.dim_k_list, trunk_info.n_k_trunks, n_keys
    )

    # Step 6: Prepare padding information using the imported function
    padding_info_params = PaddingInfoParams(
        q_list=q_list,
        k_list=k_list,
        dim_q_list=dim_q_list,
        dim_k_list=dim_k_list,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
        trunk_info=trunk_info,
    )
    padding_info = _prepare_padding_info(padding_info_params)

    # Step 7: Handle return types
    q_result, k_result = _handle_return_types(q_new, k_new, q_is_list, k_is_list)

    return q_result, k_result, padding_info


def _is_small_tensor_case(
    q: torch.Tensor, k: torch.Tensor, config: DenseTrunkConfig
) -> bool:
    """
    Check if tensor dimensions are small enough to skip trunking.

    Args:
        q: Query tensor.
        k: Key tensor.
        config: Configuration for dense trunk operations.

    Returns:
        True if tensors are small enough, False otherwise.
    """
    return q.shape[-2] <= config.n_queries and k.shape[-2] <= config.n_keys


def _handle_small_tensors(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: DenseTrunkConfig
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """
    Handle small tensors where trunking is not needed.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        config: Configuration for dense trunk operations.

    Returns:
        Result tuple if tensors are small enough, None otherwise.
    """
    if not _is_small_tensor_case(q, k, config):
        return None

    # For small tensors, no need for trunking - use as-is
    q_trunked, k_trunked, v_trunked = q, k, v

    # Handle attention bias
    attn_bias_trunked = (
        config.attn_bias
        if config.attn_bias is not None
        else _create_zero_tensor([0], dtype=q.dtype, device=q.device)
    )

    return q_trunked, k_trunked, v_trunked, attn_bias_trunked, 0


def _rearrange_to_dense_trunk_impl(
    config: RearrangementConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Implementation of dense trunk rearrangement with bundled parameters.

    Args:
        config: Configuration for rearrangement.

    Returns:
        Trunked query, key, value, attention bias, and padding length.
    """
    # Create configuration object for trunk operations
    trunk_config = DenseTrunkConfig(
        n_queries=config.n_queries,
        n_keys=config.n_keys,
        attn_bias=config.attn_bias,
        inf=config.inf,
    )

    # Check for small tensor case (optimization)
    small_tensors_result = _handle_small_tensors(
        config.q, config.k, config.v, trunk_config
    )
    if small_tensors_result is not None:
        return small_tensors_result

    # Process query tensor using imported function
    query_info = _reshape_query_to_trunk(config.q, config.n_queries)
    q_trunked = query_info.trunked_tensor

    # Calculate number of key trunks
    n_k_trunks = (config.k.shape[-2] + config.n_keys - 1) // config.n_keys

    # Process key and value tensors using imported function
    k_trunked, v_trunked, attn_bias_list = _process_keys_values_chunks(
        KeysValuesChunkParams(
            k=config.k,
            v=config.v,
            n_keys=config.n_keys,
            n_k_trunks=n_k_trunks,
            attn_bias=config.attn_bias,
            inf=config.inf,
        )
    )

    # Create bias configuration
    bias_config = AttentionBiasConfig(
        n_q_trunks=query_info.num_trunks,
        n_queries=config.n_queries,
        n_q_pad=query_info.padding_length,
        original_length=config.q.shape[-2],
        inf=config.inf,
    )

    # Process attention bias using imported function
    attn_bias_trunked = _process_attention_bias(
        q_trunked, k_trunked, config.attn_bias, attn_bias_list, bias_config
    )

    return q_trunked, k_trunked, v_trunked, attn_bias_trunked, query_info.padding_length


def rearrange_to_dense_trunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
    inf: float = 1e10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Rearrange query, key, and value tensors into dense trunks for efficient attention.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        n_queries: Number of queries per trunk.
        n_keys: Number of keys per trunk.
        attn_bias: Attention bias tensor. Defaults to None.
        inf: Value for masked positions. Defaults to 1e10.

    Returns:
        Trunked query, key, value, attention bias, and padding length.
    """
    # Use config object to bundle parameters
    config = RearrangementConfig(
        q=q, k=k, v=v, n_queries=n_queries, n_keys=n_keys, attn_bias=attn_bias, inf=inf
    )
    return _rearrange_to_dense_trunk_impl(config)
