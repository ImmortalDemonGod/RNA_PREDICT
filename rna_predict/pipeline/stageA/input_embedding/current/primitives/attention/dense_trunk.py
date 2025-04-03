"""
Dense trunk operations module for attention mechanisms.

This module contains functions for rearranging tensors into dense trunks
for efficient attention operations.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from ..attention_utils import PaddingInfo
from ..core_transforms import (
    TrunkInfo,
    _apply_trunk_slices,
    _calculate_trunk_info,
    _create_zero_tensor,
    _handle_return_types,
    _prepare_tensor_lists,
    _validate_dimensions,
    _validate_input_types,
)
from .mask_operations import (
    MaskCreationConfig,
    TensorMasksConfig,
    _create_masks,
    create_tensor_masks,
)


@dataclass
class DenseTrunkConfig:
    """Configuration for dense trunk operations."""

    n_queries: int
    n_keys: int
    attn_bias: Optional[torch.Tensor] = None
    inf: float = 1e10
    compute_mask: bool = True


@dataclass
class AttentionBiasConfig:
    """Configuration for attention bias processing."""

    n_q_trunks: int
    n_queries: int
    n_q_pad: int
    original_length: int
    inf: float = 1e10


@dataclass
class QueryTrunkInfo:
    """Information about query trunk processing."""

    trunked_tensor: torch.Tensor
    padding_length: int
    total_length: int
    num_trunks: int


@dataclass
class ChunkInfo:
    """Information about a tensor chunk."""

    start: int
    end: int
    length: int


@dataclass
class RearrangementConfig:
    """Configuration for tensor rearrangement operations."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    n_queries: int
    n_keys: int
    attn_bias: Optional[torch.Tensor] = None
    inf: float = 1e10


@dataclass
class MaskConfigParams:
    """Configuration parameters for creating a mask configuration."""

    n_queries: int
    n_keys: int
    q_list: List[torch.Tensor]
    k_list: List[torch.Tensor]
    dim_q_list: List[int]
    dim_k_list: List[int]
    n_q_trunks: int
    n_k_trunks: int
    total_q: int


@dataclass
class TrunkDimensionsParams:
    """Parameters for calculating trunk dimensions."""

    q_list: List[torch.Tensor]
    k_list: List[torch.Tensor]
    dim_q_list: List[int]
    dim_k_list: List[int]
    n_queries: int
    n_keys: int
    trunk_info: Optional[TrunkInfo] = None


@dataclass
class PaddingInfoParams:
    """Parameters for preparing padding information."""

    q_list: List[torch.Tensor]
    k_list: List[torch.Tensor]
    dim_q_list: List[int]
    dim_k_list: List[int]
    n_queries: int
    n_keys: int
    compute_mask: bool
    trunk_info: Optional[TrunkInfo] = None


@dataclass
class TensorChunkParams:
    """Parameters for processing a tensor chunk."""

    tensor: torch.Tensor
    chunk_info: ChunkInfo
    chunk_size: int
    padding_value: Union[float, int]
    device: torch.device
    fill_value_func: Callable[[Any], torch.Tensor]
    dim: int = -2


@dataclass
class KeysValuesChunkParams:
    """Parameters for processing keys and values into chunks."""

    k: torch.Tensor
    v: torch.Tensor
    n_keys: int
    n_k_trunks: int
    attn_bias: Optional[torch.Tensor] = None
    inf: float = 1e10


def _process_tensor_to_trunks(
    tensor_list: List[torch.Tensor],
    dim_list: List[int],
    n_trunks: int,
    items_per_trunk: int,
) -> List[torch.Tensor]:
    """
    Process tensors into trunks with a common implementation.

    Args:
        tensor_list: List of tensors to process
        dim_list: List of dimensions to process
        n_trunks: Number of trunks
        items_per_trunk: Number of items per trunk

    Returns:
        List of processed tensor trunks
    """
    result = []

    # Process each trunk
    for trunk_idx in range(n_trunks):
        # Apply trunk slices
        sliced_tensors = _apply_trunk_slices(
            tensor_list, dim_list, trunk_idx, items_per_trunk
        )
        # Concatenate slices
        result.append(torch.cat(sliced_tensors, dim=dim_list[0]))

    return result


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
        window_size=1,  # Default window size
        original_query_length=params.total_q,
    )


def _prepare_padding_info(
    params: PaddingInfoParams,
) -> PaddingInfo:
    """
    Prepare padding information for rearrangement.

    Args:
        params: Parameters for padding info preparation

    Returns:
        PaddingInfo: Padding information
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
                device=params.q_list[0].device,
            )

            q_mask_value, k_mask_value, mask_trunked_value = create_tensor_masks(
                mask_slices, tensor_masks_config
            )

    # Create and return the properly typed PaddingInfo
    return PaddingInfo(
        q_mask=q_mask_value, k_mask=k_mask_value, mask_trunked=mask_trunked_value
    )


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
        q (Union[torch.Tensor, List[torch.Tensor]]): Query tensor or list of tensors
        k (Union[torch.Tensor, List[torch.Tensor]]): Key tensor or list of tensors
        dim_q (Union[int, List[int]]): Query dimension(s) to rearrange
        dim_k (Union[int, List[int]]): Key dimension(s) to rearrange
        n_queries (int, optional): Number of queries per trunk. Defaults to 32.
        n_keys (int, optional): Number of keys per trunk. Defaults to 128.
        compute_mask (bool, optional): Whether to compute masks. Defaults to True.

    Returns:
        Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]], PaddingInfo]:
            Rearranged query tensor(s), rearranged key tensor(s), and padding information
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

    # Step 5: Process queries and keys
    q_new = _process_tensor_to_trunks(
        trunk_info.q_list, trunk_info.dim_q_list, trunk_info.n_q_trunks, n_queries
    )
    k_new = _process_tensor_to_trunks(
        trunk_info.k_list, trunk_info.dim_k_list, trunk_info.n_k_trunks, n_keys
    )

    # Step 6: Prepare padding information
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
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        config (DenseTrunkConfig): Configuration for dense trunk operations

    Returns:
        bool: True if tensors are small enough, False otherwise
    """
    return q.shape[-2] <= config.n_queries and k.shape[-2] <= config.n_keys


def _handle_small_tensors(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: DenseTrunkConfig
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """
    Handle small tensors where trunking is not needed.

    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        config (DenseTrunkConfig): Configuration for dense trunk operations

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
            Result tuple if tensors are small enough, None otherwise
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


def _calculate_padding_needed(n: int, n_queries: int) -> int:
    """
    Calculate padding needed for the query dimension.

    Args:
        n (int): Original query dimension size
        n_queries (int): Number of queries per trunk

    Returns:
        int: Padding length
    """
    return (n_queries - (n % n_queries)) % n_queries


def _create_padding_tensor(
    original_tensor: torch.Tensor, padding_length: int, padding_dim: int = -2
) -> torch.Tensor:
    """
    Create padding tensor for the query.

    Args:
        original_tensor (torch.Tensor): Original tensor to match
        padding_length (int): Length of padding
        padding_dim (int): Dimension to pad

    Returns:
        torch.Tensor: Padding tensor
    """
    if padding_length <= 0:
        return original_tensor

    padding_shape = list(original_tensor.shape)
    padding_shape[padding_dim] = padding_length

    return torch.zeros(
        padding_shape, dtype=original_tensor.dtype, device=original_tensor.device
    )


def _reshape_query_to_trunk(q: torch.Tensor, n_queries: int) -> QueryTrunkInfo:
    """
    Reshape query tensor into trunks with padding if needed.

    Args:
        q (torch.Tensor): Query tensor
        n_queries (int): Number of queries per trunk

    Returns:
        QueryTrunkInfo: Information about processed query tensor
    """
    # Calculate padding needed
    n = q.shape[-2]
    padding_length = _calculate_padding_needed(n, n_queries)

    # Calculate derived values
    total_length = n + padding_length
    num_trunks = total_length // n_queries

    # Create padded tensor if needed
    if padding_length > 0:
        padding = _create_padding_tensor(q, padding_length)
        q = torch.cat([q, padding], dim=-2)

    # Reshape tensor into trunks
    q_shape = list(q.shape)
    new_shape = q_shape[:-2] + [num_trunks, n_queries] + q_shape[-1:]
    q_trunked = q.reshape(*new_shape)

    return QueryTrunkInfo(
        trunked_tensor=q_trunked,
        padding_length=padding_length,
        total_length=total_length,
        num_trunks=num_trunks,
    )


def _get_chunk_info(index: int, chunk_size: int, total_size: int) -> ChunkInfo:
    """
    Get information about a specific chunk.

    Args:
        index (int): Chunk index
        chunk_size (int): Size of each chunk
        total_size (int): Total size of the tensor

    Returns:
        ChunkInfo: Information about the chunk
    """
    start = index * chunk_size
    end = min(start + chunk_size, total_size)
    length = end - start

    return ChunkInfo(start=start, end=end, length=length)


def _process_tensor_chunk(
    params: TensorChunkParams,
) -> torch.Tensor:
    """
    Process a tensor chunk with customizable padding.

    Args:
        params: Parameters for tensor chunk processing

    Returns:
        Processed chunk with padding if needed
    """
    # Extract chunk - determine which dimension to slice
    if params.dim == -2:
        chunk = params.tensor[..., params.chunk_info.start : params.chunk_info.end, :]
    else:
        chunk = params.tensor[..., params.chunk_info.start : params.chunk_info.end]

    # Add padding if needed
    if params.chunk_info.length < params.chunk_size:
        pad_shape = list(chunk.shape)
        pad_shape[params.dim] = params.chunk_size - params.chunk_info.length
        padding = params.fill_value_func(pad_shape)
        chunk = torch.cat([chunk, padding], dim=params.dim)

    return chunk


def _process_chunk(
    tensor: torch.Tensor,
    chunk_info: ChunkInfo,
    chunk_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Process a single tensor chunk.

    Args:
        tensor (torch.Tensor): Input tensor
        chunk_info (ChunkInfo): Information about the chunk
        chunk_size (int): Size of each chunk
        dtype (torch.dtype): Data type for padding
        device (torch.device): Device for padding

    Returns:
        torch.Tensor: Processed chunk with padding if needed
    """

    # Use zero padding for regular tensors
    def zeros_func(shape):
        return torch.zeros(shape, dtype=dtype, device=device)

    chunk_params = TensorChunkParams(
        tensor=tensor,
        chunk_info=chunk_info,
        chunk_size=chunk_size,
        padding_value=0,
        device=device,
        fill_value_func=zeros_func,
        dim=-2,
    )
    return _process_tensor_chunk(chunk_params)


def _process_bias_chunk(
    bias: torch.Tensor,
    chunk_info: ChunkInfo,
    chunk_size: int,
    inf: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Process a bias tensor chunk.

    Args:
        bias (torch.Tensor): Input bias tensor
        chunk_info (ChunkInfo): Information about the chunk
        chunk_size (int): Size of each chunk
        inf (float): Value for masked positions
        device (torch.device): Device for padding

    Returns:
        torch.Tensor: Processed bias chunk with padding if needed
    """

    # Use -inf padding for bias tensors
    def inf_func(shape):
        return torch.full(shape, -inf, device=device)

    chunk_params = TensorChunkParams(
        tensor=bias,
        chunk_info=chunk_info,
        chunk_size=chunk_size,
        padding_value=-inf,
        device=device,
        fill_value_func=inf_func,
        dim=-1,
    )
    return _process_tensor_chunk(chunk_params)


def _process_keys_values_chunks(
    params: KeysValuesChunkParams,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
    """
    Process keys and values into trunked chunks.

    Args:
        params: KeysValuesChunkParams containing all necessary parameters.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
            Trunked key tensor, trunked value tensor, and attention bias chunks if provided
    """
    k_list = []
    v_list = []
    attn_bias_list = []

    # Process each key/value trunk
    for i in range(params.n_k_trunks):
        # Get chunk information
        chunk_info = _get_chunk_info(i, params.n_keys, params.k.shape[-2])

        # Process key chunk
        k_chunk = _process_chunk(
            params.k, chunk_info, params.n_keys, params.k.dtype, params.k.device
        )
        k_list.append(k_chunk)

        # Process value chunk
        v_chunk = _process_chunk(
            params.v, chunk_info, params.n_keys, params.v.dtype, params.v.device
        )
        v_list.append(v_chunk)

        # Process bias chunk if provided
        if params.attn_bias is not None:
            bias_chunk = _process_bias_chunk(
                params.attn_bias,
                chunk_info,
                params.n_keys,
                params.inf,
                params.attn_bias.device,
            )
            attn_bias_list.append(bias_chunk)

    # Stack chunks
    k_trunked = torch.stack(k_list, dim=-3)
    v_trunked = torch.stack(v_list, dim=-3)

    return (
        k_trunked,
        v_trunked,
        attn_bias_list if params.attn_bias is not None else None,
    )


def _pad_attention_bias(
    attn_bias: torch.Tensor, config: AttentionBiasConfig
) -> torch.Tensor:
    """
    Pad attention bias if needed.

    Args:
        attn_bias (torch.Tensor): Attention bias tensor
        config (AttentionBiasConfig): Configuration for attention bias

    Returns:
        torch.Tensor: Padded attention bias
    """
    if config.n_q_pad <= 0:
        return attn_bias

    # Create padding for attention bias
    bias_pad_shape = list(attn_bias.shape)
    bias_pad_shape[-2] = config.n_q_pad
    bias_padding = torch.full(bias_pad_shape, -config.inf, device=attn_bias.device)

    # Concatenate with original bias
    return torch.cat([attn_bias, bias_padding], dim=-2)


def _reshape_bias_for_trunked_query(
    attn_bias: torch.Tensor, config: AttentionBiasConfig
) -> torch.Tensor:
    """
    Reshape attention bias to match trunked query.

    Args:
        attn_bias (torch.Tensor): Attention bias tensor
        config (AttentionBiasConfig): Configuration for attention bias

    Returns:
        torch.Tensor: Reshaped attention bias
    """
    # Pad bias if needed
    padded_bias = _pad_attention_bias(attn_bias, config)

    # Reshape to match trunked query
    bias_shape = list(padded_bias.shape)

    # Replace the last-2 dimension with (n_q_trunks, n_queries)
    new_bias_shape = (
        bias_shape[:-2] + [config.n_q_trunks, config.n_queries] + bias_shape[-1:]
    )

    return padded_bias.reshape(*new_bias_shape)


def _create_different_dim_bias(
    q_trunked: torch.Tensor, k_trunked: torch.Tensor, config: AttentionBiasConfig
) -> torch.Tensor:
    """
    Create attention bias for the case when bias has different dimensions.

    Args:
        q_trunked (torch.Tensor): Trunked query tensor
        k_trunked (torch.Tensor): Trunked key tensor
        config (AttentionBiasConfig): Configuration for attention bias

    Returns:
        torch.Tensor: Created attention bias
    """
    # Initialize attn_bias_trunked with zeros
    attn_bias_shape = list(q_trunked.shape[:-1]) + [
        k_trunked.shape[-3],
        k_trunked.shape[-2],
    ]

    attn_bias_trunked = torch.zeros(attn_bias_shape, device=q_trunked.device)

    # Set valid attention regions
    for i in range(config.n_q_trunks):
        q_start = i * config.n_queries
        q_end = min(q_start + config.n_queries, config.original_length)

        # Skip if outside original length
        if q_start >= config.original_length:
            continue

        # Set valid attention region to zeros (allows attention)
        if q_end > q_start:
            attn_bias_trunked[..., i, : q_end - q_start, :, :] = 0

    return attn_bias_trunked


def _process_attention_bias(
    q_trunked: torch.Tensor,
    k_trunked: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    attn_bias_list: Optional[List[torch.Tensor]],
    config: AttentionBiasConfig,
) -> torch.Tensor:
    """
    Process attention bias for trunked tensors.

    Args:
        q_trunked (torch.Tensor): Trunked query tensor
        k_trunked (torch.Tensor): Trunked key tensor
        attn_bias (Optional[torch.Tensor]): Original attention bias tensor
        attn_bias_list (Optional[List[torch.Tensor]]): List of processed bias chunks
        config (AttentionBiasConfig): Configuration for bias processing

    Returns:
        torch.Tensor: Processed attention bias
    """
    # No bias case
    if attn_bias is None:
        return torch.zeros((0,), device=q_trunked.device)

    # Determine if dimensions match trunked query
    matches_shape = attn_bias.ndim == q_trunked.ndim - 1

    if matches_shape:
        # Process bias that matches query dimensions
        attn_bias_trunked = _reshape_bias_for_trunked_query(attn_bias, config)

        # Stack with key trunks if available
        if attn_bias_list is not None:
            attn_bias_trunked = torch.stack(attn_bias_list, dim=-2)
    else:
        # Handle case with different dimensions
        attn_bias_trunked = _create_different_dim_bias(q_trunked, k_trunked, config)

    return attn_bias_trunked


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
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        n_queries (int): Number of queries per trunk
        n_keys (int): Number of keys per trunk
        attn_bias (torch.Tensor, optional): Attention bias tensor. Defaults to None.
        inf (float, optional): Value for masked positions. Defaults to 1e10.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            Trunked query, key, value, attention bias, and padding length
    """
    # Use config object to bundle parameters
    config = RearrangementConfig(
        q=q, k=k, v=v, n_queries=n_queries, n_keys=n_keys, attn_bias=attn_bias, inf=inf
    )
    return _rearrange_to_dense_trunk_impl(config)


def _rearrange_to_dense_trunk_impl(
    config: RearrangementConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Implementation of dense trunk rearrangement with bundled parameters.

    Args:
        config (RearrangementConfig): Configuration for rearrangement

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            Trunked query, key, value, attention bias, and padding length
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

    # Process query tensor
    query_info = _reshape_query_to_trunk(config.q, config.n_queries)
    q_trunked = query_info.trunked_tensor

    # Calculate number of key trunks
    n_k_trunks = (config.k.shape[-2] + config.n_keys - 1) // config.n_keys

    # Process key and value tensors
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

    # Process attention bias
    attn_bias_trunked = _process_attention_bias(
        q_trunked, k_trunked, config.attn_bias, attn_bias_list, bias_config
    )

    return q_trunked, k_trunked, v_trunked, attn_bias_trunked, query_info.padding_length
