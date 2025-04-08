"""
Utility functions for processing tensor chunks in dense trunk attention.
"""

from typing import List, Optional, Tuple

import torch

from .config_types import ChunkInfo, KeysValuesChunkParams, TensorChunkParams


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
    Process a single tensor chunk using zero padding.

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
    Process a bias tensor chunk using -inf padding.

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
        dim=-1,  # Bias padding is typically on the last dimension
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
