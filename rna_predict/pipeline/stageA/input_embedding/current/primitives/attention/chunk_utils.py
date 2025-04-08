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

    # Check for zero sequence length early
    k_seq_len = params.k.shape[-2] if params.k.ndim >= 2 else 0
    v_seq_len = params.v.shape[-2] if params.v.ndim >= 2 else 0 # Assuming V follows K shape convention

    # If K sequence length is 0, no trunks are processed.
    if k_seq_len == 0:
        # Create empty tensors with the expected output rank and dimensions,
        # but with the trunk dimension size set to 0.
        # Expected output shape: (B, ..., n_k_trunks=0, n_keys, D)
        k_shape = list(params.k.shape)
        v_shape = list(params.v.shape)

        # Assume B is dim 0, S is dim -2, D is dim -1
        # Target shape: [B] + [...] + [0, n_keys, D]
        k_empty_shape = k_shape[:-2] + [0, params.n_keys, k_shape[-1]]
        v_empty_shape = v_shape[:-2] + [0, params.n_keys, v_shape[-1]]

        k_trunked = torch.empty(k_empty_shape, dtype=params.k.dtype, device=params.k.device)
        v_trunked = torch.empty(v_empty_shape, dtype=params.v.dtype, device=params.v.device)

        # Bias list remains empty
        attn_bias_output = None # Or empty list if downstream expects list? Let's return None.

    else:
        # Process each key/value trunk (original logic)
        for i in range(params.n_k_trunks):
            # Get chunk information
            chunk_info = _get_chunk_info(i, params.n_keys, k_seq_len) # Use k_seq_len

            # Process key chunk
            k_chunk = _process_chunk(
                params.k, chunk_info, params.n_keys, params.k.dtype, params.k.device
            )
            k_list.append(k_chunk)

            # Process value chunk - use same chunk_info based on K
            # Need to handle potential V length mismatch if logic allows it
            # Assuming V length matches K length for chunking purposes here
            v_chunk_info = _get_chunk_info(i, params.n_keys, v_seq_len) # Use v_seq_len for slicing V
            v_chunk = _process_chunk(
                 params.v, v_chunk_info, params.n_keys, params.v.dtype, params.v.device
            )
            # Ensure v_chunk has the same sequence length as k_chunk after padding
            # This might require adjusting the padding logic if V len differs significantly
            # For now, assume padding handles it based on params.n_keys.
            v_list.append(v_chunk)

            # Process bias chunk if provided
            if params.attn_bias is not None:
                # Bias chunking depends on K dimension
                bias_chunk = _process_bias_chunk(
                    params.attn_bias,
                    chunk_info, # Use K's chunk_info for bias slicing along K dim
                    params.n_keys,
                    params.inf,
                    params.attn_bias.device,
                )
                attn_bias_list.append(bias_chunk)

        # Stack non-empty chunks
        k_trunked = torch.stack(k_list, dim=-3)
        v_trunked = torch.stack(v_list, dim=-3)
        attn_bias_output = attn_bias_list if params.attn_bias is not None else None


    return (
        k_trunked,
        v_trunked,
        attn_bias_output,
    )
