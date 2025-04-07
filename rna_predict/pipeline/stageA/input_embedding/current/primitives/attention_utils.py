"""
Attention utilities module for neural network operations.

This module contains specialized utility functions for handling various attention
operations, such as local attention, bias creation, and optimization functions.
"""

import warnings  # <<< ADDED IMPORT
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TypedDict, Union

import torch

from .attention_base import AttentionInputs, _attention


@dataclass
class AttentionChunkConfig:
    """Configuration for chunked attention processing."""

    chunk_size: int
    n_chunks_q: int
    n_chunks_k: int


class PaddingInfo(TypedDict, total=False):
    """Type for padding information in attention operations."""

    q_mask: Union[List[torch.Tensor], None]
    k_mask: Union[List[torch.Tensor], None]
    mask_trunked: Union[torch.Tensor, None]


@dataclass
class LocalAttentionInputs:
    """Inputs for local attention."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    n_queries: int
    n_keys: int
    attn_bias: Optional[torch.Tensor] = None
    trunked_attn_bias: Optional[torch.Tensor] = None
    inf: float = 1e10
    use_efficient_implementation: bool = False
    attn_weight_dropout_p: float = 0.0
    inplace_safe: bool = False
    chunk_size: Optional[int] = None


@dataclass
class ChunkProcessingInputs:
    """Inputs for processing an attention chunk."""

    q_chunk: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    bias_slice: Optional[torch.Tensor]
    use_efficient_implementation: bool
    attn_weight_dropout_p: float
    inplace_safe: bool


@dataclass
class BiasCreationInputs:
    """Inputs for creating local attention bias."""

    n: int
    n_queries: int
    n_keys: int
    inf: float = 1e10
    device: Optional[torch.device] = None


def optimized_concat_split(attn_bias: torch.Tensor, n_queries: int) -> torch.Tensor:
    """
    Split attn_bias in an optimized manner for n_queries.

    Args:
        attn_bias (torch.Tensor): Attention bias tensor
        n_queries (int): Number of queries

    Returns:
        torch.Tensor: Optimized attention bias
    """
    n_q = attn_bias.shape[-2]
    chunks = []

    # Optimize by processing n_queries chunks
    for i in range(0, n_q, n_queries):
        chunk = attn_bias[..., i : i + n_queries, :]
        chunks.append(chunk)

    return torch.cat(chunks, dim=-3)


def _calculate_bias_shape(inputs: BiasCreationInputs) -> Tuple[int, int, int, int]:
    """
    Calculate bias shape for local attention.

    Args:
        inputs (BiasCreationInputs): Input parameters

    Returns:
        Tuple[int, int, int, int]: Calculated bias shape
    """
    n_chunks = (inputs.n + inputs.n_queries - 1) // inputs.n_queries
    return (1, n_chunks, inputs.n_queries, inputs.n_keys)


def create_local_attn_bias(inputs: BiasCreationInputs) -> torch.Tensor:
    """
    Create local attention bias tensor.

    Args:
        inputs (BiasCreationInputs): Input parameters

    Returns:
        torch.Tensor: Local attention bias tensor
    """
    # Calculate parameters
    bias_shape = _calculate_bias_shape(inputs)

    # Initialize bias tensor
    device = inputs.device if inputs.device is not None else torch.device("cpu")
    attn_bias = torch.full(bias_shape, -inputs.inf, device=device)

    # Set valid attention regions
    n_chunks = bias_shape[1]
    for i in range(n_chunks):
        q_start = i * inputs.n_queries
        q_end = min(q_start + inputs.n_queries, inputs.n)

        if q_start >= inputs.n:
            continue

        start_idx = max(0, q_start - inputs.n_queries)
        end_idx = min(inputs.n, q_start + 2 * inputs.n_queries)

        # Determine the actual window width
        window_width = min(inputs.n_keys, end_idx - start_idx)

        # Set the bias to 0 for the valid window
        attn_bias[0, i, : q_end - q_start, :window_width] = 0

    return attn_bias


def _determine_chunking(
    q: torch.Tensor, n_queries: int, n_keys: int, chunk_size: Optional[int]
) -> AttentionChunkConfig:
    """
    Determine chunking configuration for local attention.

    Args:
        q (torch.Tensor): Query tensor
        n_queries (int): Number of queries per chunk
        n_keys (int): Number of keys per chunk
        chunk_size (int, optional): Override chunk size if provided

    Returns:
        AttentionChunkConfig: Chunking configuration
    """
    # Use provided chunk size or calculate based on tensor size
    if chunk_size is not None:
        actual_chunk_size = chunk_size
    else:
        # Heuristic to determine chunk size based on tensor dimensions
        # Smaller tensors can use larger chunks
        if (
            q.shape[-4] <= 32
        ):  # Small batch size (Corrected condition to include 32) # CORRECTED INDEX -4
            actual_chunk_size = min(128, q.shape[-2])
        else:
            actual_chunk_size = min(64, q.shape[-2])

    # Calculate number of chunks
    n_chunks_q = (q.shape[-2] + actual_chunk_size - 1) // actual_chunk_size
    n_chunks_k = (q.shape[-2] + actual_chunk_size - 1) // actual_chunk_size

    return AttentionChunkConfig(
        chunk_size=actual_chunk_size, n_chunks_q=n_chunks_q, n_chunks_k=n_chunks_k
    )


def _process_attention_chunk(inputs: ChunkProcessingInputs) -> torch.Tensor:
    """
    Process a single chunk for chunked attention.

    Args:
        inputs (ChunkProcessingInputs): Input parameters for chunk processing

    Returns:
        torch.Tensor: Processed attention output
    """
    # Adapt the bias slice if it comes from a trunked bias
    processed_bias_slice = inputs.bias_slice
    if processed_bias_slice is not None and processed_bias_slice.ndim == 5:
        # Original shape: [..., n_heads, 1, n_queries, n_keys]
        # Expected shape by _attention: [..., n_heads, n_queries_chunk, n_keys]
        # We remove the singleton dimension corresponding to the chunk index
        processed_bias_slice = processed_bias_slice.squeeze(-3)
        # Now shape is [..., n_heads, n_queries, n_keys]
        # We might need to further slice if n_queries_chunk < n_queries
        n_queries_chunk = inputs.q_chunk.shape[
            -2
        ]  # Get the actual query chunk size # CORRECTED INDEX -2
        if processed_bias_slice.shape[-2] != n_queries_chunk:
            # Slice to match the actual query chunk dimension
            processed_bias_slice = processed_bias_slice[..., :n_queries_chunk, :]

    # Convert to AttentionInputs for compatibility with the new _attention signature
    attention_inputs = AttentionInputs(
        q=inputs.q_chunk,
        k=inputs.k,
        v=inputs.v,
        attn_bias=processed_bias_slice,  # Use the potentially reshaped bias
        use_efficient_implementation=inputs.use_efficient_implementation,
        attn_weight_dropout_p=inputs.attn_weight_dropout_p,
        inplace_safe=inputs.inplace_safe,
    )

    # Process chunk with regular attention
    return _attention(attention_inputs)


def _get_attention_bias(
    inputs: LocalAttentionInputs, q_shape: Tuple[int, ...]
) -> Optional[torch.Tensor]:
    """
    Get appropriate attention bias based on inputs.

    Args:
        inputs (LocalAttentionInputs): Input parameters
        q_shape (Tuple[int, ...]): Shape of query tensor

    Returns:
        Optional[torch.Tensor]: Attention bias tensor or None
    """
    if inputs.trunked_attn_bias is not None:
        return inputs.trunked_attn_bias

    if inputs.attn_bias is not None:
        return inputs.attn_bias

    # Create local attention bias for windowed attention
    bias_inputs = BiasCreationInputs(
        n=q_shape[-2],
        n_queries=inputs.n_queries,
        n_keys=inputs.n_keys,
        inf=inputs.inf,
        device=inputs.q.device,
    )
    return create_local_attn_bias(bias_inputs)


def _process_small_tensors(inputs: LocalAttentionInputs) -> Optional[torch.Tensor]:
    """
    Process small tensors directly without chunking.

    Args:
        inputs (LocalAttentionInputs): Input parameters

    Returns:
        Optional[torch.Tensor]: Processed attention output or None if not applicable
    """
    # Check if tensors are small enough to process directly
    if inputs.q.shape[-2] <= inputs.n_queries and inputs.k.shape[-2] <= inputs.n_keys:
        # --- Start Fix: Select the correct bias ---
        # Prioritize attn_bias, fall back to trunked_attn_bias
        bias_to_process = inputs.attn_bias
        if bias_to_process is None:
            bias_to_process = inputs.trunked_attn_bias
            # If trunked_attn_bias is used, it might be 5D [..., H, B, Q, K]
            # We need to adapt it for the expected 4D [..., H, Q, K] in _attention
            if bias_to_process is not None and bias_to_process.ndim == 5:
                # Assuming the extra dim is the block dim at index -3
                # Squeeze it out if it's size 1, otherwise raise error or warn
                if bias_to_process.shape[-3] == 1:
                    bias_to_process = bias_to_process.squeeze(-3)
                else:
                    # This case is ambiguous, maybe log and proceed without bias
                    warnings.warn(
                        f"Trunked bias has unexpected 5D shape {bias_to_process.shape} "
                        f"with block dim != 1. Cannot safely adapt for small tensor processing. Skipping bias."
                    )
                    bias_to_process = None

        processed_bias = None  # Initialize bias to pass to _attention as None
        if bias_to_process is not None:
            # Apply size check and reshape logic to the selected bias
            try:
                # --- Start Mypy Fix ---
                assert bias_to_process is not None  # Add assertion for type checker
                # --- End Mypy Fix ---
                expected_size = inputs.q.shape[-2] * inputs.k.shape[-2]
                actual_size = bias_to_process.numel()

                # Check if the total number of elements matches.
                # This is less strict than exact shape match and allows for broadcasting.
                if expected_size == actual_size:
                    # Reshape if the total element count matches
                    # Target shape for bias in _attention is typically [..., H, N_q, N_kv]
                    # We need to infer H (num_heads) if possible, or assume broadcasting works.
                    # Let's try reshaping to the direct query/key dims first.
                    target_bias_shape = (
                        *bias_to_process.shape[:-2],
                        inputs.q.shape[-2],
                        inputs.k.shape[-2],
                    )
                    processed_bias = bias_to_process.reshape(target_bias_shape)

                # If sizes don't match, broadcasting might still work, but let's warn and proceed without bias
                # as the original code did, to maintain stability for now.
                # A more robust solution might try broadcasting directly in _attention.
                elif expected_size != actual_size:
                    print(  # Keep the print for debugging visibility
                        f"Warning: Selected bias shape mismatch in _process_small_tensors. "
                        f"Expected size: {expected_size}, actual size: {actual_size} (from {bias_to_process.shape}). "
                        f"Skipping bias application for stability."
                    )
                    processed_bias = None  # Explicitly set to None if check fails

            except (RuntimeError, ValueError) as e:
                # If reshaping fails, skip using the bias
                # --- Start Mypy Fix ---
                # Ensure bias_to_process is not None before accessing shape in error message
                bias_shape = (
                    bias_to_process.shape
                    if bias_to_process is not None
                    else "Unknown (None)"
                )
                # --- End Mypy Fix ---
                print(
                    f"Warning: Couldn't reshape selected bias from {bias_shape} to match query/key dimensions. "
                    f"q shape: {inputs.q.shape}, k shape: {inputs.k.shape}. Error: {e}. "
                    f"Skipping bias application for stability."
                )
                processed_bias = None
        # --- End Fix ---

        # Convert to AttentionInputs for compatibility
        attention_inputs = AttentionInputs(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            attn_bias=processed_bias,  # Pass the potentially reshaped or None bias
            use_efficient_implementation=inputs.use_efficient_implementation,
            attn_weight_dropout_p=inputs.attn_weight_dropout_p,
            inplace_safe=inputs.inplace_safe,
        )

        return _attention(attention_inputs)

    return None


def _get_bias_slice(
    local_bias: Optional[torch.Tensor], chunk_idx: int, start_idx: int, end_idx: int
) -> Optional[torch.Tensor]:
    """
    Get the appropriate bias slice for a chunk.

    Args:
        local_bias (Optional[torch.Tensor]): Attention bias
        chunk_idx (int): Current chunk index
        start_idx (int): Start index of the chunk
        end_idx (int): End index of the chunk

    Returns:
        Optional[torch.Tensor]: Bias slice for the chunk
    """
    if local_bias is None:
        return None

    # Extract appropriate slice of bias
    if local_bias.ndim == 5:  # For trunked bias (5D) # CORRECTED CONDITION
        return local_bias[..., chunk_idx : chunk_idx + 1, :, :]
    else:  # For regular bias (e.g., 4D)
        return local_bias[..., start_idx:end_idx, :]


def _process_chunks(
    inputs: LocalAttentionInputs,
    chunking_config: AttentionChunkConfig,
    local_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Process attention in chunks.

    Args:
        inputs (LocalAttentionInputs): Input parameters
        chunking_config (AttentionChunkConfig): Chunking configuration
        local_bias (Optional[torch.Tensor]): Attention bias

    Returns:
        torch.Tensor: Processed attention output
    """
    # Initialize output tensor
    o = torch.zeros_like(inputs.q)

    # Process each chunk of queries
    for i in range(chunking_config.n_chunks_q):
        # Calculate chunk boundaries
        start_idx = i * chunking_config.chunk_size
        end_idx = min(start_idx + chunking_config.chunk_size, inputs.q.shape[-2])

        if start_idx >= inputs.q.shape[-2]:
            continue

        # Extract query chunk
        q_chunk = inputs.q[..., start_idx:end_idx, :, :]

        # Get bias slice for this chunk
        bias_slice = _get_bias_slice(local_bias, i, start_idx, end_idx)

        # Process chunk
        chunk_inputs = ChunkProcessingInputs(
            q_chunk=q_chunk,
            k=inputs.k,
            v=inputs.v,
            bias_slice=bias_slice,
            use_efficient_implementation=inputs.use_efficient_implementation,
            attn_weight_dropout_p=inputs.attn_weight_dropout_p,
            inplace_safe=inputs.inplace_safe,
        )

        o_chunk = _process_attention_chunk(chunk_inputs)

        # Store result
        o[..., start_idx:end_idx, :, :] = o_chunk

    return o


def _local_attention(inputs: LocalAttentionInputs) -> torch.Tensor:
    """
    Local attention implementation with support for chunking and optimizations.

    Args:
        inputs (LocalAttentionInputs): Input parameters including query, key, value tensors,
                                      and configuration options

    Returns:
        torch.Tensor: Local attention output
    """
    # Early validation and checks
    if inputs.q.shape[-2] == 0:
        return inputs.q

    assert inputs.n_keys >= inputs.n_queries, (
        f"n_keys ({inputs.n_keys}) must be >= n_queries ({inputs.n_queries})"
    )

    # Try processing small tensors directly
    small_tensor_result = _process_small_tensors(inputs)
    if small_tensor_result is not None:
        return small_tensor_result

    # Calculate chunking configuration
    chunking_config = _determine_chunking(
        inputs.q, inputs.n_queries, inputs.n_keys, inputs.chunk_size
    )

    # Choose processing strategy based on tensor shape and configuration
    if inputs.q.ndim >= 3 and chunking_config.n_chunks_q > 1:
        # Use chunked processing with local attention bias
        local_bias = _get_attention_bias(inputs, inputs.q.shape)
        return _process_chunks(inputs, chunking_config, local_bias)

    # Fallback to regular attention for non-chunked tensors
    attention_inputs = AttentionInputs(
        q=inputs.q,
        k=inputs.k,
        v=inputs.v,
        attn_bias=inputs.attn_bias,
        use_efficient_implementation=inputs.use_efficient_implementation,
        attn_weight_dropout_p=inputs.attn_weight_dropout_p,
        inplace_safe=inputs.inplace_safe,
    )

    return _attention(attention_inputs)


# Public API functions
# These are the functions that should be called by other modules


def local_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    """
    Local attention implementation with support for chunking and optimizations.
    This is the public API function that should be used by other modules.

    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        **kwargs: Additional keyword arguments:
            n_queries (int): Number of queries per chunk
            n_keys (int): Number of keys per chunk
            attn_bias (torch.Tensor, optional): Attention bias
            trunked_attn_bias (torch.Tensor, optional): Pre-computed trunked attention bias
            inf (float, optional): Value for masked positions
            use_efficient_implementation (bool, optional): Whether to use efficient implementation
            attn_weight_dropout_p (float, optional): Dropout probability
            inplace_safe (bool, optional): Whether it's safe to use inplace operations
            chunk_size (int, optional): Override chunk size for processing

    Returns:
        torch.Tensor: Local attention output
    """
    # Set default values for required arguments
    n_queries = kwargs.get("n_queries", q.shape[-2])
    n_keys = kwargs.get("n_keys", k.shape[-2])

    # Create input dataclass
    inputs = LocalAttentionInputs(
        q=q,
        k=k,
        v=v,
        n_queries=n_queries,
        n_keys=n_keys,
        attn_bias=kwargs.get("attn_bias"),
        trunked_attn_bias=kwargs.get("trunked_attn_bias"),
        inf=kwargs.get("inf", 1e10),
        use_efficient_implementation=kwargs.get("use_efficient_implementation", False),
        attn_weight_dropout_p=kwargs.get("attn_weight_dropout_p", 0.0),
        inplace_safe=kwargs.get("inplace_safe", False),
        chunk_size=kwargs.get("chunk_size"),
    )

    return _local_attention(inputs)
