"""
Tensor processing functions for attention operations.

This module contains functions for processing tensors in attention operations.
"""

import warnings
from typing import Optional, Tuple

import torch

from .attention_core import AttentionInputs, attention as _attention
from .attention_types import AttentionChunkConfig, ChunkProcessingInputs, LocalAttentionInputs
from .attention_bias import (
    _select_attention_bias, _reshape_bias_tensor, _apply_fallback_bias_adjustment,
    _fix_dimension_mismatch, _get_bias_slice
)


def _determine_chunking(
    q_size: int, k_size: int, n_queries: int, n_keys: int, chunk_size: Optional[int]
) -> Tuple[AttentionChunkConfig, bool]:
    """
    Determine chunking configuration for attention.

    Args:
        q_size (int): Query size
        k_size (int): Key size
        n_queries (int): Number of queries
        n_keys (int): Number of keys
        chunk_size (Optional[int]): Chunk size or None

    Returns:
        Tuple[AttentionChunkConfig, bool]: Chunking configuration and whether chunking is needed
    """
    # Default to no chunking
    use_chunking = False
    
    # Determine if chunking is needed
    if chunk_size is not None:
        # Use specified chunk size
        chunk_size_q = min(chunk_size, n_queries)
        chunk_size_k = min(chunk_size, n_keys)
    else:
        # Use default chunk sizes
        chunk_size_q = n_queries
        chunk_size_k = n_keys
    
    # Calculate number of chunks
    n_chunks_q = (q_size + chunk_size_q - 1) // chunk_size_q
    n_chunks_k = (k_size + chunk_size_k - 1) // chunk_size_k
    
    # Determine if chunking is needed
    if n_chunks_q > 1 or n_chunks_k > 1:
        use_chunking = True
    
    # Create chunking configuration
    config = AttentionChunkConfig(
        chunk_size=chunk_size if chunk_size is not None else max(n_queries, n_keys),
        n_chunks_q=n_chunks_q,
        n_chunks_k=n_chunks_k,
    )
    
    return config, use_chunking


def _process_attention_chunk(inputs: ChunkProcessingInputs) -> torch.Tensor:
    """
    Process a single attention chunk.

    Args:
        inputs (ChunkProcessingInputs): Input parameters

    Returns:
        torch.Tensor: Processed attention output
    """
    # Create AttentionInputs for the chunk
    attention_inputs = AttentionInputs(
        q=inputs.q_chunk,
        k=inputs.k,
        v=inputs.v,
        attn_bias=inputs.bias_slice,
        use_efficient_implementation=inputs.use_efficient_implementation,
        attn_weight_dropout_p=inputs.attn_weight_dropout_p,
        inplace_safe=inputs.inplace_safe,
    )

    # Process the chunk
    return _attention(attention_inputs)


def _process_small_tensors(inputs: LocalAttentionInputs) -> Optional[torch.Tensor]:
    """
    Process small tensors directly without chunking.

    Args:
        inputs (LocalAttentionInputs): Input parameters

    Returns:
        Optional[torch.Tensor]: Processed attention output or None if not applicable
    """
    # Check if tensors are small enough to process directly
    if not (inputs.q.shape[-2] <= inputs.n_queries and inputs.k.shape[-2] <= inputs.n_keys):
        return None
    
    # Step 1: Select the appropriate bias tensor
    bias_to_process = _select_attention_bias(inputs)
    processed_bias = None
    
    # Step 2: Process the bias if present
    if bias_to_process is not None:
        # Try to reshape the bias tensor
        processed_bias = _reshape_bias_tensor(bias_to_process, inputs.q.shape, inputs.k.shape)
        
        # If reshaping failed, log the issue and try fallback method
        if processed_bias is None:
            bias_shape = bias_to_process.shape if bias_to_process is not None else "Unknown (None)"
            warnings.warn(
                f"Warning: Couldn't reshape bias from {bias_shape} to match query/key dimensions. "
                f"q shape: {inputs.q.shape}, k shape: {inputs.k.shape}. "
                f"Attempting fallback bias adjustment."
            )
            processed_bias = _apply_fallback_bias_adjustment(
                bias_to_process, inputs.q.shape, inputs.k.shape, inputs.q.device
            )
    
    # Step 3: Fix dimension mismatches if needed
    if processed_bias is not None and inputs.q.dim() >= 3 and processed_bias.dim() >= 3:
        # Check for dimension mismatch at dim 2 (common issue in tests)
        q_dim_2 = inputs.q.size(2) if inputs.q.dim() > 2 else None
        bias_dim_2 = processed_bias.size(2) if processed_bias.dim() > 2 else None
        
        if q_dim_2 is not None and bias_dim_2 is not None and q_dim_2 != bias_dim_2:
            processed_bias = _fix_dimension_mismatch(processed_bias, q_dim_2, bias_dim_2)
    
    # Step 4: Create attention inputs and process
    attention_inputs = AttentionInputs(
        q=inputs.q,
        k=inputs.k,
        v=inputs.v,
        attn_bias=processed_bias,
        use_efficient_implementation=inputs.use_efficient_implementation,
        attn_weight_dropout_p=inputs.attn_weight_dropout_p,
        inplace_safe=inputs.inplace_safe,
    )
    
    return _attention(attention_inputs)


def _process_chunks(
    inputs: LocalAttentionInputs, config: AttentionChunkConfig
) -> torch.Tensor:
    """
    Process attention in chunks.

    Args:
        inputs (LocalAttentionInputs): Input parameters
        config (AttentionChunkConfig): Chunking configuration

    Returns:
        torch.Tensor: Processed attention output
    """
    # Get input tensors
    q, k, v = inputs.q, inputs.k, inputs.v
    
    # Initialize output chunks list
    out_chunks = []
    
    # Process each query chunk
    for q_chunk_idx in range(config.n_chunks_q):
        # Calculate query chunk indices
        q_start_idx = q_chunk_idx * config.chunk_size
        q_end_idx = min(q_start_idx + config.chunk_size, q.shape[-2])
        
        # Get query chunk
        q_chunk = q[..., q_start_idx:q_end_idx, :]
        
        # Get bias slice for this chunk
        bias_slice = _get_bias_slice(inputs.attn_bias, q_start_idx, q_end_idx)
        
        # Create chunk processing inputs
        chunk_inputs = ChunkProcessingInputs(
            q_chunk=q_chunk,
            k=k,
            v=v,
            bias_slice=bias_slice,
            use_efficient_implementation=inputs.use_efficient_implementation,
            attn_weight_dropout_p=inputs.attn_weight_dropout_p,
            inplace_safe=inputs.inplace_safe,
        )
        
        # Process the chunk
        out_chunk = _process_attention_chunk(chunk_inputs)
        
        # Add to output chunks
        out_chunks.append(out_chunk)
    
    # Concatenate output chunks
    return torch.cat(out_chunks, dim=-2)
