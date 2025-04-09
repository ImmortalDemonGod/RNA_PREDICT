"""
Local attention implementation.

This module contains functions for performing local attention operations.
"""

from typing import Optional

import torch

from .attention_types import LocalAttentionInputs
from .attention_tensor import _determine_chunking, _process_small_tensors, _process_chunks


def _local_attention(inputs: LocalAttentionInputs) -> torch.Tensor:
    """
    Perform local attention operation.

    Args:
        inputs (LocalAttentionInputs): Input parameters

    Returns:
        torch.Tensor: Attention output
    """
    # Try processing small tensors directly
    small_tensor_result = _process_small_tensors(inputs)
    if small_tensor_result is not None:
        return small_tensor_result
    
    # Determine chunking configuration
    q_size, k_size = inputs.q.shape[-2], inputs.k.shape[-2]
    config, use_chunking = _determine_chunking(
        q_size, k_size, inputs.n_queries, inputs.n_keys, inputs.chunk_size
    )
    
    # Process in chunks if needed
    if use_chunking:
        return _process_chunks(inputs, config)
    else:
        # This should not happen as small tensors are already handled
        raise ValueError(
            f"Unexpected state: tensors not small enough for direct processing "
            f"but chunking not required. q_size={q_size}, k_size={k_size}, "
            f"n_queries={inputs.n_queries}, n_keys={inputs.n_keys}"
        )


def local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
    trunked_attn_bias: Optional[torch.Tensor] = None,
    inf: float = 1e10,
    use_efficient_implementation: bool = False,
    attn_weight_dropout_p: float = 0.0,
    inplace_safe: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Perform local attention with optional chunking.

    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        n_queries (int): Number of queries
        n_keys (int): Number of keys
        attn_bias (Optional[torch.Tensor], optional): Attention bias. Defaults to None.
        trunked_attn_bias (Optional[torch.Tensor], optional): Trunked attention bias. Defaults to None.
        inf (float, optional): Infinity value. Defaults to 1e10.
        use_efficient_implementation (bool, optional): Whether to use efficient implementation. Defaults to False.
        attn_weight_dropout_p (float, optional): Attention weight dropout probability. Defaults to 0.0.
        inplace_safe (bool, optional): Whether inplace operations are safe. Defaults to False.
        chunk_size (Optional[int], optional): Chunk size. Defaults to None.

    Returns:
        torch.Tensor: Attention output
    """
    # Create inputs
    inputs = LocalAttentionInputs(
        q=q,
        k=k,
        v=v,
        n_queries=n_queries,
        n_keys=n_keys,
        attn_bias=attn_bias,
        trunked_attn_bias=trunked_attn_bias,
        inf=inf,
        use_efficient_implementation=use_efficient_implementation,
        attn_weight_dropout_p=attn_weight_dropout_p,
        inplace_safe=inplace_safe,
        chunk_size=chunk_size,
    )
    
    # Perform local attention
    return _local_attention(inputs)
