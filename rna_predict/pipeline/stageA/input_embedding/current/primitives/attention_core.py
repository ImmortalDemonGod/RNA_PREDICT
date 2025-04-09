"""
Core attention mechanism implementation.

This module contains the core attention function and related dataclasses
for representing attention inputs and configurations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .attention_weights import validate_attention_shapes, handle_dimension_mismatch, compute_attention_weights


@dataclass
class AttentionInputs:
    """Inputs for attention operation."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attn_bias: Optional[torch.Tensor] = None
    use_efficient_implementation: bool = False
    attn_weight_dropout_p: float = 0.0
    inplace_safe: bool = False


@dataclass
class ProcessQueryInputs:
    """Inputs for processing query-key-value attention."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_x: torch.Tensor
    attn_bias: Optional[torch.Tensor] = None
    inplace_safe: bool = False


@dataclass
class ProcessDifferentQueryInputs:
    """Inputs for processing different query-key-value attention."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_x: torch.Tensor
    attn_bias: Optional[torch.Tensor] = None
    trunked_attn_bias: Optional[torch.Tensor] = None
    n_queries: Optional[int] = None
    n_keys: Optional[int] = None
    inf: Optional[float] = 1e10
    inplace_safe: bool = False
    chunk_size: Optional[int] = None


@dataclass
class ForwardInputs:
    """Inputs for attention forward pass."""

    q_x: torch.Tensor
    kv_x: torch.Tensor
    attn_bias: Optional[torch.Tensor] = None
    trunked_attn_bias: Optional[torch.Tensor] = None
    n_queries: Optional[int] = None
    n_keys: Optional[int] = None
    inf: Optional[float] = 1e10
    inplace_safe: bool = False
    chunk_size: Optional[int] = None


@dataclass
class AttentionConfig:
    """Configuration for Attention module."""

    c_q: int
    c_k: int
    c_v: int
    c_hidden: int
    num_heads: int
    gating: bool = True
    q_linear_bias: bool = False
    local_attention_method: str = "global_attention_with_bias"
    use_efficient_implementation: bool = False
    attn_weight_dropout_p: float = 0.0


def attention(inputs: AttentionInputs) -> torch.Tensor:
    """Attention.

    Args:
        inputs (AttentionInputs): Inputs for attention calculation

    Returns:
        torch.Tensor: output tensor
    """
    # Validate input shapes
    validate_attention_shapes(inputs.q, inputs.k, inputs.v)

    # Try efficient implementation if requested
    if inputs.use_efficient_implementation:
        try:
            attn_output = F.scaled_dot_product_attention(
                query=inputs.q,
                key=inputs.k,
                value=inputs.v,
                attn_mask=inputs.attn_bias,
                dropout_p=inputs.attn_weight_dropout_p,
            )
            return attn_output
        except RuntimeError:
            # Fall back to manual implementation
            pass

    # Transpose key for matrix multiplication
    # [..., n_kv, d] -> [..., d, n_kv]
    k_transposed = inputs.k.transpose(-1, -2)

    # Handle potential dimension mismatch
    q_adj, k_adj = handle_dimension_mismatch(inputs.q, k_transposed)

    # Compute attention weights
    attn_weight = compute_attention_weights(q_adj, k_adj, inputs.attn_bias)

    # Apply dropout if specified
    if inputs.attn_weight_dropout_p > 0.0:
        attn_weight = F.dropout(attn_weight, p=inputs.attn_weight_dropout_p)

    # Apply attention weights to values
    return torch.matmul(attn_weight, inputs.v)
