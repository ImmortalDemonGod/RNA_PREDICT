"""
Attention processing utilities.

This module contains functions for processing attention operations,
including handling different query-key-value configurations and small tensor processing.
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from .attention_core import (
    AttentionInputs,
    ProcessQueryInputs,
    ProcessDifferentQueryInputs,
    attention,
)


def process_same_query_keyvalue(inputs: ProcessQueryInputs, num_heads: int,
                               attn_weight_dropout_p: float,
                               use_efficient_implementation: bool) -> torch.Tensor:
    """
    Process attention when query and key/value are the same.

    Args:
        inputs (ProcessQueryInputs): Input parameters
        num_heads (int): Number of attention heads
        attn_weight_dropout_p (float): Dropout probability for attention weights
        use_efficient_implementation (bool): Whether to use efficient implementation

    Returns:
        torch.Tensor: processed attention output
    """
    # Move head dimension for standard attention calculation
    # [..., n_q, h, d_h] -> [..., h, n_q, d_h]
    q = inputs.q.transpose(-2, -3)
    k = inputs.k.transpose(-2, -3)
    v = inputs.v.transpose(-2, -3)

    # Use efficient SDPA if available
    if use_efficient_implementation:
        try:
            o = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=inputs.attn_bias,
                dropout_p=attn_weight_dropout_p,
            )
            # [..., h, n_q, d_h] -> [..., n_q, h, d_h]
            return o.transpose(-2, -3)
        except RuntimeError:
            # Fall back to manual implementation
            pass

    # Otherwise use batch matmul
    return _process_with_batch_matmul(
        q, k, v, inputs.attn_bias, num_heads,
        attn_weight_dropout_p, use_efficient_implementation,
        inputs.inplace_safe
    )


def _process_with_batch_matmul(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    num_heads: int,
    attn_weight_dropout_p: float,
    use_efficient_implementation: bool,
    inplace_safe: bool
) -> torch.Tensor:
    """
    Process attention using batch matrix multiplication.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        attn_bias: Optional attention bias
        num_heads: Number of attention heads
        attn_weight_dropout_p: Dropout probability
        use_efficient_implementation: Whether to use efficient implementation
        inplace_safe: Whether inplace operations are safe

    Returns:
        torch.Tensor: Processed attention output
    """
    bsz = q.shape[0]
    q = q.reshape(-1, *q.shape[-2:])
    k = k.reshape(-1, *k.shape[-2:])
    v = v.reshape(-1, *v.shape[-2:])

    # Get attention scores
    reshaped_bias = _reshape_attention_bias(attn_bias, num_heads)

    attention_inputs = AttentionInputs(
        q=q,
        k=k,
        v=v,
        attn_bias=reshaped_bias,
        use_efficient_implementation=use_efficient_implementation,
        attn_weight_dropout_p=attn_weight_dropout_p,
        inplace_safe=inplace_safe,
    )

    attn_output = attention(attention_inputs)

    # Reshape back
    h = num_heads
    attn_output = attn_output.reshape(bsz, h, *attn_output.shape[-2:])

    # [..., h, n_q, d_h] -> [..., n_q, h, d_h]
    return attn_output.transpose(-2, -3)


def _reshape_attention_bias(
    attn_bias: Optional[torch.Tensor], num_heads: int
) -> Optional[torch.Tensor]:
    """
    Reshape attention bias to match attention weights.

    Args:
        attn_bias: Original attention bias
        num_heads: Number of attention heads

    Returns:
        Optional[torch.Tensor]: Reshaped bias or None if reshaping fails
    """
    if attn_bias is None:
        return None

    try:
        # First ensure bias has correct number of heads
        if attn_bias.shape[1] != num_heads:
            # If bias has wrong number of heads, expand/repeat to match
            bias = attn_bias.unsqueeze(1)  # [B, 1, N_q, N_kv]
            bias = bias.expand(-1, num_heads, -1, -1)  # [B, H, N_q, N_kv]
        else:
            bias = attn_bias

        # Now reshape to match attention weights
        return bias.reshape(-1, *bias.shape[-2:])  # [B*H, N_q, N_kv]
    except RuntimeError as e:
        warnings.warn(
            f"Could not reshape attn_bias from {attn_bias.shape} to match attention weights. Error: {str(e)}"
        )
        return None  # Skip bias if reshape fails


def process_different_query_keyvalue(
    inputs: ProcessDifferentQueryInputs,
    use_efficient_implementation: bool,
    attn_weight_dropout_p: float,
    local_attention_method: str
) -> torch.Tensor:
    """
    Process attention when query and key/value are different.

    Args:
        inputs (ProcessDifferentQueryInputs): Input parameters
        use_efficient_implementation (bool): Whether to use efficient implementation
        attn_weight_dropout_p (float): Dropout probability for attention weights
        local_attention_method (str): Method to use for local attention

    Returns:
        torch.Tensor: processed attention output
    """
    # For other cases, import the local attention function
    from .attention_utils import LocalAttentionInputs, _local_attention

    # Use efficient implementation if available
    if "global_attention_with_bias" in local_attention_method:
        # Handle n_queries and n_keys for type compatibility
        actual_n_queries = (
            inputs.n_queries if inputs.n_queries is not None else inputs.q.size(-2)
        )
        actual_n_keys = (
            inputs.n_keys if inputs.n_keys is not None else inputs.k.size(-2)
        )

        # Create the LocalAttentionInputs instance
        local_attn_inputs = LocalAttentionInputs(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            n_queries=actual_n_queries,
            n_keys=actual_n_keys,
            attn_bias=inputs.attn_bias,
            trunked_attn_bias=inputs.trunked_attn_bias,
            inf=inputs.inf if inputs.inf is not None else 1e10,
            use_efficient_implementation=use_efficient_implementation,
            attn_weight_dropout_p=attn_weight_dropout_p,
            inplace_safe=inputs.inplace_safe,
            chunk_size=inputs.chunk_size,
        )

        # This implementation requires advanced handling, use _local_attention with dataclass
        return _local_attention(local_attn_inputs)
    else:
        # Simple attention without special handling
        attention_inputs = AttentionInputs(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            attn_bias=inputs.attn_bias,
            use_efficient_implementation=use_efficient_implementation,
            attn_weight_dropout_p=attn_weight_dropout_p,
            inplace_safe=inputs.inplace_safe,
        )
        return attention(attention_inputs)


class SmallTensorParams:
    """
    Parameter object for small tensor attention processing.
    """
    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ):
        """
        Initialize parameters for small tensor processing.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            bias: Optional attention bias
            mask: Optional attention mask
        """
        self.q = q
        self.k = k
        self.v = v
        self.bias = bias
        self.mask = mask


def _ensure_batch_dimensions(params: SmallTensorParams) -> None:
    """
    Ensure all tensors have the same batch dimensions.

    Args:
        params: Parameter object containing tensors
    """
    batch_dims = params.q.shape[:-2]
    for t_name in ['k', 'v']:
        t = getattr(params, t_name)
        if t.shape[:-2] != batch_dims:
            setattr(params, t_name, t.expand(*batch_dims, *t.shape[-2:]))


def _compute_attention_scores(params: SmallTensorParams) -> torch.Tensor:
    """
    Compute attention scores between query and key tensors.

    Args:
        params: Parameter object containing tensors

    Returns:
        torch.Tensor: Attention scores
    """
    return torch.matmul(params.q, params.k.transpose(-2, -1)) / math.sqrt(params.q.size(-1))


def _apply_attention_bias(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Apply attention bias to scores.

    Args:
        scores: Attention scores
        bias: Attention bias

    Returns:
        torch.Tensor: Scores with bias applied
    """
    from rna_predict.utils.shape_utils import adjust_attention_bias
    adjusted_bias = adjust_attention_bias(bias, scores.shape, tensor_name="attention_bias")
    return scores + adjusted_bias


def _apply_attention_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply attention mask to scores.

    Args:
        scores: Attention scores
        mask: Attention mask

    Returns:
        torch.Tensor: Scores with mask applied
    """
    # Ensure mask has compatible shape
    if mask.shape[-2:] != scores.shape[-2:]:
        mask = mask.expand(*scores.shape[:-2], *mask.shape[-2:])
    return scores.masked_fill(~mask, float("-inf"))


def process_small_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Process attention for small tensors that fit in memory.

    Args:
        q: Query tensor [..., N_q, d]
        k: Key tensor [..., N_k, d]
        v: Value tensor [..., N_k, d]
        bias: Optional attention bias [..., N_q, N_k]
        mask: Optional attention mask [..., N_q, N_k]

    Returns:
        Output tensor [..., N_q, d]
    """
    # Create a parameter object to reduce the number of arguments
    params = SmallTensorParams(q, k, v, bias, mask)

    # Ensure all tensors have same batch dimensions
    _ensure_batch_dimensions(params)

    # Compute attention scores
    scores = _compute_attention_scores(params)

    # Add bias if provided
    if params.bias is not None:
        scores = _apply_attention_bias(scores, params.bias)

    # Apply mask if provided
    if params.mask is not None:
        scores = _apply_attention_mask(scores, params.mask)

    # Apply attention
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, params.v)
