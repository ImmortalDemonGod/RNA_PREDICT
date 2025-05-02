"""
Attention processing utilities.

This module contains functions for processing attention operations,
including handling different query-key-value configurations and small tensor processing.
"""

import math
import warnings
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

from .attention_core import (
    AttentionInputs,
    ProcessDifferentQueryInputs,
    ProcessQueryInputs,
    attention,
)


def process_same_query_keyvalue(
    inputs: ProcessQueryInputs,
    num_heads: int,
    attn_weight_dropout_p: float,
    use_efficient_implementation: bool,
) -> torch.Tensor:
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
    # Create parameter object
    tensor_inputs = TensorInputs(q=q, k=k, v=v)
    config = AttentionConfig(
        num_heads=num_heads,
        attn_weight_dropout_p=attn_weight_dropout_p,
        use_efficient_implementation=use_efficient_implementation,
        inplace_safe=inputs.inplace_safe,
    )
    params = BatchMatmulParams(
        tensors=tensor_inputs, attn_bias=inputs.attn_bias, config=config
    )
    return _process_with_batch_matmul(params)


class TensorInputs(NamedTuple):
    """
    Base class for tensor inputs.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


class AttentionConfig(NamedTuple):
    """
    Configuration for attention processing.
    """

    num_heads: int
    attn_weight_dropout_p: float
    use_efficient_implementation: bool
    inplace_safe: bool


class BatchMatmulParams(NamedTuple):
    """
    Parameter object for batch matrix multiplication attention processing.
    Uses NamedTuple for immutability and reduced overhead.
    """

    tensors: TensorInputs
    attn_bias: Optional[torch.Tensor]
    config: AttentionConfig


def _process_with_batch_matmul(params: BatchMatmulParams) -> torch.Tensor:
    """
    Process attention using batch matrix multiplication.

    Args:
        params: Parameter object containing all necessary inputs

    Returns:
        torch.Tensor: Processed attention output
    """
    bsz = params.tensors.q.shape[0]
    q = params.tensors.q.reshape(-1, *params.tensors.q.shape[-2:])
    k = params.tensors.k.reshape(-1, *params.tensors.k.shape[-2:])
    v = params.tensors.v.reshape(-1, *params.tensors.v.shape[-2:])

    # Get attention scores
    reshaped_bias = _reshape_attention_bias(params.attn_bias, params.config.num_heads)

    attention_inputs = AttentionInputs(
        q=q,
        k=k,
        v=v,
        attn_bias=reshaped_bias,
        use_efficient_implementation=params.config.use_efficient_implementation,
        attn_weight_dropout_p=params.config.attn_weight_dropout_p,
        inplace_safe=params.config.inplace_safe,
    )

    attn_output = attention(attention_inputs)

    # Reshape back
    h = params.config.num_heads
    attn_output = attn_output.reshape(bsz, h, *attn_output.shape[-2:])

    # [..., h, n_q, d_h] -> [..., n_q, h, d_h]
    return attn_output.transpose(-2, -3)


def _reshape_attention_bias(
    attn_bias: Optional[torch.Tensor], num_heads: int
) -> Optional[torch.Tensor]:
    """
    Reshape attention bias to match attention weights robustly for any expected input shape.

    Args:
        attn_bias: Original attention bias
        num_heads: Number of attention heads

    Returns:
        Optional[torch.Tensor]: Reshaped bias or None if reshaping fails
    """
    import warnings
    import torch
    if attn_bias is None:
        return None

    try:
        print(f"[INSTRUMENT][_reshape_attention_bias] attn_bias.shape={attn_bias.shape}, num_heads={num_heads}, attn_bias.dtype={attn_bias.dtype}")
        bias = attn_bias
        # If bias has batch and head dims, try to match them to num_heads
        # Accepts [B, 1, Nq, Nk], [B, H, Nq, Nk], [1, 1, H, Nq, Nk], [1, 1, 2, 170, 170], etc.
        shape = list(bias.shape)
        # Case: [B, 1, Nq, Nk] -> [B, H, Nq, Nk]
        if len(shape) == 4 and shape[1] != num_heads:
            bias = bias.expand(shape[0], num_heads, shape[2], shape[3])
            print(f"[INSTRUMENT][_reshape_attention_bias] expanded bias to shape={bias.shape}")
        # Case: [1, 1, H, Nq, Nk] or [B, 1, H, Nq, Nk] -> [B, H, Nq, Nk]
        elif len(shape) == 5 and shape[2] == num_heads:
            bias = bias.squeeze(1)  # Remove singleton dim if present
            print(f"[INSTRUMENT][_reshape_attention_bias] squeezed bias to shape={bias.shape}")
        # If still not matching, try to broadcast or fallback
        if bias.shape[1] != num_heads:
            # Try to repeat or expand as last resort
            if bias.shape[1] == 1:
                bias = bias.expand(bias.shape[0], num_heads, *bias.shape[2:])
                print(f"[INSTRUMENT][_reshape_attention_bias] forced expand to shape={bias.shape}")
            else:
                print(f"[INSTRUMENT][_reshape_attention_bias][ERROR] Could not match num_heads: bias.shape={bias.shape}, num_heads={num_heads}")
                warnings.warn(f"Could not match num_heads: bias.shape={bias.shape}, num_heads={num_heads}")
                return None
        # Now flatten batch and head dims for attention
        print(f"[INSTRUMENT][_reshape_attention_bias] before final reshape: bias.shape={bias.shape}")
        result = bias.reshape(-1, *bias.shape[-2:])  # [B*H, N_q, N_kv]
        print(f"[INSTRUMENT][_reshape_attention_bias] after final reshape: result.shape={result.shape}")
        return result
    except Exception as e:
        print(f"[INSTRUMENT][_reshape_attention_bias][ERROR] Could not robustly reshape attn_bias from {attn_bias.shape} to match attention weights. Error: {str(e)}")
        warnings.warn(
            f"Could not robustly reshape attn_bias from {attn_bias.shape} to match attention weights. Error: {str(e)}"
        )
        return None  # Skip bias if reshape fails


def process_different_query_keyvalue(
    inputs: ProcessDifferentQueryInputs,
    use_efficient_implementation: bool,
    attn_weight_dropout_p: float,
    local_attention_method: str,
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


class SmallTensorParams(NamedTuple):
    """
    Parameter object for small tensor attention processing.
    Uses NamedTuple for immutability and reduced overhead.
    """

    tensors: TensorInputs
    bias: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


def _ensure_batch_dimensions(params: SmallTensorParams) -> TensorInputs:
    """
    Ensure all tensors have the same batch dimensions.

    Args:
        params: Parameter object containing tensors

    Returns:
        TensorInputs: Tensors with consistent batch dimensions
    """
    batch_dims = params.tensors.q.shape[:-2]
    q, k, v = params.tensors.q, params.tensors.k, params.tensors.v

    # Expand k and v if needed
    if k.shape[:-2] != batch_dims:
        k = k.expand(*batch_dims, *k.shape[-2:])
    if v.shape[:-2] != batch_dims:
        v = v.expand(*batch_dims, *v.shape[-2:])

    return TensorInputs(q=q, k=k, v=v)


def _compute_attention_scores(tensors: TensorInputs) -> torch.Tensor:
    """
    Compute attention scores between query and key tensors.

    Args:
        tensors: Object containing query and key tensors

    Returns:
        torch.Tensor: Attention scores
    """
    return torch.matmul(tensors.q, tensors.k.transpose(-2, -1)) / math.sqrt(
        tensors.q.size(-1)
    )


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

    adjusted_bias = adjust_attention_bias(
        bias, scores.shape, tensor_name="attention_bias"
    )
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


class SmallTensorInputs(NamedTuple):
    """
    Input parameters for small tensor processing.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    bias: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


def process_small_tensors(inputs: SmallTensorInputs) -> torch.Tensor:
    """
    Process attention for small tensors that fit in memory.

    Args:
        inputs: Input parameters containing query, key, value tensors and optional bias/mask

    Returns:
        Output tensor [..., N_q, d]
    """
    # Create a parameter object to reduce the number of arguments in internal functions
    tensor_inputs = TensorInputs(q=inputs.q, k=inputs.k, v=inputs.v)
    params = SmallTensorParams(
        tensors=tensor_inputs, bias=inputs.bias, mask=inputs.mask
    )

    # Ensure all tensors have same batch dimensions
    adjusted_tensors = _ensure_batch_dimensions(params)

    # Compute attention scores
    scores = _compute_attention_scores(adjusted_tensors)

    # Add bias if provided
    if params.bias is not None:
        scores = _apply_attention_bias(scores, params.bias)

    # Apply mask if provided
    if params.mask is not None:
        scores = _apply_attention_mask(scores, params.mask)

    # Apply attention
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, adjusted_tensors.v)
