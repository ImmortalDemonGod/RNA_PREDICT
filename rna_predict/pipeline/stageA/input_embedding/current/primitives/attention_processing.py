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
                               Processes attention when the query and key/value tensors are identical.
                               
                               Transposes the input query, key, and value tensors to align their dimensions with the 
                               standard attention computation. When efficient PyTorch scaled dot product attention is 
                               enabled and available, it is used; otherwise, the function falls back to a manual 
                               implementation based on batch matrix multiplication.
                               
                               Args:
                                   inputs (ProcessQueryInputs): Container for query, key, value tensors, optional attention 
                                       bias, and a flag indicating in-place safety.
                                   num_heads (int): The number of attention heads.
                                   attn_weight_dropout_p (float): Dropout probability for the attention weights.
                                   use_efficient_implementation (bool): Whether to attempt the efficient scaled dot product attention.
                                   
                               Returns:
                                   torch.Tensor: The output attention tensor with dimensions rearranged to match the input format.
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
    Processes attention using batch matrix multiplication.
    
    Reshapes the query, key, and value tensors to merge batch dimensions and applies an optional
    attention bias adjusted to match the specified number of attention heads. Constructs an input
    configuration for computing the attention output via batch matrix multiplication. Finally, the
    output is reshaped to restore the original batch and head dimensions with an appropriate axis
    transposition.
    
    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        attn_bias: Optional tensor containing the attention bias.
        num_heads: Number of attention heads.
        attn_weight_dropout_p: Dropout probability applied to attention weights.
        use_efficient_implementation: Flag indicating whether to use an efficient attention algorithm.
        inplace_safe: Indicates if in-place operations are permissible.
    
    Returns:
        A tensor representing the computed attention output.
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
    Reshape an attention bias tensor to match the attention weights.
    
    This function ensures that the bias tensor has the required number of attention heads.
    If the tensor's head dimension does not equal the specified num_heads, the tensor is expanded
    to duplicate the bias across the necessary heads and then reshaped by merging the batch and head dimensions.
    If the input bias is None or if a reshaping error occurs, the function returns None.
      
    Args:
        attn_bias: The original attention bias tensor, or None.
        num_heads: The required number of attention heads.
      
    Returns:
        The reshaped attention bias tensor with merged batch and head dimensions,
        or None if the input is None or reshaping fails.
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
    Process attention when query and key/value tensors differ.
    
    Depending on the specified local_attention_method, this function applies either an advanced
    local attention mechanism with bias handling or a standard attention computation. When using
    global attention with bias, the number of queries and keys is inferred from the input tensors
    if not explicitly provided.
    
    Args:
        inputs (ProcessDifferentQueryInputs): Container for query, key, and value tensors along with
            optional attention bias parameters.
        use_efficient_implementation (bool): Flag to enable an efficient attention computation.
        attn_weight_dropout_p (float): Dropout probability applied to the attention weights.
        local_attention_method (str): Specifies the attention method; use "global_attention_with_bias" for
            bias-aware local attention, otherwise standard attention is used.
    
    Returns:
        torch.Tensor: The computed attention output.
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
        Initializes a container for small tensor attention parameters.
        
        This constructor stores the query, key, and value tensors along with an optional attention
        bias and mask for use in small tensor processing.
            
        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            bias: Optional attention bias tensor.
            mask: Optional attention mask tensor.
        """
        self.q = q
        self.k = k
        self.v = v
        self.bias = bias
        self.mask = mask


def _ensure_batch_dimensions(params: SmallTensorParams) -> None:
    """
    Ensures that the key and value tensors have the same batch dimensions as the query tensor.
    
    This function extracts the batch dimensions from the query tensor and checks the key and
    value tensors. If their batch dimensions do not match, they are expanded to align with the query's
    batch dimensions, ensuring consistency for subsequent attention computations.
    
    Args:
        params: A SmallTensorParams instance containing tensors 'q', 'k', and 'v'.
    """
    batch_dims = params.q.shape[:-2]
    for t_name in ['k', 'v']:
        t = getattr(params, t_name)
        if t.shape[:-2] != batch_dims:
            setattr(params, t_name, t.expand(*batch_dims, *t.shape[-2:]))


def _compute_attention_scores(params: SmallTensorParams) -> torch.Tensor:
    """
    Compute scaled dot-product attention scores.
    
    This function computes the attention scores by multiplying the query tensor with
    the transposed key tensor and scaling the result by the square root of the size
    of the query's last dimension. It assumes that the provided SmallTensorParams
    instance contains matching query (q) and key (k) tensors.
    
    Args:
        params: A SmallTensorParams instance containing the query and key tensors.
    
    Returns:
        torch.Tensor: The computed, scaled attention scores.
    """
    return torch.matmul(params.q, params.k.transpose(-2, -1)) / math.sqrt(params.q.size(-1))


def _apply_attention_bias(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies a bias to attention scores.
    
    This function adjusts the shape of the provided bias tensor to match the score's
    dimensions using an external utility and then adds the adjusted bias to the scores.
    
    Args:
        scores: The attention scores tensor.
        bias: The tensor representing attention bias, which may be reshaped to fit the scores.
    
    Returns:
        The attention scores with the adjusted bias applied.
    """
    from rna_predict.utils.shape_utils import adjust_attention_bias
    adjusted_bias = adjust_attention_bias(bias, scores.shape, tensor_name="attention_bias")
    return scores + adjusted_bias


def _apply_attention_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply the attention mask to the attention scores.
    
    If the mask's last two dimensions do not match those of scores, it is expanded accordingly.
    Positions where the mask is False are set to negative infinity, preventing them from affecting
    the subsequent attention computation.
    
    Args:
        scores (torch.Tensor): The attention scores tensor.
        mask (torch.Tensor): A boolean tensor where False indicates positions to mask out.
    
    Returns:
        torch.Tensor: The scores tensor with masked positions set to -∞.
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
    Compute scaled dot-product attention for small tensors.
    
    This function computes the attention output for small tensors that fit in memory.
    It first ensures that the query (q), key (k), and value (v) tensors have consistent
    batch dimensions. It then calculates the attention scores via a dot-product between q and k,
    optionally adds an attention bias, and applies a mask if provided before normalizing the scores
    with softmax. The final output is obtained by computing the weighted sum of v using these
    attention weights.
    
    Args:
        q: Query tensor of shape [..., N_q, d].
        k: Key tensor of shape [..., N_k, d].
        v: Value tensor of shape [..., N_k, d].
        bias: Optional tensor of shape [..., N_q, N_k] added to the attention scores.
        mask: Optional tensor of shape [..., N_q, N_k] applied to filter attention scores.
    
    Returns:
        Tensor of shape [..., N_q, d] representing the computed attention output.
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
