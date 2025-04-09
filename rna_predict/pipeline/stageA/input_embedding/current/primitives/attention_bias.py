"""
Bias creation and manipulation for attention operations.

This module contains functions for creating and manipulating attention bias tensors.
"""

import warnings
from typing import Optional, Tuple

import torch

from .attention_types import BiasCreationInputs, LocalAttentionInputs


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
        Tuple[int, int, int, int]: Bias shape (batch_size, heads, queries, keys)
    """
    return (1, 1, inputs.n_queries, inputs.n_keys)


def create_local_attn_bias(inputs: BiasCreationInputs) -> torch.Tensor:
    """
    Create local attention bias.

    Args:
        inputs (BiasCreationInputs): Input parameters

    Returns:
        torch.Tensor: Local attention bias
    """
    # Calculate bias shape
    bias_shape = _calculate_bias_shape(inputs)

    # Create bias tensor
    device = inputs.device if inputs.device is not None else torch.device("cpu")
    bias = torch.zeros(bias_shape, device=device)

    # Apply masking
    n_q, n_k = inputs.n_queries, inputs.n_keys
    n = inputs.n

    # Handle case where n_q > n or n_k > n
    if n_q > n or n_k > n:
        # Create mask for valid positions
        valid_q = min(n_q, n)
        valid_k = min(n_k, n)

        # Set invalid positions to -inf
        if valid_q < n_q:
            bias[..., valid_q:, :] = -inputs.inf
        if valid_k < n_k:
            bias[..., :, valid_k:] = -inputs.inf

    return bias


def _select_attention_bias(inputs: LocalAttentionInputs) -> Optional[torch.Tensor]:
    """
    Select and prepare the appropriate attention bias tensor.
    
    Args:
        inputs (LocalAttentionInputs): Input parameters
        
    Returns:
        Optional[torch.Tensor]: Processed bias tensor or None if not applicable
    """
    # Prioritize attn_bias, fall back to trunked_attn_bias
    bias_to_process = inputs.attn_bias
    
    # If no direct bias, try trunked bias
    if bias_to_process is None:
        bias_to_process = inputs.trunked_attn_bias
        
        # Handle 5D trunked bias case
        if bias_to_process is not None and bias_to_process.ndim == 5:
            # If the block dimension is 1, we can safely squeeze it
            if bias_to_process.shape[-3] == 1:
                bias_to_process = bias_to_process.squeeze(-3)
            else:
                # This case is ambiguous, proceed without bias
                warnings.warn(
                    f"Trunked bias has unexpected 5D shape {bias_to_process.shape} "
                    f"with block dim != 1. Cannot safely adapt for small tensor processing. Skipping bias."
                )
                bias_to_process = None
    
    return bias_to_process


def _reshape_bias_tensor(bias: torch.Tensor, q_shape: Tuple[int, ...], k_shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
    """
    Reshape bias tensor to match query and key dimensions.
    
    Args:
        bias: Attention bias tensor
        q_shape: Shape of query tensor
        k_shape: Shape of key tensor
        
    Returns:
        Optional[torch.Tensor]: Reshaped bias tensor or None if reshaping fails
    """
    try:
        expected_size = q_shape[-2] * k_shape[-2]
        actual_size = bias.numel()
        
        # If total elements match, we can reshape directly
        if expected_size == actual_size:
            target_bias_shape = (*bias.shape[:-2], q_shape[-2], k_shape[-2])
            return bias.reshape(target_bias_shape)
        
        # If sizes don't match, use shape_utils to adjust
        from rna_predict.utils.shape_utils import adjust_attention_bias
        target_scores_shape = (*bias.shape[:-2], q_shape[-2], k_shape[-2])
        return adjust_attention_bias(
            bias,
            target_scores_shape,
            tensor_name="trunked_attention_bias"
        )
    except (RuntimeError, ValueError):
        # Reshaping failed
        return None


def _apply_fallback_bias_adjustment(bias: Optional[torch.Tensor], q_shape: Tuple[int, ...], k_shape: Tuple[int, ...], device: torch.device) -> Optional[torch.Tensor]:
    """
    Apply fallback bias adjustment when primary reshaping fails.
    
    Args:
        bias: Original bias tensor (may be None)
        q_shape: Shape of query tensor
        k_shape: Shape of key tensor
        device: Device for tensor creation
        
    Returns:
        Optional[torch.Tensor]: Adjusted bias tensor or None if adjustment fails
    """
    try:
        from rna_predict.utils.shape_utils import adjust_attention_bias
        
        # Create target shape for scores
        target_scores_shape = (1, 1, q_shape[-2], k_shape[-2])
        
        # If bias is None, create a zero bias tensor
        if bias is None:
            bias = torch.zeros(target_scores_shape, device=device)
        
        # Adjust bias to match target shape
        return adjust_attention_bias(
            bias,
            target_scores_shape,
            tensor_name="fallback_attention_bias"
        )
    except Exception as fallback_error:
        warnings.warn(f"Fallback bias adjustment failed: {fallback_error}. Proceeding without bias.")
        return None


def _create_expanded_bias_tensor(bias: torch.Tensor, target_dim_size: int) -> torch.Tensor:
    """
    Create a new bias tensor with expanded dimension 2.
    
    Args:
        bias: Original bias tensor
        target_dim_size: Target size for dimension 2
        
    Returns:
        torch.Tensor: New bias tensor with expanded dimension 2
    """
    if bias.dim() == 5:  # 5D case
        new_bias = torch.zeros(
            bias.shape[0],
            bias.shape[1],
            target_dim_size,
            bias.shape[3],
            bias.shape[4],
            device=bias.device,
            dtype=bias.dtype
        )
    else:  # Other dimensionality cases
        new_shape = list(bias.shape)
        new_shape[2] = target_dim_size
        new_bias = torch.zeros(new_shape, device=bias.device, dtype=bias.dtype)
    
    return new_bias


def _copy_bias_data(new_bias: torch.Tensor, bias: torch.Tensor, source_dim_size: int) -> torch.Tensor:
    """
    Copy data from original bias to new bias tensor.
    
    Args:
        new_bias: Target bias tensor
        bias: Source bias tensor
        source_dim_size: Size of dimension 2 in source tensor
        
    Returns:
        torch.Tensor: New bias tensor with copied data
    """
    if new_bias.dim() == 3:
        new_bias[:, :, :source_dim_size] = bias
    elif new_bias.dim() == 4:
        new_bias[:, :, :source_dim_size, :] = bias
    elif new_bias.dim() == 5:
        new_bias[:, :, :source_dim_size] = bias
    
    return new_bias


def _fix_dimension_mismatch(bias: torch.Tensor, q_dim_2: int, bias_dim_2: int) -> torch.Tensor:
    """
    Fix dimension mismatch between query and bias tensors at dimension 2.
    
    Args:
        bias: Attention bias tensor
        q_dim_2: Size of dimension 2 in query tensor
        bias_dim_2: Size of dimension 2 in bias tensor
        
    Returns:
        torch.Tensor: Adjusted bias tensor
    """
    # Case 1: Expand bias from dimension 4 to 5
    if q_dim_2 == 5 and bias_dim_2 == 4:
        new_bias = _create_expanded_bias_tensor(bias, 5)
        return _copy_bias_data(new_bias, bias, bias_dim_2)
    
    # Case 2: Slice bias from dimension 5 to 4
    if q_dim_2 == 4 and bias_dim_2 == 5:
        return bias[:, :, :4] if bias.dim() >= 3 else bias
    
    # Default: return original bias if no specific case matches
    return bias


def _get_bias_slice(
    attn_bias: Optional[torch.Tensor], q_start_idx: int, q_end_idx: int
) -> Optional[torch.Tensor]:
    """
    Get a slice of the attention bias for a specific query range.

    Args:
        attn_bias (Optional[torch.Tensor]): Attention bias tensor or None
        q_start_idx (int): Start index for query dimension
        q_end_idx (int): End index for query dimension

    Returns:
        Optional[torch.Tensor]: Sliced bias or None if input is None
    """
    if attn_bias is None:
        return None

    # Handle different bias shapes
    if attn_bias.dim() >= 4:
        # For 4D+ bias tensors, slice along the query dimension
        return attn_bias[..., q_start_idx:q_end_idx, :]
    else:
        # For other shapes, return as is
        return attn_bias
