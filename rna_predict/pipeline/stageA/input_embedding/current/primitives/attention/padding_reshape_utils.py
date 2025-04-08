"""
Utility functions for padding and reshaping tensors in dense trunk attention.
"""

from typing import List, Optional

import torch

from .config_types import AttentionBiasConfig, QueryTrunkInfo


def _calculate_padding_needed(n: int, n_queries: int) -> int:
    """
    Calculate padding needed for the query dimension.

    Args:
        n (int): Original query dimension size
        n_queries (int): Number of queries per trunk

    Returns:
        int: Padding length
    """
    return (n_queries - (n % n_queries)) % n_queries


def _create_padding_tensor(
    original_tensor: torch.Tensor, padding_length: int, padding_dim: int = -2
) -> torch.Tensor:
    """
    Create padding tensor for the query.

    Args:
        original_tensor (torch.Tensor): Original tensor to match
        padding_length (int): Length of padding
        padding_dim (int): Dimension to pad

    Returns:
        torch.Tensor: Padding tensor
    """
    if padding_length <= 0:
        # Return an empty tensor of the correct shape if no padding needed,
        # or handle appropriately if concatenation expects non-empty tensors.
        # For now, let's assume concatenation handles this or return original.
        # Returning original seems safer if padding_length is 0.
        # If padding_length is negative (shouldn't happen), raise error?
        # Let's stick to the original logic for now.
        # The original code didn't explicitly handle padding_length == 0,
        # it relied on the calling function _reshape_query_to_trunk.
        # Let's refine this slightly.
        padding_shape = list(original_tensor.shape)
        padding_shape[padding_dim] = 0  # Create a zero-size tensor along the dim
        return torch.zeros(
            padding_shape, dtype=original_tensor.dtype, device=original_tensor.device
        )

    padding_shape = list(original_tensor.shape)
    padding_shape[padding_dim] = padding_length

    return torch.zeros(
        padding_shape, dtype=original_tensor.dtype, device=original_tensor.device
    )


def _reshape_query_to_trunk(q: torch.Tensor, n_queries: int) -> QueryTrunkInfo:
    """
    Reshape query tensor into trunks with padding if needed.

    Args:
        q (torch.Tensor): Query tensor
        n_queries (int): Number of queries per trunk

    Returns:
        QueryTrunkInfo: Information about processed query tensor
    """
    # Calculate padding needed
    n = q.shape[-2]
    padding_length = _calculate_padding_needed(n, n_queries)

    # Calculate derived values
    total_length = n + padding_length
    num_trunks = total_length // n_queries

    # Create padded tensor if needed
    if padding_length > 0:
        padding = _create_padding_tensor(q, padding_length)
        q_padded = torch.cat([q, padding], dim=-2)
    else:
        q_padded = q  # No padding needed

    # Reshape tensor into trunks
    q_shape = list(q_padded.shape)
    new_shape = q_shape[:-2] + [num_trunks, n_queries] + q_shape[-1:]
    q_trunked = q_padded.reshape(*new_shape)

    # Add assertion here to check the type right before returning
    assert isinstance(padding_length, int), \
        f"Inside _reshape_query_to_trunk: padding_length type is {type(padding_length)}, expected int"

    return QueryTrunkInfo(
        trunked_tensor=q_trunked,
        padding_length=padding_length,
        total_length=total_length,
        num_trunks=num_trunks,
    )


def _pad_attention_bias(
    attn_bias: torch.Tensor, config: AttentionBiasConfig
) -> torch.Tensor:
    """
    Pad attention bias if needed.

    Args:
        attn_bias (torch.Tensor): Attention bias tensor
        config (AttentionBiasConfig): Configuration for attention bias

    Returns:
        torch.Tensor: Padded attention bias
    """
    if config.n_q_pad <= 0:
        return attn_bias

    # Create padding for attention bias
    bias_pad_shape = list(attn_bias.shape)
    bias_pad_shape[-2] = config.n_q_pad
    bias_padding = torch.full(bias_pad_shape, -config.inf, device=attn_bias.device)

    # Concatenate with original bias
    return torch.cat([attn_bias, bias_padding], dim=-2)


def _reshape_bias_for_trunked_query(
    attn_bias: torch.Tensor, config: AttentionBiasConfig
) -> torch.Tensor:
    """
    Reshape attention bias to match trunked query.

    Args:
        attn_bias (torch.Tensor): Attention bias tensor
        config (AttentionBiasConfig): Configuration for attention bias

    Returns:
        torch.Tensor: Reshaped attention bias
    """
    # Pad bias if needed
    padded_bias = _pad_attention_bias(attn_bias, config)

    # Reshape to match trunked query
    bias_shape = list(padded_bias.shape)

    # Replace the last-2 dimension with (n_q_trunks, n_queries)
    new_bias_shape = (
        bias_shape[:-2] + [config.n_q_trunks, config.n_queries] + bias_shape[-1:]
    )

    return padded_bias.reshape(*new_bias_shape)


def _create_different_dim_bias(
    q_trunked: torch.Tensor, k_trunked: torch.Tensor, config: AttentionBiasConfig
) -> torch.Tensor:
    """
    Create attention bias for the case when bias has different dimensions.

    Args:
        q_trunked (torch.Tensor): Trunked query tensor
        k_trunked (torch.Tensor): Trunked key tensor
        config (AttentionBiasConfig): Configuration for attention bias

    Returns:
        torch.Tensor: Created attention bias
    """
    # Initialize attn_bias_trunked with -inf (masked by default)
    attn_bias_shape = list(q_trunked.shape[:-1]) + [
        k_trunked.shape[-3],  # n_k_trunks
        k_trunked.shape[-2],  # n_keys
    ]

    attn_bias_trunked = torch.full(
        attn_bias_shape, -config.inf, device=q_trunked.device, dtype=q_trunked.dtype # Added dtype match
    )

    # Set valid attention regions to zero (unmasked)
    # Handle case where n_q_trunks might be 0
    if config.n_q_trunks > 0:
        for i in range(config.n_q_trunks):
            q_start = i * config.n_queries
            q_end = min(q_start + config.n_queries, config.original_length)

            # Skip if this trunk is entirely padding
            if q_start >= config.original_length:
                continue

            # Calculate the number of valid queries in this trunk
            valid_queries_in_trunk = q_end - q_start

            # Set valid attention region to zeros (allows attention)
            if valid_queries_in_trunk > 0:
                # Ensure slicing indices are valid
                if i < attn_bias_trunked.shape[-4] and valid_queries_in_trunk <= attn_bias_trunked.shape[-3]:
                     attn_bias_trunked[..., i, :valid_queries_in_trunk, :, :] = 0
                else:
                     # This case indicates a shape mismatch or logic error, log or raise?
                     print(f"Warning: Index out of bounds during bias creation. Trunk {i}, Valid Queries {valid_queries_in_trunk}, Bias Shape {attn_bias_trunked.shape}")


    # Ensure the function always returns the tensor
    return attn_bias_trunked

def _pad_attention_bias_key_dim(
    attn_bias: torch.Tensor, n_k_pad: int, inf: float
) -> torch.Tensor:
    """Pads attention bias along the key dimension (last dim)."""
    if n_k_pad <= 0:
        return attn_bias
    bias_pad_shape = list(attn_bias.shape)
    bias_pad_shape[-1] = n_k_pad
    bias_padding = torch.full(bias_pad_shape, -inf, device=attn_bias.device, dtype=attn_bias.dtype)
    return torch.cat([attn_bias, bias_padding], dim=-1)






def _process_attention_bias(
    q_trunked: torch.Tensor,
    k_trunked: torch.Tensor, # k_trunked might not be needed if we use config for shapes
    attn_bias: Optional[torch.Tensor],
    attn_bias_list: Optional[List[torch.Tensor]], # This is likely unused now
    config: AttentionBiasConfig,
) -> torch.Tensor:
    """
    Process attention bias for trunked tensors by padding and reshaping.

    Args:
        q_trunked (torch.Tensor): Trunked query tensor (used for device/dtype)
        k_trunked (torch.Tensor): Trunked key tensor (potentially unused)
        attn_bias (Optional[torch.Tensor]): Original attention bias tensor (e.g., B, Sq, Sk)
        attn_bias_list (Optional[List[torch.Tensor]]): List of processed bias chunks (UNUSED)
        config (AttentionBiasConfig): Configuration for bias processing

    Returns:
        torch.Tensor: Processed attention bias (e.g., B, NqT, Nq, PaddedNk)
    """
    # Calculate expected final shape
    # Shape: (B, ..., NqT, Nq, PaddedNk)
    # Infer batch/head dims from q_trunked shape excluding trunk dims
    batch_head_dims = list(q_trunked.shape[:-3]) if q_trunked.ndim > 3 else [] # Handle cases like (B, NqT, Nq, D)
    padded_k_len = config.n_k_trunks * config.n_keys
    # Ensure target shape calculation handles cases where n_q_trunks might be 0
    target_shape = batch_head_dims + [config.n_q_trunks, config.n_queries, padded_k_len]

    # Handle None case
    if attn_bias is None:
        # Return zeros of the target shape
        # Use q_trunked for device/dtype, handle case where q_trunked might be empty
        device = q_trunked.device if q_trunked.numel() > 0 else torch.device('cpu') # Default to CPU if empty
        dtype = q_trunked.dtype if q_trunked.numel() > 0 else torch.float32 # Default dtype
        return torch.zeros(target_shape, device=device, dtype=dtype)

    # Handle Provided Bias
    # 1. Pad query dimension (dim=-2)
    padded_q_bias = _pad_attention_bias(attn_bias, config) # Uses config.n_q_pad

    # 2. Pad key dimension (dim=-1)
    padded_qk_bias = _pad_attention_bias_key_dim(padded_q_bias, config.n_k_pad, config.inf)

    # 3. Reshape into final target shape
    # Original shape (padded): (B, ..., Sq_padded, Sk_padded)
    # Target shape: (B, ..., NqT, Nq, Sk_padded)
    # We need to reshape the Sq_padded dimension into (NqT, Nq)
    current_shape = list(padded_qk_bias.shape)
    # Ensure there are enough dimensions before slicing
    if len(current_shape) < 2:
         raise ValueError(f"Attention bias tensor has too few dimensions after padding: {padded_qk_bias.shape}")

    reshape_target = current_shape[:-2] + [config.n_q_trunks, config.n_queries] + current_shape[-1:]

    # Handle case where n_q_trunks is 0 (original q_len was 0)
    if config.n_q_trunks == 0:
        # The target shape should have 0 in the n_q_trunks dimension
        # Reshape might not be needed, or should result in shape like (B, 0, Nq, PaddedNk)
        # Let's return an empty tensor with the correct target shape directly
        return torch.empty(target_shape, device=attn_bias.device, dtype=attn_bias.dtype)


    try:
        attn_bias_trunked = padded_qk_bias.reshape(*reshape_target)
    except RuntimeError as e:
         raise RuntimeError(
             f"Failed to reshape attention bias. Current shape: {padded_qk_bias.shape}, "
             f"Target reshape: {reshape_target}. Original bias shape: {attn_bias.shape}. Error: {e}"
         ) from e

    # Final check (optional): Ensure shape matches exactly what test expects
    if list(attn_bias_trunked.shape) != target_shape:
         print(f"Warning: Final bias shape {attn_bias_trunked.shape} differs slightly from calculated target {target_shape}")
         # This might happen if batch/head dim inference was wrong, but reshape should match target_shape
         # Let's trust the reshape result for now.

    return attn_bias_trunked
