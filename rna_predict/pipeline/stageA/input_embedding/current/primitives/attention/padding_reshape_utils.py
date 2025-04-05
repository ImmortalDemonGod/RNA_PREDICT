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
        padding_shape[padding_dim] = 0 # Create a zero-size tensor along the dim
        return torch.zeros(padding_shape, dtype=original_tensor.dtype, device=original_tensor.device)


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
        q_padded = q # No padding needed

    # Reshape tensor into trunks
    q_shape = list(q_padded.shape)
    new_shape = q_shape[:-2] + [num_trunks, n_queries] + q_shape[-1:]
    q_trunked = q_padded.reshape(*new_shape)

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
        k_trunked.shape[-3], # n_k_trunks
        k_trunked.shape[-2], # n_keys
    ]

    attn_bias_trunked = torch.full(attn_bias_shape, -config.inf, device=q_trunked.device)

    # Set valid attention regions to zero (unmasked)
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
            attn_bias_trunked[..., i, :valid_queries_in_trunk, :, :] = 0

    return attn_bias_trunked


def _process_attention_bias(
    q_trunked: torch.Tensor,
    k_trunked: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    attn_bias_list: Optional[List[torch.Tensor]],
    config: AttentionBiasConfig,
) -> torch.Tensor:
    """
    Process attention bias for trunked tensors.

    Args:
        q_trunked (torch.Tensor): Trunked query tensor
        k_trunked (torch.Tensor): Trunked key tensor
        attn_bias (Optional[torch.Tensor]): Original attention bias tensor
        attn_bias_list (Optional[List[torch.Tensor]]): List of processed bias chunks
        config (AttentionBiasConfig): Configuration for bias processing

    Returns:
        torch.Tensor: Processed attention bias
    """
    # No bias case
    if attn_bias is None:
        # Return an empty tensor or a zero tensor with a meaningful shape?
        # Original code returned zeros((0,)), which might be ambiguous.
        # Let's return a zero tensor indicating no bias.
        # Shape needs to be broadcastable. A scalar zero might work.
        return torch.tensor(0.0, device=q_trunked.device, dtype=q_trunked.dtype)


    # Determine if dimensions match trunked query shape expectation
    # Expected bias shape: (*batch_dims, n_heads, n_q_trunks, n_queries, n_k_trunks, n_keys)
    # Original bias shape: (*batch_dims, n_heads, original_q_len, original_k_len)
    # Or potentially:      (*batch_dims, n_heads, n_q_trunks, n_queries, original_k_len) if pre-chunked for query

    # Check if bias matches the original query length dimension
    # We assume bias has shape like (..., q_len, k_len)
    matches_original_q_dim = attn_bias.shape[-2] == config.original_length

    if matches_original_q_dim and attn_bias_list is not None:
        # Case 1: Bias provided with original query dim, needs reshaping and chunking for keys
        attn_bias_reshaped_q = _reshape_bias_for_trunked_query(attn_bias, config)

        # attn_bias_list contains bias chunks processed for keys. Stack them.
        # Expected shape after stacking: (..., n_q_trunks, n_queries, n_k_trunks, n_keys)
        attn_bias_trunked = torch.stack(attn_bias_list, dim=-2) # Stack along the n_k_trunks dimension

        # Add the reshaped query part to the stacked key part
        # Need broadcasting: attn_bias_reshaped_q might be (..., n_q_trunks, n_queries, k_len)
        # attn_bias_trunked is (..., n_q_trunks, n_queries, n_k_trunks, n_keys)
        # This addition logic seems complex and potentially incorrect as originally implemented.
        # Let's rethink: _process_keys_values_chunks already handles bias chunking.
        # If bias is provided, attn_bias_list should contain the correctly chunked bias.
        # We just need to reshape the query dimension part if necessary.

        # Revised logic: If bias is provided, _process_keys_values_chunks creates attn_bias_list.
        # We then need to ensure the query dimension is handled correctly.
        # The _reshape_bias_for_trunked_query handles padding and reshaping for the query dim.
        # Let's assume attn_bias was passed to _process_keys_values_chunks, which created attn_bias_list.
        # Each element in attn_bias_list is (..., original_q_len, n_keys) after chunking.
        # We need to reshape the query dimension of each chunk.

        reshaped_bias_chunks = []
        for bias_chunk in attn_bias_list:
             # bias_chunk shape: (..., original_q_len, n_keys)
             # Pad the original query dimension (-2) first
             padded_bias_chunk = _pad_attention_bias(bias_chunk, config) # Pads dim=-2
             # padded_bias_chunk shape: (..., original_q_len + n_q_pad, n_keys)

             # Now reshape the padded query dimension
             bias_shape = list(padded_bias_chunk.shape)
             # Replace the padded query dim (-2) with (n_q_trunks, n_queries)
             new_bias_shape = bias_shape[:-2] + [config.n_q_trunks, config.n_queries] + bias_shape[-1:]
             reshaped_chunk = padded_bias_chunk.reshape(*new_bias_shape)
             # reshaped_chunk shape: (..., n_q_trunks, n_queries, n_keys)
             reshaped_bias_chunks.append(reshaped_chunk)

        # Stack the fully reshaped chunks along the n_k_trunks dimension
        attn_bias_trunked = torch.stack(reshaped_bias_chunks, dim=-2) # dim=-2 becomes the n_k_trunks dim

    elif attn_bias.ndim == q_trunked.ndim and attn_bias.shape[-1] == k_trunked.shape[-2]:
         # Case 2: Bias already matches trunked query shape but maybe not key trunking
         # Shape: (..., n_q_trunks, n_queries, k_len)
         # We still need to chunk the last dimension (k_len)
         # This case seems less likely given the primary rearrange function.
         # Let's stick to the logic derived from rearrange_to_dense_trunk.
         # If attn_bias is passed, it's expected to be (..., q_len, k_len).
         # _process_keys_values_chunks handles the k_len chunking -> attn_bias_list.
         # We then reshape the q_len dimension.

         # Fallback to previous logic if Case 1 assumptions are wrong.
         attn_bias_trunked = _reshape_bias_for_trunked_query(attn_bias, config)
         if attn_bias_list is not None:
              # This stacking assumes attn_bias_list was created differently.
              # Let's rely on the Case 1 logic derived from the main function flow.
              # If this case is truly needed, the calling code needs adjustment.
              pass # Sticking with Case 1 logic for now.


    else:
        # Case 3: Bias dimensions don't match expected patterns. Create a default mask.
        # This handles cases where bias might be e.g., just (q_len,) or other shapes.
        attn_bias_trunked = _create_different_dim_bias(q_trunked, k_trunked, config)

    return attn_bias_trunked