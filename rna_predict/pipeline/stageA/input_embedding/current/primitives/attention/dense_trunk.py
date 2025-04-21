"""
Dense trunk operations module for attention mechanisms.

This module contains functions for rearranging tensors into dense trunks
for efficient attention operations. It serves as the main interface,
utilizing helper modules for specific functionalities like configuration,
chunking, padding, masking, and core processing.
"""

import math # Added for ceiling division if needed here, or ensure it's available
from typing import List, Optional, Tuple, Union, Dict, Any # Added Dict, Any

import torch
import logging

logger = logging.getLogger(__name__)

# Imports from local modules within the same package level
# Assuming PaddingInfo is no longer the direct return type
# from ..attention_utils import PaddingInfo
from ..core_transforms import (
    _calculate_trunk_info, # Keep this
    _create_zero_tensor,
    # _handle_return_types, # Remove this import
    _prepare_tensor_lists,
    _validate_dimensions,
    _validate_input_types, # Import TrunkInfo dataclass
)

# Imports from the newly created sub-modules
from .chunk_utils import _process_keys_values_chunks
from .config_types import (
    AttentionBiasConfig,
    DenseTrunkConfig,
    KeysValuesChunkParams,
    PaddingInfoParams,
    RearrangementConfig,
)
from .masking_padding_utils import _prepare_padding_info
from .padding_reshape_utils import (
    _process_attention_bias,
    _reshape_query_to_trunk,
)
# Updated import for the rewritten function
from .trunk_processing import _process_tensor_to_trunks


def _create_empty_output_tensor(
    original_tensor_list: List[torch.Tensor],
    num_trunks: int,
    items_per_trunk: int,
) -> torch.Tensor:
    """
    Creates an empty tensor with the expected output shape but batch dimension 0.
    Shape: (0, NumTrunks, ItemsPerTrunk, D) or similar.
    """
    if not original_tensor_list:
        # Cannot determine dtype, device, or feature dim - return truly empty tensor
        # Or raise error? Let's return a default float tensor on CPU.
        # This case should ideally not happen if input validation is done properly.
        logger.warning("Creating default empty tensor due to empty input list.")
        return torch.empty(0, num_trunks, items_per_trunk, 0, dtype=torch.float32, device='cpu')

    # Infer properties from the first tensor in the original list
    # Find first non-empty tensor if possible
    example_tensor = None
    for t in original_tensor_list:
        if t.numel() > 0:
            example_tensor = t
            break
    if example_tensor is None: # All tensors were empty
         example_tensor = original_tensor_list[0] # Use first one anyway for dtype/device

    dtype = example_tensor.dtype
    device = example_tensor.device
    # Handle case where feature dim might be 0 if input was e.g., (B, S, 0)
    feature_dim_size = example_tensor.shape[-1] if example_tensor.ndim > 0 else 0

    # Handle potential multi-dim tensors (e.g., with heads)
    # Assume batch is dim 0, features are last dim
    other_dims = list(example_tensor.shape[1:-1]) if example_tensor.ndim > 2 else []

    # Construct shape: (0, *other_dims_before_seq, num_trunks, items_per_trunk, *other_dims_after_seq_if_any, feature_dim_size)
    # Simpler assumption based on typical use: (0, num_trunks, items_per_trunk, D)
    # Let's refine based on expected output structure of _process_tensor_to_trunks
    # Expected output per tensor: (B, *OtherDimsBetweenB&S, NumTrunks, ItemsPerTrunk, *OtherDimsBetweenS&D, D)
    # We need the shape for batch size 0.
    # Example: Input (B, S, D) -> Output (B, NumTrunks, ItemsPerTrunk, D) -> Empty (0, NumTrunks, ItemsPerTrunk, D)
    # Example: Input (B, H, S, D) -> Output (B, H, NumTrunks, ItemsPerTrunk, D) -> Empty (0, H, NumTrunks, ItemsPerTrunk, D)

    empty_shape = [0] + other_dims # Start with 0 batch, add intermediate dims (like heads)
    empty_shape.extend([num_trunks, items_per_trunk, feature_dim_size]) # Add trunk dims and feature dim

    return torch.empty(empty_shape, dtype=dtype, device=device)


def rearrange_qk_to_dense_trunk(
    q: Union[torch.Tensor, List[torch.Tensor]],
    k: Union[torch.Tensor, List[torch.Tensor]],
    dim_q: Union[int, List[int]],
    dim_k: Union[int, List[int]],
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: # Updated return type hint
    """
    Rearrange query and key tensors into dense trunks. Handles list inputs by
    processing each tensor and concatenating the results along the batch dimension.

    Args:
        q: Query tensor or list of tensors.
        k: Key tensor or list of tensors.
        dim_q: Query sequence dimension(s) to rearrange.
        dim_k: Key sequence dimension(s) to rearrange.
        n_queries: Number of queries per trunk. Defaults to 32.
        n_keys: Number of keys per trunk. Defaults to 128.
        compute_mask: Whether to compute masks. Defaults to True.

    Returns:
        Rearranged query tensor (concatenated), rearranged key tensor (concatenated),
        and padding information dictionary.
    """
    # Step 1: Validate inputs
    if n_queries <= 0:
        raise ValueError(f"n_queries must be positive, got {n_queries}")
    if n_keys <= 0:
        raise ValueError(f"n_keys must be positive, got {n_keys}")

    q_is_list, k_is_list = _validate_input_types(q, k)
    q_list, k_list = _prepare_tensor_lists(q, k, q_is_list, k_is_list)

    # Step 2: Handle dimension lists
    if isinstance(dim_q, int):
        dim_q_list = [dim_q] * len(q_list) # Replicate dim for each tensor in list
    else: # dim_q is already a list
        dim_q_list = dim_q
    if isinstance(dim_k, int):
        dim_k_list = [dim_k] * len(k_list) # Replicate dim for each tensor in list
    else: # dim_k is already a list
        dim_k_list = dim_k

    # Step 3: Validate dimensions match list lengths
    _validate_dimensions(dim_q_list, dim_k_list, q_list, k_list)

    # Step 4: Calculate overall trunk information (needed for padding info and empty tensor shapes)
    # This still uses the original lists and dims to get total lengths
    trunk_info = _calculate_trunk_info(
        q_list, k_list, dim_q_list, dim_k_list, n_queries, n_keys
    )

    # Step 5: Process queries and keys into trunks independently
    # The rewritten function handles padding, zero-length, and reshaping per tensor
    q_processed_list = _process_tensor_to_trunks(
        q_list, dim_q_list, n_queries # Pass items_per_trunk
    )
    k_processed_list = _process_tensor_to_trunks(
        k_list, dim_k_list, n_keys # Pass items_per_trunk
    )

    # Step 6: Concatenate processed tensors along the batch dimension
    if q_processed_list:
        # Check for inconsistent ranks before concatenation
        if len(q_processed_list) > 1: # Only check if there's more than one tensor
            # Find first tensor with non-zero numel to determine expected rank
            first_valid_q = next((t for t in q_processed_list if t.numel() > 0), None)
            if first_valid_q is not None:
                expected_rank = first_valid_q.ndim
                if not all(t.ndim == expected_rank for t in q_processed_list if t.numel() > 0):
                     ranks = [t.ndim for t in q_processed_list]
                     shapes = [t.shape for t in q_processed_list]
                     raise RuntimeError(f"Inconsistent ranks found in q_processed_list before concatenation. Ranks: {ranks}, Shapes: {shapes}")
            # If all tensors are empty, concatenation might still work if shapes match except dim 0
        try:
            # Filter out tensors with 0 batch size before concatenating if necessary?
            # Let's assume _process_tensor_to_trunks returns tensors with original batch size,
            # even if seq_len is 0 (e.g., shape (B, 0, trunk_size, D)).
            q_result = torch.cat(q_processed_list, dim=0)
        except RuntimeError as e:
            shapes = [t.shape for t in q_processed_list]
            raise RuntimeError(f"Failed to concatenate q_processed_list. Shapes: {shapes}. Error: {e}") from e
    else:
        # Create an empty tensor with the correct output shape (batch dim 0)
        q_result = _create_empty_output_tensor(q_list, trunk_info.n_q_trunks, n_queries)

    if k_processed_list:
        if len(k_processed_list) > 1: # Only check if there's more than one tensor
            first_valid_k = next((t for t in k_processed_list if t.numel() > 0), None)
            if first_valid_k is not None:
                expected_rank = first_valid_k.ndim
                if not all(t.ndim == expected_rank for t in k_processed_list if t.numel() > 0):
                     ranks = [t.ndim for t in k_processed_list]
                     shapes = [t.shape for t in k_processed_list]
                     raise RuntimeError(f"Inconsistent ranks found in k_processed_list before concatenation. Ranks: {ranks}, Shapes: {shapes}")
        try:
            # Assume tensors have original batch size, even if seq_len is 0
            k_result = torch.cat(k_processed_list, dim=0)
        except RuntimeError as e:
            shapes = [t.shape for t in k_processed_list]
            raise RuntimeError(f"Failed to concatenate k_processed_list. Shapes: {shapes}. Error: {e}") from e
    else:
        # Create an empty tensor with the correct output shape (batch dim 0)
        k_result = _create_empty_output_tensor(k_list, trunk_info.n_k_trunks, n_keys)


    # Step 7: Prepare padding information (uses original lists and dims)
    padding_info_params = PaddingInfoParams(
        q_list=q_list, # Use original list
        k_list=k_list, # Use original list
        dim_q_list=dim_q_list,
        dim_k_list=dim_k_list,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
        trunk_info=trunk_info, # Use calculated trunk_info
    )
    # _prepare_padding_info now returns a Dict
    padding_info: Dict[str, Any] = _prepare_padding_info(padding_info_params)

    # Step 8: Return concatenated results and padding info dictionary
    return q_result, k_result, padding_info


def _is_small_tensor_case(
    q: torch.Tensor, k: torch.Tensor, config: DenseTrunkConfig
) -> bool:
    """
    Check if tensor dimensions are small enough to skip trunking.

    Args:
        q: Query tensor.
        k: Key tensor.
        config: Configuration for dense trunk operations.

    Returns:
        True if tensors are small enough, False otherwise.
    """
    # Handle zero-length tensors: they should not trigger the bypass
    if q.shape[-2] == 0 or k.shape[-2] == 0:
        return False
    return q.shape[-2] <= config.n_queries and k.shape[-2] <= config.n_keys


def _handle_small_tensors(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: DenseTrunkConfig
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """
    Handle small tensors where trunking is not needed.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        config: Configuration for dense trunk operations.

    Returns:
        Result tuple if tensors are small enough, None otherwise.
    """
    if not _is_small_tensor_case(q, k, config):
        return None

    # For small tensors, no need for trunking - use as-is
    # However, the expected output format might still require a trunk dimension of 1.
    # Let's reshape to add the trunk dimension for consistency with non-bypass cases.
    # Shape: (B, S, D) -> (B, 1, S, D) assuming S is dim -2
    # q_trunked = q.unsqueeze(-3) # Add trunk dim before sequence dim
    # k_trunked = k.unsqueeze(-3)
    # v_trunked = v.unsqueeze(-3)


    # Handle attention bias
    # attn_bias_trunked: torch.Tensor
    # if config.attn_bias is not None:
    #     # Also add trunk dimension to bias: (B, S_q, S_k) -> (B, 1, S_q, S_k)
    #     attn_bias_trunked = config.attn_bias.unsqueeze(-3)
    # else:
    #     # Create a zero tensor, but match the expected rank if possible
    #     # Expected bias shape: (B, NumQTrunks=1, NumQItems=S_q, PaddedKLen=S_k)
    #     # Let's create a scalar zero for simplicity, downstream might handle broadcasting.
    #     # Or create (B, 1, S_q, S_k) zero tensor?
    #     # The test `test_rearrange_to_dense_trunk_small_tensor_bypass` expects a scalar 0.
    #     attn_bias_trunked = _create_zero_tensor([1], dtype=q.dtype, device=q.device) # Scalar 0

    # Padding length is 0 for bypass case
    # padding_length = 0

    # Return tuple consistent with non-bypass case signature
    # Note: The previous test assertion checked for identity (q_out is q), which is no longer true due to unsqueeze.
    # The test needs adjustment if this bypass logic is kept.
    # Let's revert to the original bypass logic for now to match the existing test expectations.
    q_trunked, k_trunked, v_trunked = q, k, v
    attn_bias_trunked = (
         config.attn_bias
         if config.attn_bias is not None
         else _create_zero_tensor([1], dtype=q.dtype, device=q.device) # Scalar zero based on test
     )

    return q_trunked, k_trunked, v_trunked, attn_bias_trunked, 0


def _rearrange_to_dense_trunk_impl(
    config: RearrangementConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Implementation of dense trunk rearrangement with bundled parameters.

    Args:
        config: Configuration for rearrangement.

    Returns:
        Trunked query, key, value, attention bias, and padding length.
    """
    # Validate inputs
    if config.n_queries <= 0:
        raise ValueError(f"n_queries must be positive, got {config.n_queries}")
    if config.n_keys <= 0:
        raise ValueError(f"n_keys must be positive, got {config.n_keys}")
    if config.k.shape[-2] != config.v.shape[-2]:
         # Add check for K/V sequence length mismatch if required by logic
         logger.warning(f"K sequence length ({config.k.shape[-2]}) differs from V sequence length ({config.v.shape[-2]})")
         # raise ValueError("K and V sequence lengths must match for trunking") # Uncomment if strict matching is needed

    # Create configuration object for trunk operations
    trunk_config = DenseTrunkConfig(
        n_queries=config.n_queries,
        n_keys=config.n_keys,
        attn_bias=config.attn_bias,
        inf=config.inf,
    )

    # Check for small tensor case (optimization)
    small_tensors_result = _handle_small_tensors(
        config.q, config.k, config.v, trunk_config
    )
    if small_tensors_result is not None:
        # Ensure the returned tensors match the expected type hint structure
        q_out, k_out, v_out, bias_out, pad_len = small_tensors_result
        # The test expects specific shapes/types, ensure bypass returns match
        # Based on test `test_rearrange_to_dense_trunk_small_tensor_bypass`,
        # it expects original tensors and scalar zero bias if None provided.
        return q_out, k_out, v_out, bias_out, pad_len


    # --- Non-Bypass Case ---
    # Process query tensor using imported function
    # Assuming _reshape_query_to_trunk handles padding and returns (B, n_q_trunks, n_queries, D)
    query_info = _reshape_query_to_trunk(config.q, config.n_queries)
    q_trunked = query_info.trunked_tensor
    q_padding_length = query_info.padding_length # Padding applied to Q

    # Calculate number of key trunks (V follows K)
    k_seq_len = config.k.shape[-2]
    config.v.shape[-2] # Use actual V length
    n_k_trunks = math.ceil(k_seq_len / config.n_keys) if k_seq_len > 0 else 0
    # V trunking depends on K's trunking structure
    # Padded length for K determines V's effective length for trunking
    k_padded_len = n_k_trunks * config.n_keys
    k_padding_needed = k_padded_len - k_seq_len

    # Process key and value tensors using imported function
    # This function needs to handle padding internally based on n_keys
    k_trunked, v_trunked, attn_bias_list = _process_keys_values_chunks(
        KeysValuesChunkParams(
            k=config.k,
            v=config.v,
            n_keys=config.n_keys,
            n_k_trunks=n_k_trunks, # Pass calculated num trunks
            attn_bias=config.attn_bias, # Pass original bias
            inf=config.inf,
        )
    )

    # Create bias configuration - ADDED n_k_trunks, n_keys, n_k_pad
    bias_config = AttentionBiasConfig(
        n_q_trunks=query_info.num_trunks,
        n_queries=config.n_queries,
        n_q_pad=q_padding_length, # Use Q padding length
        original_length=config.q.shape[-2], # Original Q length
        inf=config.inf,
        n_k_trunks=n_k_trunks, # ADDED
        n_keys=config.n_keys,   # ADDED
        n_k_pad=k_padding_needed, # ADDED
    )
    # Remove the hasattr check as fields are now part of the class
    # if hasattr(bias_config, 'n_keys'):
    #      bias_config.n_keys = config.n_keys


    # Process attention bias using imported function
    attn_bias_trunked = _process_attention_bias(
        q_trunked, k_trunked, config.attn_bias, attn_bias_list, bias_config
    )

    # Add assertion here to check type before returning
    assert isinstance(q_padding_length, int), f"Type mismatch: q_padding_length is {type(q_padding_length)}, expected int"

    # Return Q padding length (should be int) directly
    return q_trunked, k_trunked, v_trunked, attn_bias_trunked, q_padding_length


def rearrange_to_dense_trunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
    inf: float = 1e10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Rearrange query, key, and value tensors into dense trunks for efficient attention.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        n_queries: Number of queries per trunk.
        n_keys: Number of keys per trunk.
        attn_bias: Attention bias tensor. Defaults to None.
        inf: Value for masked positions. Defaults to 1e10.

    Returns:
        Trunked query, key, value, attention bias, and padding length (for query).
    """
    # Use config object to bundle parameters
    config = RearrangementConfig(
        q=q, k=k, v=v, n_queries=n_queries, n_keys=n_keys, attn_bias=attn_bias, inf=inf
    )
    return _rearrange_to_dense_trunk_impl(config)
