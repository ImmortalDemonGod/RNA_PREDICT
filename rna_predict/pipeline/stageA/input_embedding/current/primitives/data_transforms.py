"""
Data transformation module for neural network operations.

This module contains functions for transforming, rearranging, and manipulating
tensor data for neural network operations, particularly for attention mechanisms.

This file serves as a façade that imports and re-exports functionality from
specialized modules to maintain backward compatibility.
"""

# Re-export from core_transforms
from .core_transforms import (
    RearrangeConfig,
    TrunkInfo,
    MaskCreationConfig,
    RearrangeInputConfig,
    _validate_input_types,
    _prepare_tensor_lists,
    _validate_dimensions,
    _calculate_trunk_info,
    _apply_trunk_slices,
    _handle_return_types,
    _create_zero_tensor,
)

# Re-export from attention_transforms
from .attention_transforms import (
    DenseTrunkConfig,
    AttentionBiasConfig,
    MaskSliceInfo,
    QueryTrunkInfo,
    ChunkInfo,
    _init_mask_slices,
    _calculate_mask_slice,
    _create_masks,
    _process_queries,
    _process_keys,
    _prepare_padding_info,
    rearrange_qk_to_dense_trunk,
    _is_small_tensor_case,
    _handle_small_tensors,
    _calculate_padding_needed,
    _create_padding_tensor,
    _reshape_query_to_trunk,
    _get_chunk_info,
    _process_chunk,
    _process_bias_chunk,
    _process_keys_values_chunks,
    _pad_attention_bias,
    _reshape_bias_for_trunked_query,
    _create_different_dim_bias,
    _process_attention_bias,
    rearrange_to_dense_trunk,
)

# Re-export from atom_pair_transforms
from .atom_pair_transforms import (
    AtomPairConfig,
    _validate_token_feats_shape,
    _map_tokens_to_atoms,
    broadcast_token_to_local_atom_pair,
    gather_pair_embedding_in_dense_trunk,
)
