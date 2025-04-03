"""
Tensor shape compatibility fixes for the RNA prediction pipeline.
"""

from functools import wraps

import torch

# Import utils directly here
from rna_predict.pipeline.stageA.input_embedding.current import utils as embedding_utils

from .attention_fixes import fix_attention_bias_shape, fix_rearrange_qk_to_dense_trunk
from .diffusion_fixes import (
    fix_token_indices_after_resize,
    # fix_trunk_feature_dimensions, # Removed incorrect patch
)

# Remove fix_batched_gather from embedding_fixes import
from .embedding_fixes import fix_broadcast_token_to_atom, fix_gather_pair_embedding
from .tensor_operations import fix_matrix_multiplication, fix_tensor_add
from .transformer_fixes import (
    fix_adaptive_layernorm,
    fix_atom_attention_encoder,
    fix_atom_transformer,
)


def apply_tensor_fixes():
    """
    Apply all tensor shape compatibility fixes needed for the model to function properly.
    This should be called before running the diffusion model.
    """
    # Fix tensor operations
    fix_tensor_add()
    fix_matrix_multiplication()

    # Fix attention components
    fix_attention_bias_shape()
    fix_rearrange_qk_to_dense_trunk()

    # Fix transformer components
    fix_atom_transformer()
    fix_atom_attention_encoder()
    fix_adaptive_layernorm()

    # Fix embedding and reshaping functions
    fix_gather_pair_embedding()
    fix_broadcast_token_to_atom()
    # fix_batched_gather() # Remove call to external fix function

    # --- Apply batched_gather fix directly ---
    original_batched_gather = embedding_utils.batched_gather

    @wraps(original_batched_gather)
    def patched_batched_gather(data, inds, dim=0, no_batch_dims=0):
        # Check if indices are out of bounds for the dim we're gathering from
        # Add check for non-empty inds tensor
        if inds.numel() > 0 and inds.max() >= data.shape[dim]:
            # Clip indices to valid range
            inds = torch.clamp(inds, 0, data.shape[dim] - 1)

        return original_batched_gather(data, inds, dim, no_batch_dims)

    # Assign patched function back to the correct module
    embedding_utils.batched_gather = patched_batched_gather
    # --- End batched_gather fix ---

    # Fix diffusion components
    fix_token_indices_after_resize()
    # fix_trunk_feature_dimensions() # Removed call to incorrect patch

    print("Applied all tensor shape compatibility fixes")
