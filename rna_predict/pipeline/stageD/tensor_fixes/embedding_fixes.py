"""
Embedding-related tensor shape compatibility fixes.
"""

from functools import wraps

import torch


def fix_gather_pair_embedding():
    """
    Fix the gather_pair_embedding_in_dense_trunk function to handle 3D indices.
    """
    from rna_predict.pipeline.stageA.input_embedding.current import primitives

    original_gather = primitives.gather_pair_embedding_in_dense_trunk

    @wraps(original_gather)
    def patched_gather(x, idx_q, idx_k):
        # Convert indices to long type
        idx_q = idx_q.long()
        idx_k = idx_k.long()

        # Handle 3D indices
        if idx_q.dim() == 3:
            # Reshape to 2D for gathering
            batch_size, n_queries, n_keys = idx_q.shape
            x = x.unsqueeze(1).expand(-1, n_queries, -1, -1)
            idx_q = idx_q.reshape(batch_size * n_queries, n_keys)
            idx_k = idx_k.reshape(batch_size * n_queries, n_keys)

            # Gather and reshape back
            result = original_gather(x, idx_q, idx_k)
            return result.reshape(batch_size, n_queries, n_keys, -1)

        return original_gather(x, idx_q, idx_k)

    primitives.gather_pair_embedding_in_dense_trunk = patched_gather


def fix_broadcast_token_to_atom():
    """
    Fix broadcast_token_to_atom to safely handle indices.
    """
    # Corrected import from utils instead of primitives
    from rna_predict.pipeline.stageA.input_embedding.current import utils

    original_broadcast = utils.broadcast_token_to_atom

    @wraps(original_broadcast)
    def patched_broadcast(x_token: torch.Tensor, atom_to_token_idx: torch.Tensor):
        # First, ensure atom_to_token_idx indices are valid for x_token's shape
        if atom_to_token_idx.max() >= x_token.shape[1]:
            # Clip indices to valid range
            atom_to_token_idx = torch.clamp(atom_to_token_idx, 0, x_token.shape[1] - 1)

        return original_broadcast(x_token, atom_to_token_idx)

    # Assign patched function back to the correct module
    utils.broadcast_token_to_atom = patched_broadcast


def fix_batched_gather():
    """
    Fix batched_gather to safely handle indices.
    """
    # Corrected import from utils instead of primitives
    from rna_predict.pipeline.stageA.input_embedding.current import utils

    original_batched_gather = utils.batched_gather

    @wraps(original_batched_gather)
    def patched_batched_gather(data, inds, dim=0, no_batch_dims=0):
        # Check if indices are out of bounds for the dim we're gathering from
        # Add check for non-empty inds tensor
        if inds.numel() > 0 and inds.max() >= data.shape[dim]:
            # Clip indices to valid range
            inds = torch.clamp(inds, 0, data.shape[dim] - 1)

        return original_batched_gather(data, inds, dim, no_batch_dims)

    # Assign patched function back to the correct module
    utils.batched_gather = patched_batched_gather
