"""
Tensor shape patch module for RNA prediction pipeline.

This module provides monkey patches for functions in the RNA prediction pipeline
to fix tensor shape compatibility issues. Apply these patches before running the pipeline.

This module now imports and uses the fixes from the unified script.
"""

import torch


def apply_patches():
    """
    Apply all patches to fix tensor shape compatibility issues.
    Call this function before running the pipeline.

    This now imports the unified fixes from the stageD module.
    """
    # Import and apply fixes from the unified script
    from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import apply_tensor_fixes

    # Apply all the tensor fixes
    apply_tensor_fixes()

    print("Applied tensor shape compatibility patches for RNA pipeline")


# Legacy implementation kept for backward compatibility
def adapt_indices_for_gather(
    idx_q: torch.Tensor, idx_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adapt indices for the gather_pair_embedding_in_dense_trunk function.
    This function ensures that indices have the correct shape (2D) for the gather function.

    Args:
        idx_q (torch.Tensor): Query indices, can be 2D [N_b, N_q] or 3D [N_b, N_trunk, N_q]
        idx_k (torch.Tensor): Key indices, can be 2D [N_b, N_k] or 3D [N_b, N_trunk, N_k]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Reshaped indices that are compatible with gather_pair_embedding_in_dense_trunk
    """
    # Convert indices to long type
    idx_q = idx_q.long()
    idx_k = idx_k.long()

    # Handle 3D indices by reshaping them to 2D
    if len(idx_q.shape) == 3:
        N_b, N_trunk, N_q = idx_q.shape
        idx_q = idx_q.reshape(N_b, N_trunk * N_q)
    else:
        assert (
            len(idx_q.shape) == 2
        ), f"Expected idx_q to have 2 or 3 dimensions, got {len(idx_q.shape)}"

    if len(idx_k.shape) == 3:
        N_b, N_trunk, N_k = idx_k.shape
        idx_k = idx_k.reshape(N_b, N_trunk * N_k)
    else:
        assert (
            len(idx_k.shape) == 2
        ), f"Expected idx_k to have 2 or 3 dimensions, got {len(idx_k.shape)}"

    return idx_q, idx_k


def adapt_tensors_for_addition(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adapt tensors for addition operation.
    This function ensures that two tensors have compatible shapes for addition.

    Args:
        tensor_a (torch.Tensor): First tensor
        tensor_b (torch.Tensor): Second tensor

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Reshaped tensors that are compatible for addition
    """
    # If shapes are already compatible, return as is
    if tensor_a.shape == tensor_b.shape:
        return tensor_a, tensor_b

    # Get the shapes
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape

    # For the specific case in the transformer, we know tensor_a is p_lm and tensor_b is z_transformed
    # We need to reshape z_transformed to match p_lm's dimensions at the mismatch point
    if len(shape_a) >= 6 and len(shape_b) >= 6:
        # This is specific to the transformer case where we have:
        # p_lm shape: [1, 1, 1, 10, 10, 10, 16]
        # z_transformed shape: [1, 1, 1, 32, 128, 16]

        # Reshape tensor_b to match tensor_a's dimensions at indices 4 and 5
        # This is a temporary solution for the specific case
        if shape_a[4] != shape_b[4]:
            tensor_b = tensor_b.mean(dim=4, keepdim=True).expand(
                -1, -1, -1, -1, shape_a[4], -1
            )

        if len(shape_a) > 6 and len(shape_b) <= 6:
            # Add an extra dimension to tensor_b
            tensor_b = tensor_b.unsqueeze(5).expand(-1, -1, -1, -1, -1, shape_a[5], -1)

    return tensor_a, tensor_b


# Legacy implementation kept for backward compatibility
def patch_gather_pair_embedding():
    """
    Patch the gather_pair_embedding_in_dense_trunk function to handle 3D indices.
    """
    # Import and use the unified fix
    from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import fix_gather_pair_embedding

    fix_gather_pair_embedding()


# Legacy implementation kept for backward compatibility
def patch_transformer_forward():
    """
    Patch the transformer's forward method to handle tensor shape mismatches.
    """
    # Import and use the unified fixes
    from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import (
        fix_atom_attention_encoder,
        fix_atom_transformer,
    )

    fix_atom_transformer()
    fix_atom_attention_encoder()


if __name__ == "__main__":
    # Apply patches when this module is run directly
    apply_patches()
