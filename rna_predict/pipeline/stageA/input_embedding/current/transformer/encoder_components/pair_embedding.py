"""
Pair embedding creation logic for the AtomAttentionEncoder.
"""

from typing import Any

import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
    safe_tensor_access,
)


def _process_distances(
    encoder: Any, pair_embed: torch.Tensor, ref_pos: torch.Tensor
) -> torch.Tensor:
    """
    Process distance information for pair embedding.

    Args:
        encoder: The encoder module instance (to access linear_no_bias_d, n_queries, n_keys)
        pair_embed: Initial pair embedding tensor
        ref_pos: Reference positions tensor

    Returns:
        Updated pair embedding tensor with distance features
    """
    # Ensure ref_pos has at least 3 dimensions [..., N_atom, 3]
    if ref_pos.ndim < 2:
        raise ValueError(
            f"ref_pos must have at least 2 dimensions, got shape {ref_pos.shape}"
        )

    num_atoms_in_ref_pos = ref_pos.shape[-2]

    # Process distances between atom pairs
    # Iterate up to the actual number of atoms (N_atom)
    for query_idx in range(num_atoms_in_ref_pos):
        for key_idx in range(num_atoms_in_ref_pos):
            # Get positions of query and key atoms
            # Add checks to prevent out-of-bounds access if ref_pos is smaller than expected
            if query_idx >= num_atoms_in_ref_pos:
                continue  # Should not happen with the outer loop change, but safe guard

            pos_query = ref_pos[..., query_idx, :]
            # key_idx is now guaranteed to be within bounds by the loop range
            pos_key = ref_pos[..., key_idx, :]

            # Calculate distance vector
            dist_vector = pos_query - pos_key

            # Apply distance encoding
            dist_features = encoder.linear_no_bias_d(dist_vector)

            # Update pair embedding (assuming pair_embed has compatible shape)
            # Ensure pair_embed dimensions match query/key indices
            if (
                pair_embed.ndim >= 4
                and query_idx < pair_embed.shape[-3]
                and key_idx < pair_embed.shape[-2]
            ):
                pair_embed[..., query_idx, key_idx, :] += dist_features
            # else:
            # Optional: Add warning or handling if pair_embed shape is incompatible
            # print(f"Warning: Skipping pair_embed update for query {query_idx}, key {key_idx} due to shape mismatch.")

    return pair_embed


def _process_charges(
    encoder: Any, pair_embed: torch.Tensor, ref_charge: torch.Tensor
) -> torch.Tensor:
    """
    Process charge information for pair embedding.

    Args:
        encoder: The encoder module instance (to access linear_no_bias_v)
        pair_embed: Pair embedding tensor with distance features
        ref_charge: Reference charges tensor

    Returns:
        Updated pair embedding tensor with charge features
    """
    # Initialize charge products tensor with the same shape as pair_embed
    charge_products = torch.zeros_like(
        pair_embed[..., :1]
    )  # Keep only one feature dimension

    # Get the actual size of the dimension being indexed
    if ref_charge.ndim < 2:
        raise ValueError(
            f"ref_charge must have at least 2 dimensions, got shape {ref_charge.shape}"
        )
    num_atoms_in_ref_charge = ref_charge.shape[-2]

    # Get the actual number of queries and keys from pair_embed shape
    # n_queries = pair_embed.shape[-3] # No longer using fixed n_queries/n_keys here
    # n_keys = pair_embed.shape[-2]

    # --- Start Dimension Alignment Fix ---
    # Ensure ref_charge has the same leading dimensions as pair_embed (e.g., sample dim)
    aligned_ref_charge = ref_charge
    while aligned_ref_charge.ndim < pair_embed.ndim - 1: # Compare up to atom dims
        aligned_ref_charge = aligned_ref_charge.unsqueeze(1) # Assume missing dim is sample dim (dim 1)
    # --- End Dimension Alignment Fix ---

    # Process charge products between atom pairs
    # Iterate up to the actual number of atoms (N_atom)
    for query_idx in range(num_atoms_in_ref_charge):
        for key_idx in range(num_atoms_in_ref_charge):
            # Get charges of query and key atoms
            # Add bounds check just in case ref_charge is smaller than expected by pair_embed init
            if query_idx >= pair_embed.shape[-3] or key_idx >= pair_embed.shape[-2]:
                continue
            # Use aligned_ref_charge here
            charge_query = aligned_ref_charge[..., query_idx, 0]
            charge_key = aligned_ref_charge[..., key_idx, 0]

            # Calculate charge product
            charge_product = charge_query * charge_key

            # Add to charge products
            # Rely on PyTorch broadcasting to handle potential dimension differences
            charge_products[..., query_idx, key_idx, 0] = charge_product

    # Apply volume encoding to charge products
    volume_features = encoder.linear_no_bias_v(charge_products)

    # Add charge product features to pair embedding
    return pair_embed + volume_features


def create_pair_embedding(
    encoder: Any, input_feature_dict: InputFeatureDict
) -> torch.Tensor:
    """
    Create pair embedding for atom transformer.

    Args:
        encoder: The encoder module instance (to access n_queries, n_keys, c_atompair)
        input_feature_dict: Dictionary of input features

    Returns:
        Pair embedding tensor of shape [batch_size, n_queries, n_keys, c_atompair]
    """
    # Get reference positions and charges
    ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")
    ref_charge = safe_tensor_access(input_feature_dict, "ref_charge")

    # Use config flag for minimal dimensions if present
    minimal_dim = getattr(encoder, 'minimal_pair_embedding_dim', None)
    if minimal_dim is not None:
        # Use the minimal allowed size for dry/test runs
        batch_shape = ref_pos.shape[:-2] if ref_pos is not None else (1,)
        n_atoms = min(ref_pos.shape[-2], minimal_dim) if ref_pos is not None else minimal_dim
        c_atompair = getattr(encoder, 'c_atompair', 1)
        p_lm = torch.zeros((*batch_shape, n_atoms, n_atoms, c_atompair), device=ref_pos.device, dtype=ref_pos.dtype)
        return p_lm

    # Create all-pairs distance tensor
    n_atoms = ref_pos.shape[-2]

    # Initialize the pair embedding tensor based on N_atom x N_atom
    # Shape: [..., N_atom, N_atom, c_atompair]
    p_lm = torch.zeros(
        (*ref_pos.shape[:-2], n_atoms, n_atoms, encoder.c_atompair),  # Use n_atoms
        device=ref_pos.device,
        dtype=ref_pos.dtype,
    )

    # Return empty embedding if there are no atoms
    if n_atoms == 0:
        return p_lm

    # Process distance and charge information
    p_lm = _process_distances(encoder, p_lm, ref_pos)
    p_lm = _process_charges(encoder, p_lm, ref_charge)

    return p_lm
