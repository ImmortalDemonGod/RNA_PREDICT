import torch
from typing import List, Optional
import logging

logger = logging.getLogger("rna_predict.pipeline.stageD.diffusion.bridging.hybrid")


def debug_print_hybrid_bridging_inputs(
    stage_c_output: torch.Tensor,
    residue_atom_map: List[List[int]],
    n_residues: int,
    canonical_atom_count: int,
    feature_dim: int,
    batch_size: Optional[int],
    fill_value: float,
    device: Optional[torch.device],
    debug_logging: bool = False,
):
    """
    Prints debug information for hybrid bridging inputs if debug_logging is True.
    The debug_logging argument should be set from the Hydra config (e.g., cfg.model.stageD.diffusion.debug_logging).
    """
    logger.debug("[ENTRY] debug_print_hybrid_bridging_inputs called. debug_logging: %s", debug_logging)
    if not debug_logging:
        return
    logger.debug("[hybrid_bridging_sparse_to_dense] stage_c_output.shape: %s", getattr(stage_c_output, 'shape', None))
    logger.debug("[hybrid_bridging_sparse_to_dense] n_residues: %s", n_residues)
    logger.debug("[hybrid_bridging_sparse_to_dense] canonical_atom_count: %s", canonical_atom_count)
    logger.debug("[hybrid_bridging_sparse_to_dense] feature_dim: %s", feature_dim)
    logger.debug("[hybrid_bridging_sparse_to_dense] batch_size: %s", batch_size)
    logger.debug("[hybrid_bridging_sparse_to_dense] fill_value: %s", fill_value)
    logger.debug("[hybrid_bridging_sparse_to_dense] device: %s", device)
    logger.debug("[hybrid_bridging_sparse_to_dense] len(residue_atom_map): %s", len(residue_atom_map))
    if len(residue_atom_map) > 0:
        logger.debug("[hybrid_bridging_sparse_to_dense] residue_atom_map[0:5]: %s", residue_atom_map[:5])
    else:
        logger.debug("[hybrid_bridging_sparse_to_dense] residue_atom_map is empty!")


def hybrid_bridging_sparse_to_dense(
    stage_c_output: torch.Tensor,
    residue_atom_map: List[List[int]],
    n_residues: int,
    canonical_atom_count: int,
    feature_dim: int,
    batch_size: Optional[int] = 1,
    fill_value: float = 0.0,
    device: Optional[torch.device] = None,
    debug_logging: bool = False,
):
    """
    Fill a dense atom array with predicted values from Stage C where available, and mask the rest, using residue_to_atoms logic.
    Args:
        stage_c_output: torch.Tensor of shape [batch_size, n_residues, N_sparse_atoms_per_res, feature_dim] or [n_residues, N_sparse_atoms_per_res, feature_dim]
        residue_atom_map: List[List[int]] mapping residue index to atom indices
        n_residues: Number of residues
        canonical_atom_count: Number of atoms per residue (should match mapping)
        feature_dim: Feature dimension
        batch_size: Batch size
        fill_value: Value to fill for missing atoms
        device: torch.device
        debug_logging: Whether to print debug output (should be set from Hydra config)
    Returns:
        dense_atoms: [batch_size, n_residues, canonical_atom_count, feature_dim]
        mask: [batch_size, n_residues, canonical_atom_count]
    """
    logger.debug("[ENTRY] hybrid_bridging_sparse_to_dense called. debug_logging: %s", debug_logging)
    if debug_logging:
        logger.debug("[hybrid_bridging_sparse_to_dense] ENTRY")
        logger.debug(f"  stage_c_output.shape: {getattr(stage_c_output, 'shape', None)}")
        logger.debug(f"  n_residues: {n_residues}, canonical_atom_count: {canonical_atom_count}, feature_dim: {feature_dim}, batch_size: {batch_size}")
        logger.debug(f"  len(residue_atom_map): {len(residue_atom_map)}")
        if len(residue_atom_map) > 0:
            logger.debug(f"  residue_atom_map[0]: {residue_atom_map[0]}")
            logger.debug(f"  residue_atom_map[-1]: {residue_atom_map[-1]}")
        else:
            logger.debug("  residue_atom_map is EMPTY!")
    # PATCH: Check for correct expansion
    expected_atoms = n_residues * canonical_atom_count
    if debug_logging:
        logger.debug(f"  [PATCH CHECK] Expected total atoms: {expected_atoms}")
    # --- Existing logic ---
    # Ensure all dimensions are integers for torch.full
    batch_size_int = int(batch_size) if batch_size is not None else 1
    n_residues_int = int(n_residues)
    canonical_atom_count_int = int(canonical_atom_count)
    feature_dim_int = int(feature_dim)
    dense_atoms = torch.full((batch_size_int, n_residues_int, canonical_atom_count_int, feature_dim_int), fill_value, device=device)
    mask = torch.zeros((batch_size_int, n_residues_int, canonical_atom_count_int), dtype=torch.bool, device=device)
    for b in range(batch_size_int):
        for i, atom_indices in enumerate(residue_atom_map):
            # PATCH: Print atom_indices for each residue
            if debug_logging:
                logger.debug(f"    [DEBUG] residue {i}: atom_indices={atom_indices}")
            # stage_c_output[b, i, :, :] is [N_sparse_atoms_per_res, feat]
            for j, atom_idx in enumerate(atom_indices):
                if atom_idx >= canonical_atom_count:
                    if debug_logging:
                        logger.debug(f"      [ERROR] atom_idx {atom_idx} >= canonical_atom_count {canonical_atom_count} (residue {i})")
                    continue
                dense_atoms[b, i, atom_idx, :] = stage_c_output[b, i, j, :]
                mask[b, i, atom_idx] = True
    if debug_logging:
        logger.debug("[hybrid_bridging_sparse_to_dense] dense_atoms.shape: %s", dense_atoms.shape)
        logger.debug("[hybrid_bridging_sparse_to_dense] mask.shape: %s", mask.shape)
    return dense_atoms, mask

# Example usage:
# dense_atoms, mask = hybrid_bridging_sparse_to_dense(stage_c_output, residue_atom_map, n_residues, canonical_atom_count, feature_dim, batch_size)
