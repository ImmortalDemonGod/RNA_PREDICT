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
    Bridges the architectural mismatch between Stage C (sparse atom output) and Stage D (dense atom input).

    This function transforms the sparse atom representation produced by Stage C (shape: [batch_size, n_residues, n_sparse_atoms, feature_dim])
    into the dense atom representation expected by Stage D (shape: [batch_size, n_residues, canonical_atom_count, feature_dim]).
    It uses a residue-to-atom mapping to place each sparse atom in its canonical position, and generates a mask indicating valid atoms.

    Args:
        stage_c_output (torch.Tensor): Sparse atom features from Stage C, shape [batch_size, n_residues, n_sparse_atoms, feature_dim].
        residue_atom_map (np.ndarray or torch.Tensor): Mapping from (residue, sparse_atom_idx) to canonical atom index.
        n_residues (int): Number of residues in the batch.
        canonical_atom_count (int): Number of canonical atom positions per residue (dense count, e.g. 44).
        feature_dim (int): Feature dimension for each atom.
        batch_size (int): Batch size.
        fill_value (float, optional): Value to fill for missing atoms. Defaults to 0.0.
        device (torch.device, optional): Device to use. Defaults to None.
        debug_logging (bool, optional): If True, enables debug logging. Defaults to False.

    Returns:
        dense_atoms (torch.Tensor): Dense atom tensor, shape [batch_size, n_residues, canonical_atom_count, feature_dim].
        mask (torch.Tensor): Boolean mask indicating which atoms are valid, shape [batch_size, n_residues, canonical_atom_count].

    Rationale:
        Stage C outputs a sparse set of atoms for efficiency and model design reasons, but Stage D's diffusion model
        expects a dense atom layout. This bridging function maps the sparse output into the dense format, enabling
        seamless integration between the two stages without loss of information or excessive memory use.
    """
    # Systematic debug: log input shapes and mapping
    debug_print_hybrid_bridging_inputs(
        stage_c_output,
        residue_atom_map,
        n_residues,
        canonical_atom_count,
        feature_dim,
        batch_size,
        fill_value,
        device,
        debug_logging,
    )
    logger.debug("[ENTRY] hybrid_bridging_sparse_to_dense called. debug_logging: %s", debug_logging)
    if debug_logging:
        logger.debug("[hybrid_bridging_sparse_to_dense] ENTRY")
        # Direct prints for systematic debugging
        print(f"[DEBUG][BRIDGE] stage_c_output.dim={stage_c_output.dim()}, shape={tuple(stage_c_output.shape)}")
        if stage_c_output.dim() == 2:
            logger.debug("[BRIDGE] Detected 2D sparse output, adding batch dim")
            stage_c_output = stage_c_output.unsqueeze(0)
            print(f"[DEBUG][BRIDGE] New stage_c_output.dim={stage_c_output.dim()}, shape={tuple(stage_c_output.shape)}")
        print(f"[DEBUG][BRIDGE] first residues maps lengths: {[len(lst) for lst in residue_atom_map[:5]]}")
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
    # Handle flat sparse output [batch, total_sparse_atoms, feature_dim] via offset mapping
    if stage_c_output.dim() == 3:
        B, total_sparse, feat = stage_c_output.shape
        # Total mapped atoms from mapping
        total_map = sum(len(lst) for lst in residue_atom_map)
        if total_sparse == total_map:
            # Initialize dense outputs
            dense_atoms = torch.full((B, n_residues, canonical_atom_count, feature_dim), fill_value, device=device)
            mask = torch.zeros((B, n_residues, canonical_atom_count), dtype=torch.bool, device=device)
            for b in range(B):
                offset = 0
                for i, atom_indices in enumerate(residue_atom_map):
                    # PATCH: Print atom_indices for each residue
                    if debug_logging:
                        logger.debug(f"    [DEBUG] residue {i}: atom_indices={atom_indices}")
                    for j, atom_idx in enumerate(atom_indices):
                        if atom_idx < canonical_atom_count:
                            dense_atoms[b, i, atom_idx, :] = stage_c_output[b, offset + j, :]
                            mask[b, i, atom_idx] = True
                    offset += len(atom_indices)
            if debug_logging:
                logger.debug(f"  [PATCH] Mapped flat sparse to dense: dense_atoms.shape={dense_atoms.shape}, mask.shape={mask.shape}")
            return dense_atoms, mask
        else:
            raise ValueError(f"Flat stage_c_output size {total_sparse} mismatches mapping sum {total_map}")
    # Handle per-residue sparse output [batch, n_residues, sparse_atoms_per_res, feature_dim]
    elif stage_c_output.dim() == 4:
        # Ensure all dimensions are integers for torch.full
        batch_size_int, n_residues_int, sparse_atoms_per_res, feature_dim_int = stage_c_output.shape
        canonical_atom_count_int = int(canonical_atom_count)
        dense_atoms = torch.full((batch_size_int, n_residues_int, canonical_atom_count_int, feature_dim_int), fill_value, device=device)
        mask = torch.zeros((batch_size_int, n_residues_int, canonical_atom_count_int), dtype=torch.bool, device=device)
        for b in range(batch_size_int):
            for i, atom_indices in enumerate(residue_atom_map):
                if debug_logging:
                    logger.debug(f"    [DEBUG] residue {i}: atom_indices={atom_indices}")
                for j, atom_idx in enumerate(atom_indices):
                    if j >= sparse_atoms_per_res:
                        if debug_logging:
                            logger.debug(
                                "      [WARN] residue %d has %d mapped atoms but Stage-C output supplies only %d; skipping extras",
                                i, len(atom_indices), sparse_atoms_per_res
                            )
                        break
                    if atom_idx >= canonical_atom_count_int:
                        if debug_logging:
                            logger.debug(f"      [ERROR] atom_idx {atom_idx} >= canonical_atom_count {canonical_atom_count_int} (residue {i})")
                        continue
                    dense_atoms[b, i, atom_idx, :] = stage_c_output[b, i, j, :]
                    mask[b, i, atom_idx] = True
        if debug_logging:
            logger.debug("[hybrid_bridging_sparse_to_dense] dense_atoms.shape: %s", dense_atoms.shape)
            logger.debug("[hybrid_bridging_sparse_to_dense] mask.shape: %s", mask.shape)
        return dense_atoms, mask
    else:
        raise ValueError(f"Unsupported stage_c_output dimension: {stage_c_output.dim()}")


# Example usage:
# dense_atoms, mask = hybrid_bridging_sparse_to_dense(stage_c_output, residue_atom_map, n_residues, canonical_atom_count, feature_dim, batch_size)
