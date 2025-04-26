import torch
from typing import List, Optional

# --- DEBUG PATCH START ---
def debug_print_hybrid_bridging_inputs(stage_c_output, residue_atom_map, n_residues, canonical_atom_count, feature_dim, batch_size, fill_value, device):
    print("[DEBUG][hybrid_bridging_sparse_to_dense] stage_c_output.shape:", getattr(stage_c_output, 'shape', None))
    print("[DEBUG][hybrid_bridging_sparse_to_dense] n_residues:", n_residues)
    print("[DEBUG][hybrid_bridging_sparse_to_dense] canonical_atom_count:", canonical_atom_count)
    print("[DEBUG][hybrid_bridging_sparse_to_dense] feature_dim:", feature_dim)
    print("[DEBUG][hybrid_bridging_sparse_to_dense] batch_size:", batch_size)
    print("[DEBUG][hybrid_bridging_sparse_to_dense] fill_value:", fill_value)
    print("[DEBUG][hybrid_bridging_sparse_to_dense] device:", device)
    print("[DEBUG][hybrid_bridging_sparse_to_dense] len(residue_atom_map):", len(residue_atom_map))
    if len(residue_atom_map) > 0:
        print("[DEBUG][hybrid_bridging_sparse_to_dense] residue_atom_map[0:5]:", residue_atom_map[:5])
    else:
        print("[DEBUG][hybrid_bridging_sparse_to_dense] residue_atom_map is empty!")
# --- DEBUG PATCH END ---

def hybrid_bridging_sparse_to_dense(
    stage_c_output: torch.Tensor,
    residue_atom_map: List[List[int]],
    n_residues: int,
    canonical_atom_count: int,
    feature_dim: int,
    batch_size: Optional[int] = 1,
    fill_value: float = 0.0,
    device: Optional[torch.device] = None,
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
    Returns:
        dense_atoms: [batch_size, n_residues, canonical_atom_count, feature_dim]
        mask: [batch_size, n_residues, canonical_atom_count]
    """
    # SYSTEMATIC DEBUGGING: Print all relevant shapes and config values
    print("[DEBUG][hybrid_bridging_sparse_to_dense] ENTRY")
    print(f"  stage_c_output.shape: {getattr(stage_c_output, 'shape', None)}")
    print(f"  n_residues: {n_residues}, canonical_atom_count: {canonical_atom_count}, feature_dim: {feature_dim}, batch_size: {batch_size}")
    print(f"  len(residue_atom_map): {len(residue_atom_map)}")
    if len(residue_atom_map) > 0:
        print(f"  residue_atom_map[0]: {residue_atom_map[0]}")
        print(f"  residue_atom_map[-1]: {residue_atom_map[-1]}")
    else:
        print("  residue_atom_map is EMPTY!")
    # PATCH: Check for correct expansion
    expected_atoms = n_residues * canonical_atom_count
    print(f"  [PATCH CHECK] Expected total atoms: {expected_atoms}")
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
            print(f"    [DEBUG] residue {i}: atom_indices={atom_indices}")
            # stage_c_output[b, i, :, :] is [N_sparse_atoms_per_res, feat]
            for j, atom_idx in enumerate(atom_indices):
                if atom_idx >= canonical_atom_count:
                    print(f"      [ERROR] atom_idx {atom_idx} >= canonical_atom_count {canonical_atom_count} (residue {i})")
                    continue
                dense_atoms[b, i, atom_idx, :] = stage_c_output[b, i, j, :]
                mask[b, i, atom_idx] = True
    print("[DEBUG][hybrid_bridging_sparse_to_dense] dense_atoms.shape:", dense_atoms.shape)
    print("[DEBUG][hybrid_bridging_sparse_to_dense] mask.shape:", mask.shape)
    return dense_atoms, mask

# Example usage:
# dense_atoms, mask = hybrid_bridging_sparse_to_dense(stage_c_output, residue_atom_map, n_residues, canonical_atom_count, feature_dim, batch_size)
