"""
RNA base placement functions for MP-NeRF implementation.
"""

import torch

from .rna_constants import BACKBONE_ATOMS
from .rna_atom_positioning import calculate_atom_position
from ..final_kb_rna import (
    get_base_geometry,
    get_connectivity,
)

def place_rna_bases(
    backbone_coords: torch.Tensor,
    seq: str,
    angles_mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Place base atoms for each residue in the RNA sequence.

    Args:
        backbone_coords: Tensor of shape [L, B, 3] containing backbone coordinates
        seq: RNA sequence
        angles_mask: Tensor of shape [2, L, B] containing angle masks
        device: Device to place tensors on

    Returns:
        Tensor of shape [L, max_atoms, 3] containing all atom coordinates
    """
    # Validate input
    if not isinstance(backbone_coords, torch.Tensor):
        raise ValueError("backbone_coords must be a torch.Tensor")
    if not isinstance(seq, str):
        raise ValueError("seq must be a string")
    if not isinstance(angles_mask, torch.Tensor):
        raise ValueError("angles_mask must be a torch.Tensor")
    # --- NEW: Validate all residues are known ---
    from rna_predict.utils.tensor_utils.types import STANDARD_RNA_ATOMS
    unknown_residues = [res for res in seq if res not in STANDARD_RNA_ATOMS]
    if unknown_residues:
        raise ValueError(f"[ERR-RNAPREDICT-INVALID-RES-001] Unknown residue(s) in sequence: {unknown_residues}. Sequence: {seq}")
    # Check for NaN values
    if torch.isnan(backbone_coords).any():
        print(f"[UNIQUE-ERR-RNA-NAN-BACKBONE] backbone_coords contains NaN values for seq: {seq}")

    # Get sequence length and max atoms per base
    L = len(seq)
    atom_counts = [len(STANDARD_RNA_ATOMS[res]) for res in seq]
    max_atoms = max(atom_counts)

    # Initialize output tensor to zeros for safety (prevents uninitialized NaNs)
    full_coords = torch.zeros((L, max_atoms, 3), dtype=torch.float32, device=device)

    # Copy backbone coordinates (assumes backbone atoms are always present and ordered)
    for i in range(L):
        n_backbone = len(BACKBONE_ATOMS)
        full_coords[i, :n_backbone, :] = backbone_coords[i, :n_backbone, :]
        # Universal NaN check after backbone copy
        for idx in range(n_backbone):
            if torch.isnan(full_coords[i, idx, :]).any():
                print(f"[UNIQUE-ERR-RNA-NAN-BACKBONE-COPY] NaN after backbone copy at residue {i}, atom {BACKBONE_ATOMS[idx]}, seq={seq}")

    # Place base atoms for each residue
    for i, base in enumerate(seq):
        atom_list = STANDARD_RNA_ATOMS[base]
        # Merge backbone and base connectivity for this residue
        backbone_connectivity = get_connectivity("backbone")
        base_connectivity = get_connectivity(base)
        merged_connectivity = list(backbone_connectivity) + list(base_connectivity)
        # Place all atoms in atom_list, including backbone and base atoms
        placed_atoms = {name: None for name in atom_list}
        # First, copy backbone atoms
        for idx, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name in atom_list:
                full_coords[i, atom_list.index(atom_name), :] = backbone_coords[i, idx, :]
                placed_atoms[atom_name] = backbone_coords[i, idx, :]
        # Now, place base atoms sequentially, using geometry if possible
        base_geom = get_base_geometry(base)
        bond_lengths = base_geom.get("bond_lengths", {})
        bond_angles = base_geom.get("bond_angles_deg", {})
        for idx, atom_name in enumerate(atom_list):
            if atom_name in BACKBONE_ATOMS:
                continue  # already placed
            # Find bonded atoms for this atom
            # Use connectivity info if available
            bond_partners = [pair for pair in merged_connectivity if atom_name in pair]
            # --- DEBUG: Print connectivity for OP1/OP2 ---
            if atom_name in ["OP1", "OP2"]:
                print(f"[DEBUG-NAN-TRACE] Attempting to place {atom_name} at residue {i}, placement index {idx}, seq={seq}")
                print(f"[DEBUG-NAN-TRACE] Already placed atoms: {[k for k, v in placed_atoms.items() if v is not None]}")
                print(f"[DEBUG-NAN-TRACE] Connectivity pairs for {atom_name}: {bond_partners}")
                print(f"[DEBUG-NAN-TRACE] All connectivity for this residue: {merged_connectivity}")
            # Special logic for OP1/OP2 placement: require three unique, non-collinear reference atoms
            if atom_name in ["OP1", "OP2"]:
                # Preferred order: O5', O3', C5'
                possible_refs = ["O5'", "O3'", "C5'"]
                placed_ref_names = [partner for partner in possible_refs if placed_atoms.get(partner) is not None]
                ref_candidates = ["P"] + placed_ref_names
                seen = set()
                unique_refs = []
                for ref in ref_candidates:
                    if ref not in seen:
                        seen.add(ref)
                        unique_refs.append(ref)
                if len(unique_refs) < 3:
                    for extra in atom_list[:idx]:
                        if extra not in unique_refs and placed_atoms.get(extra) is not None:
                            unique_refs.append(extra)
                        if len(unique_refs) == 3:
                            break
                if len(unique_refs) >= 3:
                    ref1 = placed_atoms[unique_refs[0]]
                    ref2 = placed_atoms[unique_refs[1]]
                    ref3 = placed_atoms[unique_refs[2]]
                    # Check for collinearity: compute the area of the triangle they form
                    v1 = ref2 - ref1
                    v2 = ref3 - ref1
                    cross = torch.linalg.cross(v1, v2, dim=-1)
                    collinear = torch.norm(cross) < 1e-3

                    # If collinear, try to find a non-collinear third reference
                    if collinear:
                        print(f"[DEBUG] Collinear references detected for {atom_name} at residue {i}: {unique_refs}")
                        # Try to find a non-collinear third reference among other placed atoms
                        for extra in atom_list[:idx]:
                            if extra not in unique_refs[:2] and placed_atoms.get(extra) is not None:
                                candidate = placed_atoms[extra]
                                v2 = candidate - ref1
                                cross = torch.linalg.cross(v1, v2, dim=-1)
                                if torch.norm(cross) >= 1e-3:
                                    ref3 = candidate
                                    unique_refs[2] = extra
                                    collinear = False
                                    print(f"[DEBUG] Found non-collinear reference: {extra}")
                                    break

                        # If still collinear, create a perpendicular reference point
                        if collinear:
                            print(f"[DEBUG] Creating artificial non-collinear reference for {atom_name} at residue {i}")
                            # Create a perpendicular vector to v1
                            if torch.abs(v1[0]) > 1e-6 or torch.abs(v1[1]) > 1e-6:
                                perp = torch.tensor([-v1[1], v1[0], 0.0], device=device)
                            else:
                                perp = torch.tensor([1.0, 0.0, 0.0], device=device)
                            if torch.norm(perp) < 1e-6:
                                print(f"[UNIQUE-ERR-RNA-NAN-FALLBACK] Perpendicular vector is near zero for {atom_name} at residue {i}, using default [1,0,0]")
                                perp = torch.tensor([1.0, 0.0, 0.0], device=device)
                            perp = perp / torch.norm(perp)
                            ref3 = ref1 + perp
                            unique_refs[2] = "artificial"
                            collinear = False
                    # --- NAN DEBUG ---
                    # Check for degenerate or ill-separated references
                    sep12 = torch.norm(ref2 - ref1)
                    sep13 = torch.norm(ref3 - ref1)
                    sep23 = torch.norm(ref3 - ref2)
                    if torch.isnan(ref1).any() or torch.isnan(ref2).any() or torch.isnan(ref3).any() \
                        or sep12 < 1e-6 or sep13 < 1e-6 or sep23 < 1e-6:
                        print(f"[UNIQUE-ERR-RNA-NAN-REFS-SEP] NaN or degenerate/ill-separated references for {atom_name} at residue {i} ({unique_refs}) in seq {seq}. sep12={sep12}, sep13={sep13}, sep23={sep23}. Skipping placement.")
                        continue  # Skip placement if references are invalid
                    print(f"[DEBUG] Reference values for {atom_name} at residue {i}: ref1={ref1}, ref2={ref2}, ref3={ref3}")
                    print(f"[DEBUG] Separations for {atom_name} at residue {i}: sep12={sep12}, sep13={sep13}, sep23={sep23}")
                    # --- END NAN DEBUG ---
                    try:
                        pos = calculate_atom_position(ref3, ref2, bond_lengths.get(f"P-{atom_name}", 1.5), bond_angles.get(f"P-{atom_name}", 120.0), 0.0, device)
                        if torch.isnan(pos).any():
                            print(f"[UNIQUE-ERR-RNA-NAN-POS] Atom placement produced NaN for {atom_name} at residue {i} (refs={unique_refs}, sep12={sep12}, sep13={sep13}, sep23={sep23}) in seq {seq}. Skipping placement.")
                            continue  # Skip placement if output is NaN
                        print(f"[DEBUG] Output position for {atom_name} at residue {i}: pos={pos}")
                        full_coords[i, idx, :] = pos
                        placed_atoms[atom_name] = pos
                    except Exception as e:
                        print(f"[DEBUG] Residue {i} base {base}: Atom '{atom_name}' placement failed with error: {e}")
                    continue  # Done with OP1/OP2, skip to next atom
            # Try to find a previously placed atom to use as reference (default logic)
            ref_atom = None
            ref_partner = None
            for pair in bond_partners:
                partner = pair[0] if pair[1] == atom_name else pair[1]
                if placed_atoms.get(partner) is not None:
                    ref_atom = placed_atoms[partner]
                    ref_partner = partner
                    break
            if ref_atom is None:
                print(f"[ERR-RNAPREDICT-NAN-SKIP-001] Residue {i} base {base}: Atom '{atom_name}' could not be placed (no reference partner found, merged connectivity: {merged_connectivity}).")
                continue  # Can't place without a reference
            # Get bond length
            bond_key = f"{ref_partner}-{atom_name}" if f"{ref_partner}-{atom_name}" in bond_lengths else f"{atom_name}-{ref_partner}"
            bond_length = bond_lengths.get(bond_key, 1.5)
            bond_angle_key = f"{ref_partner}-{atom_name}" if f"{ref_partner}-{atom_name}" in bond_angles else f"{atom_name}-{ref_partner}"
            bond_angle = bond_angles.get(bond_angle_key, 120.0)
            # Place the atom
            prev_atoms = [name for name in atom_list[:idx] if placed_atoms[name] is not None]
            if len(prev_atoms) < 2:
                print(f"[ERR-RNAPREDICT-NAN-SKIP-002] Residue {i} base {base}: Atom '{atom_name}' cannot be placed, not enough previously placed atoms ({prev_atoms}).")
                continue
            prev1 = placed_atoms[prev_atoms[-1]]
            prev2 = placed_atoms[prev_atoms[-2]]
            try:
                pos = calculate_atom_position(prev2, prev1, bond_length, bond_angle, 0.0, device)
                if torch.isnan(pos).any():
                    print(f"[DEBUG] Residue {i} base {base}: Atom '{atom_name}' placement produced NaN. prev2={prev_atoms[-2]}, prev1={prev_atoms[-1]}, ref={ref_partner}, bond_length={bond_length}, bond_angle={bond_angle}")
                full_coords[i, idx, :] = pos
                placed_atoms[atom_name] = pos
            except Exception as e:
                print(f"[DEBUG] Residue {i} base {base}: Atom '{atom_name}' placement failed with error: {e}")
                continue
            # Universal NaN check after any atom placement
            if torch.isnan(full_coords[i, idx, :]).any():
                print(f"[UNIQUE-ERR-RNA-NAN-GENERAL] NaN after placing atom {atom_name} at residue {i} (base {base}), seq={seq}")
    # After all placements, fill any remaining NaNs with zero and log unique error
    if torch.isnan(full_coords).any():
        print("[UNIQUE-ERR-RNA-NAN-ZERO-FILL] NaNs detected in output after placement; filling with zeros.")
        full_coords = torch.nan_to_num(full_coords, nan=0.0)
    return full_coords
