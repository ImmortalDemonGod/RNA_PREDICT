"""
RNA base placement functions for MP-NeRF implementation.
"""

import torch
import logging
from .rna_constants import BACKBONE_ATOMS
from .rna_atom_positioning import calculate_atom_position
from ..final_kb_rna import (
    get_base_geometry,
    get_connectivity,
)

logger = logging.getLogger("rna_predict.pipeline.stageC.mp_nerf.rna_base_placement")

#@snoop
def place_rna_bases(
    backbone_coords: torch.Tensor,
    seq: str,
    angles_mask: torch.Tensor,
    device: str = "cpu",
    debug_logging: bool = False,
) -> torch.Tensor:
    """
    Place base atoms for each residue in the RNA sequence.

    Args:
        backbone_coords: Tensor of shape [L, B, 3] containing backbone coordinates
        seq: RNA sequence
        angles_mask: Tensor of shape [2, L, B] containing angle masks
        device: Device to place tensors on
        debug_logging: Flag to control debug print statements

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
    # Instrument: print requires_grad and grad_fn for backbone_coords input
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[DEBUG-GRAD] backbone_coords (input to place_rna_bases) requires_grad: {getattr(backbone_coords, 'requires_grad', None)}, grad_fn: {getattr(backbone_coords, 'grad_fn', None)}")
    # Guaranteed output regardless of logger config
    if debug_logging:
        print(f"[DEBUG-GRAD-PRINT] backbone_coords (input to place_rna_bases) requires_grad: {getattr(backbone_coords, 'requires_grad', None)}, grad_fn: {getattr(backbone_coords, 'grad_fn', None)}")
    # --- NEW: Validate all residues are known ---
    from rna_predict.utils.tensor_utils.types import STANDARD_RNA_ATOMS
    unknown_residues = [res for res in seq if res not in STANDARD_RNA_ATOMS]
    if unknown_residues:
        raise ValueError(f"[ERR-RNAPREDICT-INVALID-RES-001] Unknown residue(s) in sequence: {unknown_residues}. Sequence: {seq}")
    # Check for NaN values
    if torch.isnan(backbone_coords).any():
        logger.error(f"[UNIQUE-ERR-RNA-NAN-BACKBONE] backbone_coords contains NaN values for seq: {seq}")

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
                logger.error(f"[UNIQUE-ERR-RNA-NAN-BACKBONE-COPY] NaN after backbone copy at residue {i}, atom {BACKBONE_ATOMS[idx]}, seq={seq}")

    # Place base atoms for each residue
    for i, base in enumerate(seq):
        atom_list = STANDARD_RNA_ATOMS[base]
        # Merge backbone and base connectivity for this residue
        backbone_connectivity = get_connectivity("backbone")
        base_connectivity = get_connectivity(base)
        merged_connectivity = list(backbone_connectivity) + list(base_connectivity)
        # Place all atoms in atom_list, including backbone and base atoms
        # Initialize placed_atoms with None values, but use torch.zeros for tensor placeholders
        placed_atoms = {}
        for name in atom_list:
            placed_atoms[name] = torch.zeros(3, device=device) if name in BACKBONE_ATOMS else None
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
            if atom_name in ["OP1", "OP2"] and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[DEBUG-NAN-TRACE] Attempting to place {atom_name} at residue {i}, placement index {idx}, seq={seq}")
                logger.debug(f"[DEBUG-NAN-TRACE] Already placed atoms: {[k for k, v in placed_atoms.items() if isinstance(v, torch.Tensor)]}")
                logger.debug(f"[DEBUG-NAN-TRACE] Connectivity pairs for {atom_name}: {bond_partners}")
                logger.debug(f"[DEBUG-NAN-TRACE] All connectivity for this residue: {merged_connectivity}")
            # Special logic for OP1/OP2 placement: require three unique, non-collinear reference atoms
            if atom_name in ["OP1", "OP2"]:
                # Preferred order: O5', O3', C5'
                possible_refs = ["O5'", "O3'", "C5'"]
                placed_ref_names = [partner for partner in possible_refs if partner in placed_atoms and isinstance(placed_atoms[partner], torch.Tensor)]
                ref_candidates = ["P"] + placed_ref_names
                seen = set()
                unique_refs = []
                for ref in ref_candidates:
                    if ref not in seen:
                        seen.add(ref)
                        unique_refs.append(ref)
                if len(unique_refs) < 3:
                    for extra in atom_list[:idx]:
                        if extra not in unique_refs and extra in placed_atoms and isinstance(placed_atoms[extra], torch.Tensor):
                            unique_refs.append(extra)
                        if len(unique_refs) == 3:
                            break
                # --- PATCH: Robust reference selection for OP1/OP2 ---
                # Gather candidate references and their coordinates
                ref_coords = []
                ref_names = []
                for ref in unique_refs:
                    if ref == 'artificial':
                        coord = torch.tensor([1.0, 0.0, 0.0], device=device)
                    else:
                        coord = placed_atoms.get(ref, None)
                    if coord is not None:
                        ref_coords.append(coord)
                        ref_names.append(ref)
                # Check for at least two distinct coordinates
                found = False
                for ref_i in range(len(ref_coords)):
                    for ref_j in range(ref_i+1, len(ref_coords)):
                        if not torch.allclose(ref_coords[ref_i], ref_coords[ref_j], atol=1e-6):
                            found = True
                            break
                    if found:
                        break
                if not found:
                    logger.error(f"[UNIQUE-ERR-RNA-NAN-REFS-INSUFFICIENT-COORDS] OP1/OP2 at residue {i}: Unable to find two distinct reference coordinates among {ref_names}. Skipping placement.")
                    # SYSTEMATIC DEBUGGING: Dump reference atom names and coordinates for diagnosis
                    for name, coord in zip(ref_names, ref_coords):
                        logger.error(f"  [DEBUG-REF-COORD] {name}: {coord}")
                    logger.error(f"  [DEBUG-PLACED-ATOMS] residue {i}: {placed_atoms}")
                    continue  # Skip placement
                # Select the first two distinct coordinates for ref1 and ref2
                ref1, ref2, ref3 = None, None, None
                for ref_i1 in range(len(ref_coords)):
                    for ref_i2 in range(ref_i1+1, len(ref_coords)):
                        if not torch.allclose(ref_coords[ref_i1], ref_coords[ref_i2], atol=1e-6):
                            ref1 = ref_coords[ref_i1]
                            ref2 = ref_coords[ref_i2]
                            ref1_name = ref_names[ref_i1]
                            ref2_name = ref_names[ref_i2]
                            break
                    if ref1 is not None:
                        break
                # Choose a third reference distinct from the first two, or create an artificial one
                ref3 = None
                for ref_k in range(len(ref_coords)):
                    if not torch.allclose(ref_coords[ref_k], ref1, atol=1e-6) and not torch.allclose(ref_coords[ref_k], ref2, atol=1e-6):
                        ref3 = ref_coords[ref_k]
                        break
                if ref3 is None:
                    # Create a third reference offset from ref1
                    ref3 = ref1 + torch.tensor([0.0, 1.0, 0.0], device=device)
                    ref_names.append('artificial_offset')
                # Log the chosen references
                logger.debug(f"[DEBUG-NAN-REFS] {atom_name} residue {i}: ref1={ref1}, ref2={ref2}, ref3={ref3}, ref_names={ref_names}")
                # Check for collinearity: compute the area of the triangle they form
                v1 = ref2 - ref1 if isinstance(ref1, torch.Tensor) and isinstance(ref2, torch.Tensor) else torch.zeros(3, device=device)
                v2 = ref3 - ref1 if isinstance(ref1, torch.Tensor) and isinstance(ref3, torch.Tensor) else torch.zeros(3, device=device)
                cross = torch.linalg.cross(v1, v2, dim=-1)
                collinear = torch.norm(cross) < 1e-3
                # If collinear, try to find a non-collinear third reference
                if collinear:
                    logger.debug(f"[DEBUG] Collinear references detected for {atom_name} at residue {i}: {unique_refs}")
                    # Try to find a non-collinear third reference among other placed atoms
                    for extra in atom_list[:idx]:
                        if extra not in unique_refs[:2] and extra in placed_atoms and isinstance(placed_atoms[extra], torch.Tensor):
                            candidate = placed_atoms[extra]
                            v2 = candidate - ref1 if isinstance(ref1, torch.Tensor) and isinstance(candidate, torch.Tensor) else torch.zeros(3, device=device)
                            cross = torch.linalg.cross(v1, v2, dim=-1)
                            if torch.norm(cross) >= 1e-3:
                                ref3 = candidate
                                unique_refs[2] = extra
                                collinear = False
                                logger.debug(f"[DEBUG] Found non-collinear reference: {extra}")
                                break

                    # If still collinear, create a perpendicular reference point
                    if collinear:
                        logger.debug(f"[DEBUG] Creating artificial non-collinear reference for {atom_name} at residue {i}")
                        # Create a perpendicular vector to v1
                        if torch.abs(v1[0]) > 1e-6 or torch.abs(v1[1]) > 1e-6:
                            perp = v1.new_tensor([-v1[1], v1[0], 0.0])
                        else:
                            perp = v1.new_tensor([1.0, 0.0, 0.0])
                        if torch.norm(perp) < 1e-6:
                            logger.error(f"[UNIQUE-ERR-RNA-NAN-FALLBACK] Perpendicular vector is near zero for {atom_name} at residue {i}, using default [1,0,0]")
                            perp = v1.new_tensor([1.0, 0.0, 0.0])
                        perp = perp / torch.norm(perp)
                        ref3 = ref1 + perp
                        unique_refs[2] = "artificial"
                        collinear = False
                # --- NAN DEBUG ---
                # Check for degenerate or ill-separated references
                default_tensor = torch.tensor(0.0, device=device)
                sep12 = torch.norm(ref2 - ref1) if isinstance(ref1, torch.Tensor) and isinstance(ref2, torch.Tensor) else default_tensor
                sep13 = torch.norm(ref3 - ref1) if isinstance(ref1, torch.Tensor) and isinstance(ref3, torch.Tensor) else default_tensor
                sep23 = torch.norm(ref3 - ref2) if isinstance(ref2, torch.Tensor) and isinstance(ref3, torch.Tensor) else default_tensor
                # Check for NaN values in reference tensors
                has_nan = False
                if isinstance(ref1, torch.Tensor) and torch.isnan(ref1).any():
                    has_nan = True
                if isinstance(ref2, torch.Tensor) and torch.isnan(ref2).any():
                    has_nan = True
                if isinstance(ref3, torch.Tensor) and torch.isnan(ref3).any():
                    has_nan = True

                if has_nan or sep12 < 1e-6 or sep13 < 1e-6 or sep23 < 1e-6:
                    logger.error(f"[UNIQUE-ERR-RNA-NAN-REFS-SEP] NaN or degenerate/ill-separated references for {atom_name} at residue {i} ({unique_refs}) in seq {seq}. sep12={sep12}, sep13={sep13}, sep23={sep23}. Skipping placement.")
                    logger.error(f"[DEBUG-NAN-REFS] {atom_name} residue {i}: ref1={ref1}, ref2={ref2}, ref3={ref3}, unique_refs={unique_refs}")
                    # If these are indices into a coordinate array, log those as well if available
                    if 'ref_indices' in locals():
                        logger.error(f"[DEBUG-NAN-REFS] Atom indices: {ref_indices}")
                    continue  # Skip placement if references are invalid
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[DEBUG] Reference values for {atom_name} at residue {i}: ref1={ref1}, ref2={ref2}, ref3={ref3}")
                    logger.debug(f"[DEBUG] Separations for {atom_name} at residue {i}: sep12={sep12}, sep13={sep13}, sep23={sep23}")
                # --- END NAN DEBUG ---
                # --- Ensure tensor types for bond_length, bond_angle, torsion_angle (OP1/OP2 special logic) ---
                bond_length = bond_lengths.get(f"P-{atom_name}", 1.5)
                bond_angle = bond_angles.get(f"P-{atom_name}", 120.0)
                torsion_angle = 0.0
                # Convert to tensors if needed
                if not torch.is_tensor(bond_length):
                    bond_length = torch.tensor(bond_length, dtype=ref2.dtype if isinstance(ref2, torch.Tensor) else torch.float32, device=device)
                if not torch.is_tensor(bond_angle):
                    bond_angle = torch.tensor(bond_angle, dtype=ref2.dtype if isinstance(ref2, torch.Tensor) else torch.float32, device=device)
                if not torch.is_tensor(torsion_angle):
                    torsion_angle = torch.tensor(torsion_angle, dtype=ref2.dtype if isinstance(ref2, torch.Tensor) else torch.float32, device=device)
                logger.debug(f"[DEBUG-PLACEMENT] Calling calculate_atom_position(OP1/OP2) with types: bond_length={type(bond_length)}, bond_angle={type(bond_angle)}, torsion_angle={type(torsion_angle)}")
                try:
                    pos = calculate_atom_position(ref3, ref2, bond_length, bond_angle, torsion_angle, device)
                    if isinstance(pos, torch.Tensor) and torch.isnan(pos).any():
                        logger.error(f"[UNIQUE-ERR-RNA-NAN-POS] Atom placement produced NaN for {atom_name} at residue {i} (refs={unique_refs}, sep12={sep12}, sep13={sep13}, sep23={sep23}) in seq {seq}. Skipping placement.")
                        continue  # Skip placement if output is NaN
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DEBUG] Output position for {atom_name} at residue {i}: pos={pos}")
                    full_coords[i, idx, :] = pos
                    placed_atoms[atom_name] = pos
                except Exception as e:
                    logger.error(f"[DEBUG] Residue {i} base {base}: Atom '{atom_name}' placement failed with error: {e}")
                continue  # Done with OP1/OP2, skip to next atom
            # Try to find a previously placed atom to use as reference (default logic)
            ref_atom = None
            ref_partner = None
            for pair in bond_partners:
                partner = pair[0] if pair[1] == atom_name else pair[1]
                if partner in placed_atoms and isinstance(placed_atoms[partner], torch.Tensor):
                    ref_atom = placed_atoms[partner]
                    ref_partner = partner
                    break
            if ref_atom is None:
                logger.error(f"[ERR-RNAPREDICT-NAN-SKIP-001] Residue {i} base {base}: Atom '{atom_name}' could not be placed (no reference partner found, merged connectivity: {merged_connectivity}).")
                continue  # Can't place without a reference
            # Get bond length
            bond_key = f"{ref_partner}-{atom_name}" if f"{ref_partner}-{atom_name}" in bond_lengths else f"{atom_name}-{ref_partner}"
            bond_length = bond_lengths.get(bond_key, 1.5)
            bond_angle_key = f"{ref_partner}-{atom_name}" if f"{ref_partner}-{atom_name}" in bond_angles else f"{atom_name}-{ref_partner}"
            bond_angle = bond_angles.get(bond_angle_key, 120.0)
            # --- Ensure tensor types for bond_length, bond_angle, torsion_angle (default logic) ---
            if not torch.is_tensor(bond_length):
                bond_length = torch.tensor(bond_length, dtype=ref_atom.dtype if isinstance(ref_atom, torch.Tensor) else torch.float32, device=device)
            if not torch.is_tensor(bond_angle):
                bond_angle = torch.tensor(bond_angle, dtype=ref_atom.dtype if isinstance(ref_atom, torch.Tensor) else torch.float32, device=device)
            torsion_angle = 0.0
            if not torch.is_tensor(torsion_angle):
                torsion_angle = torch.tensor(torsion_angle, dtype=ref_atom.dtype if isinstance(ref_atom, torch.Tensor) else torch.float32, device=device)
            logger.debug(f"[DEBUG-PLACEMENT] Calling calculate_atom_position(default) with types: bond_length={type(bond_length)}, bond_angle={type(bond_angle)}, torsion_angle={type(torsion_angle)}")
            # Place the atom
            prev_atoms = [name for name in atom_list[:idx] if name in placed_atoms and isinstance(placed_atoms[name], torch.Tensor)]
            if len(prev_atoms) < 2:
                logger.error(f"[ERR-RNAPREDICT-NAN-SKIP-002] Residue {i} base {base}: Atom '{atom_name}' cannot be placed, not enough previously placed atoms ({prev_atoms}).")
                continue
            prev1 = placed_atoms[prev_atoms[-1]]
            prev2 = placed_atoms[prev_atoms[-2]]
            try:
                pos = calculate_atom_position(prev2, prev1, bond_length, bond_angle, torsion_angle, device)
            except Exception as e:
                logger.error(f"[DEBUG] Residue {i} base {base}: Atom '{atom_name}' placement failed with error: {e}")
                continue
            # Universal NaN check after any atom placement
            if isinstance(full_coords, torch.Tensor) and torch.isnan(full_coords[i, idx, :]).any():
                logger.error(f"[UNIQUE-ERR-RNA-NAN-GENERAL] NaN after placing atom {atom_name} at residue {i} (base {base}), seq={seq}")
            full_coords[i, idx, :] = pos
            placed_atoms[atom_name] = pos
            # --- DEBUG: Check requires_grad and grad_fn after base placement ---
            if isinstance(full_coords, torch.Tensor):
                logger.debug(f"[GRAD-TRACE-BASE-PLACEMENT] full_coords.requires_grad: {full_coords.requires_grad}, grad_fn: {full_coords.grad_fn}")
                if debug_logging:
                    print(f"[DEBUG-GRAD-PRINT-BASE-PLACEMENT] full_coords.requires_grad: {full_coords.requires_grad}, grad_fn: {full_coords.grad_fn}")
    # After all placements, fill any remaining NaNs with zero and log unique error
    if isinstance(full_coords, torch.Tensor) and torch.isnan(full_coords).any():
        logger.error("[UNIQUE-ERR-RNA-NAN-ZERO-FILL] NaNs detected in output after placement; filling with zeros.")
        full_coords = torch.nan_to_num(full_coords, nan=0.0)
    # Instrument: print requires_grad and grad_fn for full_coords output
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[DEBUG-GRAD] full_coords (place_rna_bases output) requires_grad: {getattr(full_coords, 'requires_grad', None)}, grad_fn: {getattr(full_coords, 'grad_fn', None)}")
        if debug_logging:
            print(f"[DEBUG-GRAD-PRINT] full_coords (place_rna_bases output) requires_grad: {getattr(full_coords, 'requires_grad', None)}, grad_fn: {getattr(full_coords, 'grad_fn', None)}")
    return full_coords
