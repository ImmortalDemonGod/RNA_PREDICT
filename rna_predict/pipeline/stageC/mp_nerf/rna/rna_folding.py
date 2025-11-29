import logging
import math
from typing import Any, Dict 
import torch

from .rna_constants import BACKBONE_ATOMS 
from .rna_atom_positioning import calculate_atom_position 
from ..final_kb_rna import (
    get_bond_angle,
    get_bond_length,
    get_torsion_angle_name, 
    get_torsion_angle_index 
)

logger = logging.getLogger(__name__)

print(f"!!!!!!!!!! CASCADE MODULE-LEVEL: rna_folding.py LOADED FROM: {__file__} !!!!!!!!!!")

def build_rna_chain_from_internal_coords(
    scaffolds: Dict[str, torch.Tensor],
    num_residues: int,
    device: torch.device,
    atom_idx_map: Dict[str, int] 
) -> torch.Tensor:
    """
    Builds RNA chain coordinates from internal coordinates (torsion angles) and
    standard geometry (bond lengths, bond angles).

    Args:
        scaffolds: Dictionary containing at least "torsions" (N, num_torsions_per_residue).
                   Torsions are expected in radians.
        num_residues: Number of residues in the chain.
        device: PyTorch device to perform calculations on.
        atom_idx_map: Mapping from atom name to index in the output tensor.

    Returns:
        torch.Tensor of shape (num_residues, num_atoms_per_residue, 3)
        representing the Cartesian coordinates of the RNA chain.
    """
    num_atoms_per_residue = len(BACKBONE_ATOMS)
    residue_coords = torch.zeros((num_residues, num_atoms_per_residue, 3), dtype=scaffolds["torsions"].dtype, device=device)

    for i in range(num_residues):  
        p_atom_idx = atom_idx_map.get("P")
        if p_atom_idx is None:
            logger.error("[ERR-RNAPREDICT-ATOMMAP-001] 'P' atom not in atom_idx_map.")
            raise ValueError("'P' atom not in atom_idx_map.")
        residue_coords[i, p_atom_idx, :] = torch.tensor([0.0, 0.0, 0.0], dtype=scaffolds["torsions"].dtype, device=device)

        for j in range(1, len(BACKBONE_ATOMS)):
            # Define atom names for current iteration, used in calculations and potentially logging
            current_atom_name_calc = BACKBONE_ATOMS[j]
            prev_atom_name_calc = BACKBONE_ATOMS[j-1]

            current_atom_idx = atom_idx_map.get(current_atom_name_calc)
            prev_atom_idx = atom_idx_map.get(prev_atom_name_calc)

            if current_atom_idx is None or prev_atom_idx is None:
                logger.error(f"[ERR-RNAPREDICT-ATOMMAP-002] Atom index not found for {current_atom_name_calc} or {prev_atom_name_calc}")
                raise ValueError(f"Atom index not found for {current_atom_name_calc} or {prev_atom_name_calc}")

            prev_atom_coords = residue_coords[i, prev_atom_idx, :]

            bond_length_val = get_bond_length(f"{prev_atom_name_calc}-{current_atom_name_calc}")
            if bond_length_val is None:
                logger.warning(f"[WARN-RNAPREDICT-GEOM-001] Bond length for ({prev_atom_name_calc}-{current_atom_name_calc}) not found. Using default 1.5Å.")
                bond_length_val = 1.5
            # This is the current_bond_length tensor used in calculations and logging
            current_bond_length_tensor = torch.tensor(bond_length_val, dtype=scaffolds["torsions"].dtype, device=device)

            if j == 1: # Placing O5' (atom 1), relative to P (atom 0)
                offset = torch.tensor([current_bond_length_tensor.item(), 0.0, 0.0], dtype=prev_atom_coords.dtype, device=device)
                current_atom_coords = prev_atom_coords + offset
                # Logging for O5'
                log_atom_name_o5 = BACKBONE_ATOMS[j] # Should be O5'
                if i == 0 and log_atom_name_o5 == "O5'":
                    logger.info(f"[DEBUG-FOLDING] Placing O5' (res {i}, atom {j}): {log_atom_name_o5}")
                    logger.info(f"  REF P (P): {prev_atom_coords.cpu().detach().numpy()}")
                    logger.info(f"  Bond Length (P-O5'): {current_bond_length_tensor.item():.4f} Å")
            else: # Placing C5' (j=2) and subsequent atoms
                prev_prev_atom_name_calc = BACKBONE_ATOMS[j-2]
                prev_prev_atom_idx = atom_idx_map.get(prev_prev_atom_name_calc)
                if prev_prev_atom_idx is None:
                    logger.error(f"[ERR-RNAPREDICT-ATOMMAP-003] Atom index not found for {prev_prev_atom_name_calc}")
                    raise ValueError(f"Atom index not found for {prev_prev_atom_name_calc}")
                prev_prev_atom_coords = residue_coords[i, prev_prev_atom_idx, :]

                bond_angle_val = get_bond_angle(f"{prev_prev_atom_name_calc}-{prev_atom_name_calc}-{current_atom_name_calc}")
                if bond_angle_val is None:
                    logger.warning(f"[WARN-RNAPREDICT-GEOM-002] Bond angle for ({prev_prev_atom_name_calc}-{prev_atom_name_calc}-{current_atom_name_calc}) not found. Using default 109.5°.")
                    bond_angle_val = 109.5
                current_bond_angle_rad = torch.tensor(bond_angle_val * (math.pi / 180.0), dtype=scaffolds["torsions"].dtype, device=device)

                a_ref_coords_for_nerf = None
                current_torsion_angle_rad_for_nerf = None
                torsion_name_for_log_str = "N/A"

                if j == 2: # Placing C5' (atom 2). Needs P(0), O5'(1).
                    # Define atom names specifically for this logging context if needed
                    log_prev_prev_atom_c5 = BACKBONE_ATOMS[j-2] # P
                    log_prev_atom_c5 = BACKBONE_ATOMS[j-1]      # O5'
                    log_current_atom_c5 = BACKBONE_ATOMS[j]     # C5'

                    a_ref_coords_for_nerf = prev_prev_atom_coords - torch.tensor([0.0, 1.0, 0.0], dtype=prev_prev_atom_coords.dtype, device=device)
                    current_torsion_angle_rad_for_nerf = torch.tensor(0.0, dtype=scaffolds["torsions"].dtype, device=device)
                    torsion_name_for_log_str = f"FixedDummyA-{log_prev_prev_atom_c5}-{log_prev_atom_c5}-{log_current_atom_c5}"
                    if i == 0 and log_current_atom_c5 == "C5'":
                        logger.info(f"[DEBUG-FOLDING] Placing C5' (res {i}, atom {j}): {log_current_atom_c5} using 3-atom NeRF")
                        logger.info(f"  NeRF a (dummy): {a_ref_coords_for_nerf.cpu().detach().numpy()}")
                        logger.info(f"  NeRF b (P):    {prev_prev_atom_coords.cpu().detach().numpy()}")
                        logger.info(f"  NeRF c (O5'):  {prev_atom_coords.cpu().detach().numpy()}")
                        logger.info(f"  Bond Length (O5'-C5'): {current_bond_length_tensor.item():.4f} Å")
                        logger.info(f"  Bond Angle (P-O5'-C5'): {bond_angle_val:.2f}° ({current_bond_angle_rad.item():.4f} rad)")
                        logger.info(f"  Torsion Angle ({torsion_name_for_log_str}): {current_torsion_angle_rad_for_nerf.item():.4f} rad")
                elif j >= 3: # Placing C4' (j=3) and onwards.
                    nerf_a_atom_name_calc = BACKBONE_ATOMS[j-3]
                    prev_prev_prev_atom_idx = atom_idx_map.get(nerf_a_atom_name_calc)
                    if prev_prev_prev_atom_idx is None:
                        logger.error(f"[ERR-RNAPREDICT-ATOMMAP-004] Atom index not found for {nerf_a_atom_name_calc}")
                        raise ValueError(f"Atom index not found for {nerf_a_atom_name_calc}")
                    a_ref_coords_for_nerf = residue_coords[i, prev_prev_prev_atom_idx, :]

                    # Atom names for torsion definition and logging
                    torsion_log_atom1 = BACKBONE_ATOMS[j-3]
                    torsion_log_atom2 = BACKBONE_ATOMS[j-2]
                    torsion_log_atom3 = BACKBONE_ATOMS[j-1]
                    torsion_log_atom4 = BACKBONE_ATOMS[j]

                    torsion_name = get_torsion_angle_name(torsion_log_atom1, torsion_log_atom2, torsion_log_atom3, torsion_log_atom4)
                    expected_torsion_idx = get_torsion_angle_index(torsion_name) if torsion_name else -1

                    if expected_torsion_idx != -1 and \
                       scaffolds["torsions"].nelement() > 0 and \
                       scaffolds["torsions"].shape[0] > i and \
                       scaffolds["torsions"].shape[1] > expected_torsion_idx:
                        current_torsion_angle_rad_for_nerf = scaffolds["torsions"][i, expected_torsion_idx]
                    else:
                        logger.warning(f"[WARN-RNAPREDICT-GEOM-003] Torsion angle '{torsion_name}' (idx {expected_torsion_idx}) for atoms "
                                       f"{torsion_log_atom1}-{torsion_log_atom2}-{torsion_log_atom3}-{torsion_log_atom4} "
                                       f"not found or invalid in scaffolds. Using 0.0 rad.")
                        current_torsion_angle_rad_for_nerf = torch.tensor(0.0, dtype=scaffolds["torsions"].dtype, device=device)
                    torsion_name_for_log_str = torsion_name if torsion_name else "UnknownTorsion"

                    # Logging for C4'
                    log_current_atom_c4_onwards = BACKBONE_ATOMS[j]
                    if i == 0 and log_current_atom_c4_onwards == "C4'":
                        logger.info(f"[DEBUG-FOLDING] Placing C4' (res {i}, atom {j}): {log_current_atom_c4_onwards} using 4-atom NeRF")
                        logger.info(f"  NeRF a (P):    {a_ref_coords_for_nerf.cpu().detach().numpy()}")
                        logger.info(f"  NeRF b (O5'):  {prev_prev_atom_coords.cpu().detach().numpy()}")
                        logger.info(f"  NeRF c (C5'):  {prev_atom_coords.cpu().detach().numpy()}")
                        logger.info(f"  Bond Length (C5'-C4'): {current_bond_length_tensor.item():.4f} Å")
                        logger.info(f"  Bond Angle (O5'-C5'-C4'): {bond_angle_val:.2f}° ({current_bond_angle_rad.item():.4f} rad)")
                        logger.info(f"  Torsion Angle ({torsion_name_for_log_str}, idx {expected_torsion_idx}): {current_torsion_angle_rad_for_nerf.item():.4f} rad from scaffolds")
                else:
                    # This case should ideally not be reached if j starts from 1 and BACKBONE_ATOMS is not empty.
                    logger.error(f"Unexpected j value: {j} in atom placement loop leading to unhandled NeRF case.")
                    raise IndexError(f"Unexpected j value for NeRF logic: {j}")

                current_atom_coords = calculate_atom_position(
                    a=a_ref_coords_for_nerf,
                    b=prev_prev_atom_coords,
                    c=prev_atom_coords,
                    bond_length_cd=current_bond_length_tensor, # Use the tensor form
                    theta=current_bond_angle_rad,
                    chi=current_torsion_angle_rad_for_nerf,
                    device=device
                )

            residue_coords[i, current_atom_idx, :] = current_atom_coords

            # General log for placed atom
            log_final_placed_atom_name = BACKBONE_ATOMS[j]
            if i == 0 and log_final_placed_atom_name in ["O5'", "C5'", "C4'"]:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"  Placed {log_final_placed_atom_name} (res {i}) at: {current_atom_coords.cpu().detach().numpy()}")

    if torch.isnan(residue_coords).any():
        logger.error("[ERR-RNAPREDICT-NAN-COORD-001] NaN detected in final residue_coords.")
    
    print(f"!!!!!!!!!! CASCADE: build_rna_chain_from_internal_coords COMPLETED. Output shape: {residue_coords.shape} !!!!!!!!!!")
    return residue_coords

def rna_fold(
    scaffolds: Dict[str, torch.Tensor],  # Expects a dict containing 'torsions'
    sequence: str,
    device: str = "cpu",
    do_ring_closure: bool = False,      # New parameter
    debug_logging: bool = False         # New parameter
) -> Dict[str, Any]:
    """
    Main function to fold an RNA sequence given torsion angles within a scaffolds dictionary.
    """
    logger.info(f"[INFO-RNAFOLD] Starting RNA fold for sequence: '{sequence}' with {len(sequence)} residues.")
    logger.info(f"[INFO-RNAFOLD] Using device: {device}, do_ring_closure: {do_ring_closure}, debug_logging: {debug_logging}")

    torch_device = torch.device(device)
    logger.info(f"[INFO-RNAFOLD] Torch device object: {torch_device}")

    num_residues = len(sequence)

    if "torsions" not in scaffolds:
        logger.error("[ERR-RNAFOLD-SCAFFOLD-001] 'torsions' key not found in input scaffolds dictionary.")
        raise ValueError("'torsions' key not found in input scaffolds dictionary.")

    # Ensure torsions from scaffolds are on the correct device
    current_torsions = scaffolds["torsions"].to(torch_device)

    if current_torsions.shape[0] != num_residues:
        logger.error(f"[ERR-RNAFOLD-SHAPE-001] Number of residues in sequence ({num_residues}) "
                     f"does not match torsions in scaffolds ({current_torsions.shape[0]}).")
        raise ValueError("Residue count mismatch between sequence and torsion angles in scaffolds.")

    # Update scaffolds with torsions on the correct device, or create a new one for build_rna_chain_from_internal_coords
    # For simplicity, let's ensure the input 'scaffolds' dict has torsions on the right device.
    # This modifies the input 'scaffolds' dictionary if it's mutable and passed by reference.
    # A safer approach might be to create a new scaffolds dict for internal use if modification is an issue.
    processed_scaffolds = scaffolds.copy() # Avoid modifying the original dict if it's from outside
    processed_scaffolds["torsions"] = current_torsions

    atom_idx_map = {atom_name: i for i, atom_name in enumerate(BACKBONE_ATOMS)}

    coords = build_rna_chain_from_internal_coords(
        scaffolds=processed_scaffolds, # Use the scaffolds dictionary with torsions on the correct device
        num_residues=num_residues,
        device=torch_device,
        atom_idx_map=atom_idx_map
    )

    if do_ring_closure:
        logger.info("[INFO-RNAFOLD] Ring closure requested. Placeholder: not yet fully implemented.")
        # coords = ring_closure_refinement(coords, sequence) # Call actual refinement when available

    output = {
        "sequence": sequence,
        "coordinates": coords,
        "torsion_angles_input": current_torsions, # Report the torsions that were actually used
        "atom_names_ordered": BACKBONE_ATOMS
    }
    logger.info(f"[INFO-RNAFOLD] RNA fold completed. Output coordinates shape: {coords.shape}")
    return output

def ring_closure_refinement(coords: torch.Tensor, sequence: str) -> torch.Tensor:
    """
    Placeholder for ring closure refinement.
    """
    logger.warning("[WARN-RNAPREDICT-NOTIMPL-001] Ring closure refinement is not yet implemented.")
    print("!!!!!!!!!! CASCADE: ring_closure_refinement CALLED (NOT IMPLEMENTED) !!!!!!!!!!")
    return coords
