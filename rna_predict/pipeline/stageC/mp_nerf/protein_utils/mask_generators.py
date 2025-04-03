"""
Mask generation utilities for protein structures.
Functions for generating various masks and indices for protein structure manipulation.

Updated to correctly handle '_' and use SUPREME_INFO.
"""

import numpy as np

from .sidechain_data import BB_BUILD_INFO
from .supreme_data import SUPREME_INFO


def make_cloud_mask(aa: str) -> np.ndarray:
    """Generate cloud mask for a given amino acid.

    Args:
        aa: Amino acid code (single uppercase letter or '_')

    Returns:
        np.ndarray: Mask containing True for atoms that should be included in the cloud (shape 14,)
    """
    if aa == "_":
        return np.array(SUPREME_INFO["_"]["cloud_mask"], dtype=bool)
    if aa not in SUPREME_INFO:
        raise KeyError(f"Invalid amino acid code: {aa}")
    return np.array(SUPREME_INFO[aa]["cloud_mask"], dtype=bool)


def make_bond_mask(aa: str) -> np.ndarray:
    """Generate bond mask for a given amino acid.

    Args:
        aa: Amino acid code (single uppercase letter or '_')

    Returns:
        np.ndarray: Mask containing bond lengths for each atom (shape 14,)
    """
    if aa == "_":
        return np.array(SUPREME_INFO["_"]["bond_mask"], dtype=float)
    if aa not in SUPREME_INFO:
        raise KeyError(f"Invalid amino acid code: {aa}")
    return np.array(SUPREME_INFO[aa]["bond_mask"], dtype=float)


def make_theta_mask(aa: str) -> np.ndarray:
    """Generate theta mask for a given amino acid (bond angles).

    Args:
        aa: Amino acid code (single uppercase letter or '_')

    Returns:
        np.ndarray: Mask containing bond angles for each atom (shape 14,)
    """
    if aa == "_":
        return np.array(SUPREME_INFO["_"]["theta_mask"], dtype=float)
    if aa not in SUPREME_INFO:
        raise KeyError(f"Invalid amino acid code: {aa}")
    return np.array(SUPREME_INFO[aa]["theta_mask"], dtype=float)


def make_torsion_mask(aa: str, fill: bool = False) -> np.ndarray:
    """Generate torsion mask for a given amino acid (dihedral angles).

    Args:
        aa: Amino acid code (single uppercase letter or '_')
        fill: Whether to use the 'torsion_mask_filled' (no NaNs) or 'torsion_mask' (may contain NaNs).

    Returns:
        np.ndarray: Mask containing dihedral angles for each atom (shape 14,)
    """
    mask_key = "torsion_mask_filled" if fill else "torsion_mask"
    if aa == "_":
        # Ensure underscore returns float array even if values are 0 or nan
        return np.array(SUPREME_INFO["_"][mask_key], dtype=float)
    if aa not in SUPREME_INFO:
        raise KeyError(f"Invalid amino acid code: {aa}")
    # Ensure return type is float, handling potential NaNs
    return np.array(SUPREME_INFO[aa][mask_key], dtype=float)


def make_idx_mask(aa: str) -> np.ndarray:
    """Generate index mask for a given amino acid (NeRF references).

    Args:
        aa: Amino acid code (single uppercase letter or '_')

    Returns:
        np.ndarray: Mask containing indices for each atom (shape 11, 3)
    """
    if aa == "_":
        # Return the default index mask for underscore
        return np.array(SUPREME_INFO["_"]["idx_mask"], dtype=int)
    if aa not in SUPREME_INFO:
        raise KeyError(f"Invalid amino acid code: {aa}")
    # Return as integer array
    return np.array(SUPREME_INFO[aa]["idx_mask"], dtype=int)


def make_atom_token_mask(aa: str) -> np.ndarray:
    """Generate atom token mask for a given amino acid.

    Args:
        aa: Amino acid code (single uppercase letter or '_')

    Returns:
        np.ndarray: Mask containing token IDs for each atom (shape 14,)
    """
    if aa == "_":
        # Return the all-zero mask for underscore
        return np.array(SUPREME_INFO["_"]["atom_token_mask"], dtype=int)
    if aa not in SUPREME_INFO:
        raise KeyError(f"Invalid amino acid code: {aa}")
    # Return integer array of token IDs
    return np.array(SUPREME_INFO[aa]["atom_token_mask"], dtype=int)


def scn_angle_mask(seq: str) -> np.ndarray:
    """Generate SideChainNet angle mask for a sequence.
    (Potentially legacy or needs update based on SUPREME_INFO usage)
    Args:
        seq: Amino acid sequence
    Returns:
        np.ndarray: Mask containing angles for each residue (shape L, 12)
    """
    mask = np.zeros((len(seq), 12))  # 3 backbone + 3 bond + 6 sidechain angles
    # This part relies on BB_BUILD_INFO and SC_BUILD_INFO, which might be outdated
    # compared to SUPREME_INFO. Consider refactoring if needed.
    for i, aa in enumerate(seq):
        if aa != "_":
            # Use dummy values if BB_BUILD_INFO is missing keys
            ca_c_n_angle = BB_BUILD_INFO.get("BONDANGS", {}).get(
                "ca-c-n", 2.124
            )  # Default value
            n_ca_c_angle = BB_BUILD_INFO.get("BONDANGS", {}).get(
                "n-ca-c", 1.939
            )  # Default value
            c_n_ca_angle = BB_BUILD_INFO.get("BONDANGS", {}).get(
                "c-n-ca", 2.035
            )  # Default value

            mask[i, :3] = [
                np.nan,
                np.nan,
                np.nan,
            ]  # Placeholder for torsions phi, psi, omega
            mask[i, 3:6] = [n_ca_c_angle, ca_c_n_angle, c_n_ca_angle]

            # Retrieve angles from SUPREME_INFO's theta_mask instead of SC_BUILD_INFO if consistency is desired
            # Example: theta_vals = SUPREME_INFO.get(aa, {}).get("theta_mask", [0.0]*14)
            # mask[i, 6:6+len(placeholder_sidechain_angles)] = placeholder_sidechain_angles
    return mask


def scn_bond_mask(seq: str) -> np.ndarray:
    """Generate SideChainNet bond mask for a sequence.
    (Potentially legacy or needs update based on SUPREME_INFO usage)
    Args:
        seq: Amino acid sequence
    Returns:
        np.ndarray: Mask containing bond lengths for each residue (shape L, 14)
    """
    mask = np.zeros((len(seq), 14))  # N, CA, C, O, CB, ...
    # This relies on BB_BUILD_INFO and potentially SC_BUILD_INFO's bonds-vals
    for i, aa in enumerate(seq):
        if aa != "_":
            mask[i, 0] = BB_BUILD_INFO.get("BONDLENS", {}).get("c-n", 1.329)
            mask[i, 1] = BB_BUILD_INFO.get("BONDLENS", {}).get("n-ca", 1.458)
            mask[i, 2] = BB_BUILD_INFO.get("BONDLENS", {}).get("ca-c", 1.525)
            mask[i, 3] = BB_BUILD_INFO.get("BONDLENS", {}).get("c-o", 1.231)
            # Consider retrieving sidechain bonds from SUPREME_INFO if needed
            # Example: bond_vals = SUPREME_INFO.get(aa, {}).get("bond_mask", [0.0]*14)
            # num_sc_bonds = sum(1 for x in bond_vals[4:] if x != 0.0) # Count non-zero sidechain bonds
            # mask[i, 4:4+num_sc_bonds] = [x for x in bond_vals[4:] if x != 0.0]
    return mask


def scn_cloud_mask(seq: str, coords=None, strict=False) -> np.ndarray:
    """Generate SideChainNet cloud mask for a sequence.
    (Now delegates to make_cloud_mask for consistency)
    Args:
        seq: Amino acid sequence
        coords: Optional coordinates tensor (currently unused in delegation)
        strict: Optional strict flag (currently unused in delegation)
    Returns:
        np.ndarray: Mask containing True for atoms that should be included (shape L, 14)
    """
    # Simply use make_cloud_mask for each amino acid
    masks = [make_cloud_mask(aa) for aa in seq]
    return np.array(masks, dtype=bool)


def scn_index_mask(seq: str) -> np.ndarray:
    """Generate SideChainNet index mask for a sequence.
    (Now delegates to make_idx_mask for consistency)
    Args:
        seq: Amino acid sequence
    Returns:
        np.ndarray: Mask containing indices for each atom (shape L, 11, 3)
    """
    masks = [make_idx_mask(aa) for aa in seq]
    # Stack along the first dimension (sequence length)
    return np.array(masks, dtype=int)


def scn_rigid_index_mask(seq: str, c_alpha: bool = False) -> np.ndarray:
    """Generate SideChainNet rigid index mask for a sequence.
    (Now delegates to make_idx_mask and extracts the first row)
    Args:
        seq: Amino acid sequence
        c_alpha: (Currently ignored in delegation)
    Returns:
        np.ndarray: Mask containing rigid frame indices for each residue (shape L, 3)
    """
    # Use make_idx_mask and take the first row (backbone reference) for each residue
    idx_masks = [make_idx_mask(aa) for aa in seq]
    rigid_masks = [
        mask[0] for mask in idx_masks
    ]  # Take the first row [0, 1, 2] or similar
    return np.array(rigid_masks, dtype=int)


# Add other mask generation functions here
