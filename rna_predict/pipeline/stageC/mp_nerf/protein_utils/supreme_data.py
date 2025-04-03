"""
Data structures for protein structure manipulation.
Contains the SUPREME_INFO dictionary with cloud masks and other structural information
for standard amino acids.

Updated to include 'atom_token_mask' and more realistic placeholders.
"""

import numpy as np


# Placeholder logic for generating realistic-ish masks based on typical atom counts
# Backbone: N, CA, C, O (4 atoms)
# Sidechain varies. Max is 14 total atoms (including backbone).
def generate_mask(num_atoms, length=14, default_val=0.0, active_val=1.0, dtype=float):
    mask = np.full(length, default_val, dtype=dtype)
    mask[:num_atoms] = active_val
    return mask.tolist()


def generate_bool_mask(num_atoms, length=14):
    return generate_mask(
        num_atoms, length, default_val=False, active_val=True, dtype=bool
    )


# Generate somewhat plausible index masks (replace with real data if available)
def generate_idx_mask(num_sidechain_torsions, length=11):
    mask = np.zeros((length, 3), dtype=int)
    if num_sidechain_torsions > 0:  # Only set backbone reference if we have atoms
        mask[0] = [0, 1, 2]  # Backbone reference
        for i in range(num_sidechain_torsions):
            # Simple placeholder: Assume prev 3 atoms are references
            # This is NOT biochemically accurate but fills the shape
            base_idx = 3 + i  # Start indexing after backbone
            mask[i + 1] = [max(0, base_idx - 2), max(0, base_idx - 1), base_idx]
    return mask.tolist()


# Generate atom token masks
def generate_atom_token_ids(aa_code, atoms_present_mask):
    # Map standard atom names to their token IDs
    # Example: N=0, CA=1, C=2, O=3, CB=4, etc.
    token_mask = np.zeros(14, dtype=int)
    # Assign backbone tokens - use non-zero values for all present atoms
    if atoms_present_mask[0]:
        token_mask[0] = 1  # N
    if atoms_present_mask[1]:
        token_mask[1] = 2  # CA
    if atoms_present_mask[2]:
        token_mask[2] = 3  # C
    if atoms_present_mask[3]:
        token_mask[3] = 4  # O
    # Placeholder for sidechain - assign CB if present
    if atoms_present_mask[4]:
        token_mask[4] = 5  # CB
    # Assign generic tokens for other present sidechain atoms
    for i in range(5, 14):
        if atoms_present_mask[i]:
            # Use a generic sidechain token ID
            token_mask[i] = 6  # Generic sidechain atom
    return token_mask.tolist()


SUPREME_INFO = {
    # Amino Acid: Num Atoms, Num Sidechain Torsions (approx for idx_mask)
    "A": {"atoms": 5, "sc_torsions": 1},
    "C": {"atoms": 6, "sc_torsions": 1},
    "D": {"atoms": 8, "sc_torsions": 2},
    "E": {"atoms": 9, "sc_torsions": 3},
    "F": {"atoms": 11, "sc_torsions": 2},
    "G": {
        "atoms": 4,
        "sc_torsions": 0,
    },  # Glycine has no sidechain atoms beyond backbone
    "H": {"atoms": 10, "sc_torsions": 2},
    "I": {"atoms": 8, "sc_torsions": 2},
    "K": {"atoms": 9, "sc_torsions": 4},
    "L": {"atoms": 8, "sc_torsions": 2},
    "M": {"atoms": 8, "sc_torsions": 3},
    "N": {"atoms": 8, "sc_torsions": 2},
    "P": {"atoms": 7, "sc_torsions": 1},  # Proline is special
    "Q": {"atoms": 9, "sc_torsions": 3},
    "R": {"atoms": 11, "sc_torsions": 4},
    "S": {"atoms": 6, "sc_torsions": 1},
    "T": {"atoms": 7, "sc_torsions": 1},
    "V": {"atoms": 7, "sc_torsions": 1},
    "W": {"atoms": 14, "sc_torsions": 2},
    "Y": {"atoms": 12, "sc_torsions": 2},
    "_": {"atoms": 0, "sc_torsions": 0},  # Padding
}

# Populate the full SUPREME_INFO dictionary
for aa, info in SUPREME_INFO.items():
    num_atoms = info["atoms"]
    sc_torsions = info["sc_torsions"]
    cloud_m = generate_bool_mask(num_atoms)
    info["cloud_mask"] = cloud_m
    # Generate masks based on the number of atoms
    info["bond_mask"] = generate_mask(
        num_atoms, default_val=0.0, active_val=1.5
    )  # Example bond length
    info["theta_mask"] = generate_mask(
        num_atoms, default_val=0.0, active_val=np.pi * 2 / 3
    )  # Example angle (120 deg)
    # Torsion mask: NaNs for placeholders, zeros for filled
    torsion_m_nan = generate_mask(
        num_atoms, default_val=np.nan, active_val=np.nan
    )  # Initially all NaN
    # Fill backbone torsions (first 4) if atoms exist
    for i in range(min(num_atoms, 4)):
        torsion_m_nan[i] = 0.0  # Placeholder 0.0 for backbone
    # Fill sidechain torsion if present
    if num_atoms > 4:
        torsion_m_nan[4] = 0.0  # Placeholder 0.0 for first sidechain torsion
    info["torsion_mask"] = torsion_m_nan
    # Filled version has non-zero values for first num_atoms elements, zeros for the rest
    torsion_m_filled = np.zeros(14, dtype=float)  # Initially all zeros
    torsion_m_filled[:num_atoms] = 1.0  # Set ones for existing atoms
    info["torsion_mask_filled"] = torsion_m_filled.tolist()
    # Index masks
    idx_mask = generate_idx_mask(sc_torsions)
    info["idx_mask"] = idx_mask
    info["rigid_idx_mask"] = (
        idx_mask[0] if sc_torsions > 0 else [0, 0, 0]
    )  # Use first row as rigid body reference
    # Atom token mask - Use the boolean cloud mask to determine which tokens to assign
    info["atom_token_mask"] = generate_atom_token_ids(aa, cloud_m)

# Special case for Glycine (G) - Ensure sidechain atoms are masked out
SUPREME_INFO["G"]["cloud_mask"] = generate_bool_mask(4)
SUPREME_INFO["G"]["bond_mask"] = generate_mask(4, default_val=0.0, active_val=1.5)
SUPREME_INFO["G"]["theta_mask"] = generate_mask(
    4, default_val=0.0, active_val=np.pi * 2 / 3
)
SUPREME_INFO["G"]["torsion_mask"] = generate_mask(
    4, default_val=np.nan, active_val=0.0
)  # Backbone only
torsion_m_filled_g = np.zeros(14, dtype=float)  # Initially all zeros
torsion_m_filled_g[:4] = 1.0  # Set ones for backbone atoms
SUPREME_INFO["G"]["torsion_mask_filled"] = torsion_m_filled_g.tolist()
idx_mask_g = generate_idx_mask(0)  # No sidechain torsions
SUPREME_INFO["G"]["idx_mask"] = idx_mask_g
SUPREME_INFO["G"]["rigid_idx_mask"] = (
    idx_mask_g[0] if len(idx_mask_g) > 0 else [0, 0, 0]
)
SUPREME_INFO["G"]["atom_token_mask"] = generate_atom_token_ids(
    "G", SUPREME_INFO["G"]["cloud_mask"]
)

# Special case for Padding (_) - Ensure all masks are zero/False/NaN appropriately
SUPREME_INFO["_"]["cloud_mask"] = generate_bool_mask(0)
SUPREME_INFO["_"]["bond_mask"] = generate_mask(0, default_val=0.0, active_val=0.0)
SUPREME_INFO["_"]["theta_mask"] = generate_mask(0, default_val=0.0, active_val=0.0)
SUPREME_INFO["_"]["torsion_mask"] = generate_mask(
    0, default_val=np.nan, active_val=np.nan
)
SUPREME_INFO["_"]["torsion_mask_filled"] = generate_mask(
    0, default_val=0.0, active_val=0.0
)
idx_mask_pad = generate_idx_mask(0)  # This will return all zeros now
SUPREME_INFO["_"]["idx_mask"] = idx_mask_pad
SUPREME_INFO["_"]["rigid_idx_mask"] = [0, 0, 0]  # Default rigid idx
SUPREME_INFO["_"]["atom_token_mask"] = generate_atom_token_ids(
    "_", SUPREME_INFO["_"]["cloud_mask"]
)


# SUPREME_MASK is a simplified version of SUPREME_INFO that only contains the cloud mask
SUPREME_MASK = {aa: info["cloud_mask"] for aa, info in SUPREME_INFO.items()}
