"""
Protein sidechain data structures and constants.
Originally from https://github.com/jonathanking/sidechainnet
Modified by considering rigid bodies in sidechains (remove extra torsions)
"""

from typing import Dict, List, Union

# Backbone build information
BB_BUILD_INFO = {
    "BONDLENS": {
        "c-n": 1.329,
        "n-ca": 1.458,
        "ca-c": 1.525,
        "c-o": 1.231,
    },
    "BONDANGS": {
        "ca-c-n": 2.124,
        "c-n-ca": 2.035,
        "n-ca-c": 1.939,
        "ca-c-o": 2.094,
    },
    "BONDTORSIONS": {
        "n-ca-c-n": 3.141592,
        "ca-n-c-ca": 3.141592,
        "c-n-ca-c": 0.0,
        "n-ca-c-o": 0.0,
    },
}

# Atom token IDs
ATOM_TOKEN_IDS = {
    "N": 0,
    "CA": 1,
    "C": 2,
    "O": 3,
    "CB": 4,
    "CG": 5,
    "CD": 6,
    "CE": 7,
    "CZ": 8,
    "CH": 9,
    "NH1": 10,
    "NH2": 11,
    "ND1": 12,
    "ND2": 13,
    "NE": 14,
    "NE1": 15,
    "NE2": 16,
    "NZ": 17,
    "OG": 18,
    "OG1": 19,
    "OD1": 20,
    "OD2": 21,
    "OE1": 22,
    "OE2": 23,
    "OH": 24,
    "SG": 25,
    "SD": 26,
    "PAD": 27,
}

# Sidechain build information
SC_BUILD_INFO: Dict[str, Dict[str, Union[List[str], List[float], List[List[int]]]]] = {
    "A": {
        "angles-names": ["N-CA-CB"],
        "angles-types": ["N -CX-CT"],
        "angles-vals": [1.9146261894377796],
        "atom-names": ["CB"],
        "bonds-names": ["CA-CB"],
        "bonds-types": ["CX-CT"],
        "bonds-vals": [1.526],
        "torsion-names": ["C-N-CA-CB"],
        "torsion-types": ["C -N -CX-CT"],
        "torsion-vals": ["p"],
        "rigid-frames-idxs": [[0, 1, 2], [0, 1, 4]],
    },
    # ... existing code ...
}

# MP3SC information
MP3SC_INFO: Dict[str, Dict[str, Dict[str, float]]] = {
    "A": {
        "CB": {"bond_dihedral": 0.0},
    },
    # ... existing code ...
}

# Amino acid index mapping
INDEX2AAS = "ACDEFGHIKLMNPQRSTVWY_"
AAS2INDEX = {aa: i for i, aa in enumerate(INDEX2AAS)}

# Ambiguous sidechain atoms
AMBIGUOUS = {
    "D": {
        "names": [["OD1", "OD2"]],
        "indexs": [[6, 7]],
    },
    "E": {
        "names": [["OE1", "OE2"]],
        "indexs": [[7, 8]],
    },
    "F": {
        "names": [["CD1", "CD2"], ["CE1", "CE2"]],
        "indexs": [[6, 10], [7, 9]],
    },
    "Y": {
        "names": [["CD1", "CD2"], ["CE1", "CE2"]],
        "indexs": [[6, 10], [7, 9]],
    },
    "R": {"names": [["NH1", "NH2"]], "indexs": [[9, 10]]},
}

# BLOSUM substitution matrix
BLOSUM = {
    "A": [
        4.0,
        -1.0,
        -2.0,
        -2.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
        -2.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -2.0,
        -1.0,
        1.0,
        0.0,
        -3.0,
        -2.0,
        0.0,
        0.0,
    ],
    # ... existing code ...
}

# Add other static data structures here

SUPREME_INFO = {
    "A": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,  # so (mask != mask) is True
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[0, 1, 2] for _ in range(11)],
        "rigid_idx_mask": [0, 1, 2],
        "atom_token_mask": [1] * 14,
    },
    "C": {
        "cloud_mask": [False] * 14,
        "bond_mask": [2.0] * 14,
        "theta_mask": [1.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.0] * 14,
        "idx_mask": [[3, 4, 5] for _ in range(11)],
        "rigid_idx_mask": [3, 4, 5],
        "atom_token_mask": [1] * 14,
    },
    "D": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[6, 7, 8] for _ in range(11)],
        "rigid_idx_mask": [6, 7, 8],
        "atom_token_mask": [1] * 14,
    },
    "E": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[9, 10, 11] for _ in range(11)],
        "rigid_idx_mask": [9, 10, 11],
        "atom_token_mask": [1] * 14,
    },
    "F": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[12, 13, 14] for _ in range(11)],
        "rigid_idx_mask": [12, 13, 14],
        "atom_token_mask": [1] * 14,
    },
    "G": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[15, 16, 17] for _ in range(11)],
        "rigid_idx_mask": [15, 16, 17],
        "atom_token_mask": [1] * 14,
    },
    "H": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[18, 19, 20] for _ in range(11)],
        "rigid_idx_mask": [18, 19, 20],
        "atom_token_mask": [1] * 14,
    },
    "I": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[21, 22, 23] for _ in range(11)],
        "rigid_idx_mask": [21, 22, 23],
        "atom_token_mask": [1] * 14,
    },
    "K": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[24, 25, 26] for _ in range(11)],
        "rigid_idx_mask": [24, 25, 26],
        "atom_token_mask": [1] * 14,
    },
    "L": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[27, 28, 29] for _ in range(11)],
        "rigid_idx_mask": [27, 28, 29],
        "atom_token_mask": [1] * 14,
    },
    "M": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[30, 31, 32] for _ in range(11)],
        "rigid_idx_mask": [30, 31, 32],
        "atom_token_mask": [1] * 14,
    },
    "N": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[33, 34, 35] for _ in range(11)],
        "rigid_idx_mask": [33, 34, 35],
        "atom_token_mask": [1] * 14,
    },
    "P": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[36, 37, 38] for _ in range(11)],
        "rigid_idx_mask": [36, 37, 38],
        "atom_token_mask": [1] * 14,
    },
    "Q": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[39, 40, 41] for _ in range(11)],
        "rigid_idx_mask": [39, 40, 41],
        "atom_token_mask": [1] * 14,
    },
    "R": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[42, 43, 44] for _ in range(11)],
        "rigid_idx_mask": [42, 43, 44],
        "atom_token_mask": [1] * 14,
    },
    "S": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[45, 46, 47] for _ in range(11)],
        "rigid_idx_mask": [45, 46, 47],
        "atom_token_mask": [1] * 14,
    },
    "T": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[48, 49, 50] for _ in range(11)],
        "rigid_idx_mask": [48, 49, 50],
        "atom_token_mask": [1] * 14,
    },
    "V": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[51, 52, 53] for _ in range(11)],
        "rigid_idx_mask": [51, 52, 53],
        "atom_token_mask": [1] * 14,
    },
    "W": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[54, 55, 56] for _ in range(11)],
        "rigid_idx_mask": [54, 55, 56],
        "atom_token_mask": [1] * 14,
    },
    "Y": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[57, 58, 59] for _ in range(11)],
        "rigid_idx_mask": [57, 58, 59],
        "atom_token_mask": [1] * 14,
    },
    "_": {
        "cloud_mask": [False] * 14,  # No atoms present
        "bond_mask": [0.0] * 14,  # No bonds
        "theta_mask": [0.0] * 14,  # No angles
        "torsion_mask": [float("nan")] * 14,  # No torsions
        "torsion_mask_filled": [0.0] * 14,  # No torsions
        "idx_mask": [[0, 0, 0]] * 11,  # Default indices
        "rigid_idx_mask": [0, 0, 0],  # Default indices
        "atom_token_mask": [0] * 14,  # Padding/invalid atom tokens
    },
}

# Sidechain angles, bonds, and mask information
SIDECHAIN_ANGLES = {
    "A": [1.9146261894377796],  # N-CA-CB
    "C": [1.9146261894377796],  # N-CA-CB
    "D": [1.9146261894377796, 1.9146261894377796],  # N-CA-CB, CA-CB-CG
    "E": [
        1.9146261894377796,
        1.9146261894377796,
        1.9146261894377796,
    ],  # N-CA-CB, CA-CB-CG, CB-CG-CD
    "F": [1.9146261894377796, 1.9146261894377796],  # N-CA-CB, CA-CB-CG
    "G": [],  # No sidechain
    "H": [1.9146261894377796, 1.9146261894377796],  # N-CA-CB, CA-CB-CG
    "I": [1.9146261894377796, 1.9146261894377796],  # N-CA-CB, CA-CB-CG1
    "K": [
        1.9146261894377796,
        1.9146261894377796,
        1.9146261894377796,
        1.9146261894377796,
    ],  # N-CA-CB, CA-CB-CG, CB-CG-CD, CG-CD-CE
    "L": [1.9146261894377796, 1.9146261894377796],  # N-CA-CB, CA-CB-CG
    "M": [
        1.9146261894377796,
        1.9146261894377796,
        1.9146261894377796,
    ],  # N-CA-CB, CA-CB-CG, CB-CG-SD
    "N": [1.9146261894377796, 1.9146261894377796],  # N-CA-CB, CA-CB-CG
    "P": [1.9146261894377796, 1.9146261894377796],  # N-CA-CB, CA-CB-CG
    "Q": [
        1.9146261894377796,
        1.9146261894377796,
        1.9146261894377796,
    ],  # N-CA-CB, CA-CB-CG, CB-CG-CD
    "R": [
        1.9146261894377796,
        1.9146261894377796,
        1.9146261894377796,
        1.9146261894377796,
    ],  # N-CA-CB, CA-CB-CG, CB-CG-CD, CG-CD-NE
    "S": [1.9146261894377796],  # N-CA-CB
    "T": [1.9146261894377796],  # N-CA-CB
    "V": [1.9146261894377796],  # N-CA-CB
    "W": [1.9146261894377796, 1.9146261894377796],  # N-CA-CB, CA-CB-CG
    "Y": [1.9146261894377796, 1.9146261894377796],  # N-CA-CB, CA-CB-CG
    "_": [],  # Padding
}

SIDECHAIN_BONDS = {
    "A": [1.526],  # CA-CB
    "C": [1.526],  # CA-CB
    "D": [1.526, 1.526],  # CA-CB, CB-CG
    "E": [1.526, 1.526, 1.526],  # CA-CB, CB-CG, CG-CD
    "F": [1.526, 1.526],  # CA-CB, CB-CG
    "G": [],  # No sidechain
    "H": [1.526, 1.526],  # CA-CB, CB-CG
    "I": [1.526, 1.526],  # CA-CB, CB-CG1
    "K": [1.526, 1.526, 1.526, 1.526],  # CA-CB, CB-CG, CG-CD, CD-CE
    "L": [1.526, 1.526],  # CA-CB, CB-CG
    "M": [1.526, 1.526, 1.526],  # CA-CB, CB-CG, CG-SD
    "N": [1.526, 1.526],  # CA-CB, CB-CG
    "P": [1.526, 1.526],  # CA-CB, CB-CG
    "Q": [1.526, 1.526, 1.526],  # CA-CB, CB-CG, CG-CD
    "R": [1.526, 1.526, 1.526, 1.526],  # CA-CB, CB-CG, CG-CD, CD-NE
    "S": [1.526],  # CA-CB
    "T": [1.526],  # CA-CB
    "V": [1.526],  # CA-CB
    "W": [1.526, 1.526],  # CA-CB, CB-CG
    "Y": [1.526, 1.526],  # CA-CB, CB-CG
    "_": [],  # Padding
}

SIDECHAIN_MASK = {
    "A": [True] * 1,
    "C": [True] * 1,
    "D": [True] * 2,
    "E": [True] * 3,
    "F": [True] * 2,
    "G": [],
    "H": [True] * 2,
    "I": [True] * 2,
    "K": [True] * 4,
    "L": [True] * 2,
    "M": [True] * 3,
    "N": [True] * 2,
    "P": [True] * 2,
    "Q": [True] * 3,
    "R": [True] * 4,
    "S": [True] * 1,
    "T": [True] * 1,
    "V": [True] * 1,
    "W": [True] * 2,
    "Y": [True] * 2,
    "_": [],
}
