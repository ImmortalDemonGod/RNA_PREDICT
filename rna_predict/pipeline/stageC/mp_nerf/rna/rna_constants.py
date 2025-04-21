"""
RNA-specific constants and data structures for MP-NeRF implementation.
"""


###############################################################################
# We'll use a standard ordering for the backbone atoms.
BACKBONE_ATOMS = [
    "P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]

# Map from atom name to index in the BACKBONE_ATOMS list
BACKBONE_INDEX_MAP = {atom: i for i, atom in enumerate(BACKBONE_ATOMS)}

###############################################################################
# 2) STANDARD TORSION ANGLES FOR RNA BACKBONE (A-form)
###############################################################################
# These are the standard torsion angles for RNA backbone in A-form
# Values are in degrees
RNA_BACKBONE_TORSIONS_AFORM = {
    "alpha": -60.0,  # P-O5'-C5'-C4'
    "beta": 180.0,  # O5'-C5'-C4'-C3'
    "gamma": 60.0,  # C5'-C4'-C3'-O3'
    "delta": 80.0,  # C4'-C3'-O3'-P
    "epsilon": -150.0,  # C3'-O3'-P-O5'
    "zeta": -70.0,  # O3'-P-O5'-C5'
    "chi": -160.0,  # O4'-C1'-N9/N1-C4/C2 (purine/pyrimidine)
}
