# rna_predict/dataset/atom_lists.py
# Single source of truth for atom ordering and max atoms per residue

STANDARD_ATOMS = [
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4", # A/G backbone & base
    # Extend as needed for all bases, keep order fixed
]

MAX_ATOMS_PER_RES = len(STANDARD_ATOMS)
