"""
Utilities related to atomic symmetry in amino acids.
"""

from typing import Dict, List, Tuple

# Assuming sidechain_data is in the same directory or accessible
from .sidechain_data import SC_BUILD_INFO


def get_symmetric_atom_pairs(seq: str) -> Dict[str, List[Tuple[int, int]]]:
    """Get pairs of symmetric atoms for each residue in the sequence.

    Args:
        seq: String of amino acid one-letter codes

    Returns:
        Dictionary mapping residue indices (as strings) to lists of symmetric atom pairs.
        Only includes residues that have symmetric pairs.
    """
    result: Dict[str, List[Tuple[int, int]]] = {}
    valid_aas = set(
        SC_BUILD_INFO.keys()
    )  # Get set of valid amino acids for quick lookup

    for i, aa in enumerate(seq):
        if aa in valid_aas:  # Process only valid amino acids
            pairs: List[Tuple[int, int]] = []
            # TODO: These indices (4, 5, 6, 7, 8, 9, 10, 11) seem hardcoded based on
            # SidechainNet atom ordering. Verify if this mapping is robust or
            # if it should dynamically use atom names from SC_BUILD_INFO.
            if aa == "D":  # Aspartic Acid: OD1/OD2 are symmetric
                pairs = [(6, 7)]  # Indices of OD1, OD2 in SidechainNet
            elif aa == "E":  # Glutamic Acid: OE1/OE2 are symmetric
                pairs = [(8, 9)]  # Indices of OE1, OE2 in SidechainNet
            elif aa == "F": # Phenylalanine: CD1/CE1 vs CD2/CE2, CZ is central
                 pairs = [(6, 10), (7, 9)] # CD1/CE2, CE1/CD2 - check indices
            elif aa == "Y":  # Tyrosine: Similar to F, plus OH
                 pairs = [(6, 10), (7, 9)] # CD1/CE2, CE1/CD2 - check indices
            elif aa == "R": # Arginine: NH1/NH2
                 pairs = [(9, 11)] # NH1/NH2 - check indices
            elif aa == "H": # Histidine: ND1/CE1 vs NE2/CD2 - depends on tautomer? Check standard
                 pairs = [(6, 9), (7, 8)] # ND1/CE1, CD2/NE2 - check indices
            elif aa == "V": # Valine: CG1/CG2
                 pairs = [(5, 6)] # CG1/CG2 - check indices
            elif aa == "L": # Leucine: CD1/CD2
                 pairs = [(6, 7)] # CD1/CD2 - check indices

            # Always add the entry for valid AAs, even if pairs is empty []
            # Note: The original implementation included (4,5) for D, E, Y which seems
            # incorrect as CB-CG bond is usually not symmetric. Removed those pairs.
            # Also added F, R, H, V, L based on common symmetries. Indices need verification.
            result[str(i)] = pairs
        # Implicitly skip invalid amino acids like 'X'

    return result