#!/usr/bin/env python

"""
custom_torsion_example.py

Implements a manual approach for computing RNA torsion angles
using Bio.Python for parsing and NumPy for vector math.
"""

import sys
import math
import numpy as np
from Bio.PDB import PDBParser

def calc_dihedral(p1, p2, p3, p4):
    """
    Calculate the dihedral angle (in degrees) for four 3D points 
    p1, p2, p3, p4 (each a NumPy array [x,y,z]).
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-12 or norm_n2 < 1e-12:
        return None  # can't compute angle if cross product is 0

    cos_angle = np.dot(n1, n2) / (norm_n1 * norm_n2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp
    sign = np.dot(b2, np.cross(n1, n2))
    phi = math.acos(cos_angle)
    if sign < 0.0:
        phi = -phi
    return math.degrees(phi)

def get_atom_coord(residue, atom_name):
    """
    Safely fetch a given atom's coordinates in a residue.
    Returns NumPy array or None if atom not found.
    """
    atom = residue.get_atom(atom_name)
    return np.array(atom.coord, dtype=float) if atom else None

def calculate_rna_torsions_custom(pdb_file, chain_id="A"):
    """
    Calculate backbone and glycosidic torsion angles (alpha, beta, gamma,
    delta, epsilon, zeta, chi) for an RNA chain from a PDB file using 
    manual parsing with Bio.Python.
    Returns a dict of angle lists in residue order.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", pdb_file)

    chain = None
    for model in structure:
        for c in model:
            if c.id.strip() == chain_id:
                chain = c
                break
        if chain:  # found the chain
            break
    if not chain:
        raise ValueError(f"Chain {chain_id} not found in {pdb_file}")

    torsion_data = {
        "alpha": [],
        "beta": [],
        "gamma": [],
        "delta": [],
        "epsilon": [],
        "zeta": [],
        "chi":  []
    }

    residues = list(chain.get_residues())
    for i, res in enumerate(residues):
        prev_res = residues[i-1] if i > 0 else None
        next_res = residues[i+1] if i < len(residues) - 1 else None

        # alpha: O3'(i-1) - P(i) - O5'(i) - C5'(i)
        if prev_res:
            O3i_1 = get_atom_coord(prev_res, "O3'")
            Pi    = get_atom_coord(res,       "P")
            O5i   = get_atom_coord(res,       "O5'")
            C5i   = get_atom_coord(res,       "C5'")
            alpha_val = None
            if None not in (O3i_1, Pi, O5i, C5i):
                alpha_val = calc_dihedral(O3i_1, Pi, O5i, C5i)
            torsion_data["alpha"].append(alpha_val)
        else:
            torsion_data["alpha"].append(None)

        # beta: P(i) - O5'(i) - C5'(i) - C4'(i)
        Pi   = get_atom_coord(res, "P")
        O5i  = get_atom_coord(res, "O5'")
        C5i  = get_atom_coord(res, "C5'")
        C4i  = get_atom_coord(res, "C4'")
        beta_val = None
        if None not in (Pi, O5i, C5i, C4i):
            beta_val = calc_dihedral(Pi, O5i, C5i, C4i)
        torsion_data["beta"].append(beta_val)

        # gamma: O5'(i) - C5'(i) - C4'(i) - C3'(i)
        C3i = get_atom_coord(res, "C3'")
        gamma_val = None
        if None not in (O5i, C5i, C4i, C3i):
            gamma_val = calc_dihedral(O5i, C5i, C4i, C3i)
        torsion_data["gamma"].append(gamma_val)

        # delta: C5'(i) - C4'(i) - C3'(i) - O3'(i)
        O3i = get_atom_coord(res, "O3'")
        delta_val = None
        if None not in (C5i, C4i, C3i, O3i):
            delta_val = calc_dihedral(C5i, C4i, C3i, O3i)
        torsion_data["delta"].append(delta_val)

        # epsilon: C4'(i) - C3'(i) - O3'(i) - P(i+1)
        epsilon_val = None
        if next_res:
            P_ip1 = get_atom_coord(next_res, "P")
            if None not in (C4i, C3i, O3i, P_ip1):
                epsilon_val = calc_dihedral(C4i, C3i, O3i, P_ip1)
        torsion_data["epsilon"].append(epsilon_val)

        # zeta: C3'(i) - O3'(i) - P(i+1) - O5'(i+1)
        zeta_val = None
        if next_res:
            O5_ip1 = get_atom_coord(next_res, "O5'")
            P_ip1  = get_atom_coord(next_res, "P")
            if None not in (C3i, O3i, P_ip1, O5_ip1):
                zeta_val = calc_dihedral(C3i, O3i, P_ip1, O5_ip1)
        torsion_data["zeta"].append(zeta_val)

        # chi: glycosidic angle
        # For Purines (A,G): O4' - C1' - N9 - C4
        # For Pyrimidines (U,C): O4' - C1' - N1 - C2
        resname = res.get_resname().strip().upper()
        O4  = get_atom_coord(res, "O4'")
        C1  = get_atom_coord(res, "C1'")
        if resname.startswith("A") or resname.startswith("G"):
            N_base = get_atom_coord(res, "N9")
            C_base = get_atom_coord(res, "C4")
        else:
            N_base = get_atom_coord(res, "N1")
            C_base = get_atom_coord(res, "C2")

        chi_val = None
        if None not in (O4, C1, N_base, C_base):
            chi_val = calc_dihedral(O4, C1, N_base, C_base)
        torsion_data["chi"].append(chi_val)

    return torsion_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python custom_torsion_example.py <rna_structure.pdb> [chainID]")
        sys.exit(1)

    pdb_path = sys.argv[1]
    chain_id = sys.argv[2] if len(sys.argv) >= 3 else "A"

    torsions = calculate_rna_torsions_custom(pdb_path, chain_id)
    for angle_name, values in torsions.items():
        print(f"{angle_name}: {values}")

if __name__ == "__main__":
    main()
