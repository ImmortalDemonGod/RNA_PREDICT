#!/usr/bin/env python

"""
mdanalysis_torsion_example.py

Implements a method for computing RNA torsion angles using MDAnalysis.
"""

import sys
import numpy as np
import MDAnalysis as mda

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
    phi = np.arccos(cos_angle)
    if sign < 0.0:
        phi = -phi
    return np.degrees(phi)

def calculate_rna_torsions_mdanalysis(pdb_file, chain_id="A"):
    """
    Calculate backbone and glycosidic torsion angles (alpha, beta, gamma,
    delta, epsilon, zeta, chi) for an RNA chain from a PDB file using 
    MDAnalysis.
    Returns a dict of angle lists in residue order.
    """
    u = mda.Universe(pdb_file)
    chain = u.select_atoms(f"chainID {chain_id}")

    torsion_data = {
        "alpha": [],
        "beta": [],
        "gamma": [],
        "delta": [],
        "epsilon": [],
        "zeta": [],
        "chi":  []
    }

    residues = chain.residues
    for i, res in enumerate(residues):
        prev_res = residues[i-1] if i > 0 else None
        next_res = residues[i+1] if i < len(residues) - 1 else None

        # alpha: O3'(i-1) - P(i) - O5'(i) - C5'(i)
        if prev_res:
            O3i_1 = prev_res.atoms.select_atoms("name O3'").positions[0]
            Pi    = res.atoms.select_atoms("name P").positions[0]
            O5i   = res.atoms.select_atoms("name O5'").positions[0]
            C5i   = res.atoms.select_atoms("name C5'").positions[0]
            alpha_val = calc_dihedral(O3i_1, Pi, O5i, C5i)
            torsion_data["alpha"].append(alpha_val)
        else:
            torsion_data["alpha"].append(None)

        # beta: P(i) - O5'(i) - C5'(i) - C4'(i)
        Pi   = res.atoms.select_atoms("name P").positions[0]
        O5i  = res.atoms.select_atoms("name O5'").positions[0]
        C5i  = res.atoms.select_atoms("name C5'").positions[0]
        C4i  = res.atoms.select_atoms("name C4'").positions[0]
        beta_val = calc_dihedral(Pi, O5i, C5i, C4i)
        torsion_data["beta"].append(beta_val)

        # gamma: O5'(i) - C5'(i) - C4'(i) - C3'(i)
        C3i = res.atoms.select_atoms("name C3'").positions[0]
        gamma_val = calc_dihedral(O5i, C5i, C4i, C3i)
        torsion_data["gamma"].append(gamma_val)

        # delta: C5'(i) - C4'(i) - C3'(i) - O3'(i)
        O3i = res.atoms.select_atoms("name O3'").positions[0]
        delta_val = calc_dihedral(C5i, C4i, C3i, O3i)
        torsion_data["delta"].append(delta_val)

        # epsilon: C4'(i) - C3'(i) - O3'(i) - P(i+1)
        epsilon_val = None
        if next_res:
            P_ip1 = next_res.atoms.select_atoms("name P").positions[0]
            epsilon_val = calc_dihedral(C4i, C3i, O3i, P_ip1)
        torsion_data["epsilon"].append(epsilon_val)

        # zeta: C3'(i) - O3'(i) - P(i+1) - O5'(i+1)
        zeta_val = None
        if next_res:
            O5_ip1 = next_res.atoms.select_atoms("name O5'").positions[0]
            P_ip1  = next_res.atoms.select_atoms("name P").positions[0]
            zeta_val = calc_dihedral(C3i, O3i, P_ip1, O5_ip1)
        torsion_data["zeta"].append(zeta_val)

        # chi: glycosidic angle
        # For Purines (A,G): O4' - C1' - N9 - C4
        # For Pyrimidines (U,C): O4' - C1' - N1 - C2
        O4  = res.atoms.select_atoms("name O4'").positions[0]
        C1  = res.atoms.select_atoms("name C1'").positions[0]
        if res.resname in ['A', 'G']:
            N_base = res.atoms.select_atoms("name N9").positions[0]
            C_base = res.atoms.select_atoms("name C4").positions[0]
        else:
            N_base = res.atoms.select_atoms("name N1").positions[0]
            C_base = res.atoms.select_atoms("name C2").positions[0]

        chi_val = calc_dihedral(O4, C1, N_base, C_base)
        torsion_data["chi"].append(chi_val)

    return torsion_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python mdanalysis_torsion_example.py <rna_structure.pdb> [chainID]")
        sys.exit(1)

    pdb_path = sys.argv[1]
    chain_id = sys.argv[2] if len(sys.argv) >= 3 else "A"

    torsions = calculate_rna_torsions_mdanalysis(pdb_path, chain_id)
    for angle_name, values in torsions.items():
        print(f"{angle_name}: {values}")

if __name__ == "__main__":
    main()
