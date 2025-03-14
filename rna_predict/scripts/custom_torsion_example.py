#!/usr/bin/env python

"""
mdanalysis_torsion_example.py

Demonstrates how to calculate torsion angles (e.g. α, β, γ, δ, ε, ζ, χ) 
in an RNA structure using MDAnalysis.
"""

import sys
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals
import numpy as np

def compute_dihedral(atom_selection):
    """
    Helper function to compute a single dihedral using MDAnalysis's calc_dihedrals.
    Expects a list of 4 MDAnalysis atom objects.
    Returns the angle in degrees or None if any atom is missing.
    """
    if None in atom_selection or len(atom_selection) < 4:
        return None
    coords = np.array([atm.position for atm in atom_selection]).reshape(1, 4, 3)
    radians = calc_dihedrals(coords)[0]
    return np.degrees(radians)

def calculate_rna_torsions_mdanalysis(pdb_file, chain_id="A"):
    """
    Calculates backbone (alpha, beta, gamma, delta, epsilon, zeta)
    and glycosidic (chi) torsion angles for the specified RNA chain.
    Args:
        pdb_file (str): Path to the PDB/mmCIF file containing the RNA structure.
        chain_id (str): Which chain to analyze, default 'A'.
    Returns:
        dict: { 'alpha': [...], 'beta': [...], 'gamma': [...], 
                'delta': [...], 'epsilon': [...], 'zeta': [...], 'chi': [...] }
              Each entry is a list of angles (or None) in residue order.
    """
    u = mda.Universe(pdb_file)
    # Select chain by known identifiers (adjust as needed)
    rna_chain = u.select_atoms(f"segid {chain_id} or chainid {chain_id} or chain {chain_id}")

    torsion_data = {
        "alpha": [],
        "beta": [],
        "gamma": [],
        "delta": [],
        "epsilon": [],
        "zeta": [],
        "chi":  []
    }

    residues = rna_chain.residues
    n_res = len(residues)
    for i, res in enumerate(residues):
        prev_res = residues[i - 1] if i > 0 else None
        next_res = residues[i + 1] if i < n_res - 1 else None
        
        # α: O3'(i-1) - P(i) - O5'(i) - C5'(i)
        if prev_res:
            alpha_val = compute_dihedral([
                prev_res.atoms.select_atoms("name O3'")[0] if prev_res.atoms.select_atoms("name O3'") else None,
                res.atoms.select_atoms("name P")[0]       if res.atoms.select_atoms("name P") else None,
                res.atoms.select_atoms("name O5'")[0]     if res.atoms.select_atoms("name O5'") else None,
                res.atoms.select_atoms("name C5'")[0]     if res.atoms.select_atoms("name C5'") else None
            ])
        else:
            alpha_val = None
        torsion_data["alpha"].append(alpha_val)

        # β: P(i) - O5'(i) - C5'(i) - C4'(i)
        beta_val = compute_dihedral([
            res.atoms.select_atoms("name P")[0]     if res.atoms.select_atoms("name P") else None,
            res.atoms.select_atoms("name O5'")[0]   if res.atoms.select_atoms("name O5'") else None,
            res.atoms.select_atoms("name C5'")[0]   if res.atoms.select_atoms("name C5'") else None,
            res.atoms.select_atoms("name C4'")[0]   if res.atoms.select_atoms("name C4'") else None
        ])
        torsion_data["beta"].append(beta_val)

        # γ: O5'(i) - C5'(i) - C4'(i) - C3'(i)
        gamma_val = compute_dihedral([
            res.atoms.select_atoms("name O5'")[0] if res.atoms.select_atoms("name O5'") else None,
            res.atoms.select_atoms("name C5'")[0] if res.atoms.select_atoms("name C5'") else None,
            res.atoms.select_atoms("name C4'")[0] if res.atoms.select_atoms("name C4'") else None,
            res.atoms.select_atoms("name C3'")[0] if res.atoms.select_atoms("name C3'") else None
        ])
        torsion_data["gamma"].append(gamma_val)

        # δ: C5'(i) - C4'(i) - C3'(i) - O3'(i)
        delta_val = compute_dihedral([
            res.atoms.select_atoms("name C5'")[0] if res.atoms.select_atoms("name C5'") else None,
            res.atoms.select_atoms("name C4'")[0] if res.atoms.select_atoms("name C4'") else None,
            res.atoms.select_atoms("name C3'")[0] if res.atoms.select_atoms("name C3'") else None,
            res.atoms.select_atoms("name O3'")[0] if res.atoms.select_atoms("name O3'") else None
        ])
        torsion_data["delta"].append(delta_val)

        # ε: C4'(i) - C3'(i) - O3'(i) - P(i+1)
        if next_res:
            epsilon_val = compute_dihedral([
                res.atoms.select_atoms("name C4'")[0] if res.atoms.select_atoms("name C4'") else None,
                res.atoms.select_atoms("name C3'")[0] if res.atoms.select_atoms("name C3'") else None,
                res.atoms.select_atoms("name O3'")[0] if res.atoms.select_atoms("name O3'") else None,
                next_res.atoms.select_atoms("name P")[0] if next_res.atoms.select_atoms("name P") else None
            ])
        else:
            epsilon_val = None
        torsion_data["epsilon"].append(epsilon_val)

        # ζ: C3'(i) - O3'(i) - P(i+1) - O5'(i+1)
        if next_res:
            zeta_val = compute_dihedral([
                res.atoms.select_atoms("name C3'")[0] if res.atoms.select_atoms("name C3'") else None,
                res.atoms.select_atoms("name O3'")[0] if res.atoms.select_atoms("name O3'") else None,
                next_res.atoms.select_atoms("name P")[0]   if next_res.atoms.select_atoms("name P") else None,
                next_res.atoms.select_atoms("name O5'")[0] if next_res.atoms.select_atoms("name O5'") else None
            ])
        else:
            zeta_val = None
        torsion_data["zeta"].append(zeta_val)

        # χ: glycosidic angle
        # Purines (A,G): O4' - C1' - N9 - C4
        # Pyrimidines (U,C): O4' - C1' - N1 - C2
        resname = res.resname.strip().upper()
        O4  = res.atoms.select_atoms("name O4'")[0] if res.atoms.select_atoms("name O4'") else None
        C1  = res.atoms.select_atoms("name C1'")[0] if res.atoms.select_atoms("name C1'") else None
        if resname.startswith("A") or resname.startswith("G"):
            N_base = res.atoms.select_atoms("name N9")[0] if res.atoms.select_atoms("name N9") else None
            C_base = res.atoms.select_atoms("name C4")[0] if res.atoms.select_atoms("name C4") else None
        else:
            N_base = res.atoms.select_atoms("name N1")[0] if res.atoms.select_atoms("name N1") else None
            C_base = res.atoms.select_atoms("name C2")[0] if res.atoms.select_atoms("name C2") else None

        chi_val = None
        if O4 and C1 and N_base and C_base:
            chi_val = compute_dihedral([O4, C1, N_base, C_base])
        torsion_data["chi"].append(chi_val)

    return torsion_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python mdanalysis_torsion_example.py <rna_structure.pdb> [chainID]")
        sys.exit(1)

    pdb_path = sys.argv[1]
    chain_id = sys.argv[2] if len(sys.argv) >= 3 else "A"

    angles = calculate_rna_torsions_mdanalysis(pdb_path, chain_id=chain_id)
    for angle_name, values in angles.items():
        print(f"{angle_name}: {values}")

if __name__ == "__main__":
    main()
