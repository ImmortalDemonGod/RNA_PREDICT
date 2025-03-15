#!/usr/bin/env python

"""
mdanalysis_torsion_example.py

Implements a method for computing RNA torsion angles using MDAnalysis.
Handles both PDB and mmCIF:
 - If .pdb, parse directly with MDAnalysis.
 - If .cif, convert to a temporary PDB via BioPython, then parse that with MDAnalysis.
"""

import sys
import os
import tempfile
import snoop
import numpy as np
import MDAnalysis as mda

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
# We'll implement a real angle-difference scoring function to compare with CSV angles:
import pandas as pd
import numpy as np
from io import StringIO
import re
from mdanalysis_torsion_example import calculate_rna_torsions_mdanalysis as local_calc

def safe_select_atom(res, name):
    """
    Return the first position array if the named atom is present
    in the residue, else return None.
    """
    if res is None:
        return None
    sel = res.atoms.select_atoms(f"name {name}")
    if sel and len(sel.positions) > 0:
        return sel.positions[0]
    return None

def gather_atoms_for_alpha(prev_res, cur_res):
    """
    Gather atoms for alpha torsion:
    O3'(i-1) - P(i) - O5'(i) - C5'(i).
    """
    return (
        safe_select_atom(prev_res, "O3'"),
        safe_select_atom(cur_res,  "P"),
        safe_select_atom(cur_res,  "O5'"),
        safe_select_atom(cur_res,  "C5'")
    )

def gather_atoms_for_beta(cur_res):
    """
    Gather atoms for beta torsion:
    P(i) - O5'(i) - C5'(i) - C4'(i).
    """
    return (
        safe_select_atom(cur_res, "P"),
        safe_select_atom(cur_res, "O5'"),
        safe_select_atom(cur_res, "C5'"),
        safe_select_atom(cur_res, "C4'")
    )

def gather_atoms_for_gamma(cur_res):
    """
    Gather atoms for gamma torsion:
    O5'(i) - C5'(i) - C4'(i) - C3'(i).
    """
    return (
        safe_select_atom(cur_res, "O5'"),
        safe_select_atom(cur_res, "C5'"),
        safe_select_atom(cur_res, "C4'"),
        safe_select_atom(cur_res, "C3'")
    )

def gather_atoms_for_delta(cur_res):
    """
    Gather atoms for delta torsion:
    C5'(i) - C4'(i) - C3'(i) - O3'(i).
    """
    return (
        safe_select_atom(cur_res, "C5'"),
        safe_select_atom(cur_res, "C4'"),
        safe_select_atom(cur_res, "C3'"),
        safe_select_atom(cur_res, "O3'")
    )

def gather_atoms_for_epsilon(cur_res, next_res):
    """
    Gather atoms for epsilon torsion:
    C4'(i) - C3'(i) - O3'(i) - P(i+1).
    """
    return (
        safe_select_atom(cur_res,  "C4'"),
        safe_select_atom(cur_res,  "C3'"),
        safe_select_atom(cur_res,  "O3'"),
        safe_select_atom(next_res, "P")
    )

def gather_atoms_for_zeta(cur_res, next_res):
    """
    Gather atoms for zeta torsion:
    C3'(i) - O3'(i) - P(i+1) - O5'(i+1).
    """
    return (
        safe_select_atom(cur_res,  "C3'"),
        safe_select_atom(cur_res,  "O3'"),
        safe_select_atom(next_res, "P"),
        safe_select_atom(next_res, "O5'")
    )

def gather_atoms_for_chi(cur_res):
    """
    Gather atoms for the glycosidic angle (chi).
    For purines: O4' - C1' - N9 - C4
    For pyrimidines: O4' - C1' - N1 - C2
    """
    O4  = safe_select_atom(cur_res, "O4'")
    C1  = safe_select_atom(cur_res, "C1'")
    resname = cur_res.resname.strip().upper()
    if resname.startswith("A") or resname.startswith("G"):
        N_base = safe_select_atom(cur_res, "N9")
        C_base = safe_select_atom(cur_res, "C4")
    else:
        N_base = safe_select_atom(cur_res, "N1")
        C_base = safe_select_atom(cur_res, "C2")
    return (O4, C1, N_base, C_base)

def compute_dihedral_or_nan(atom_tuple):
    """
    If any of the four atoms is None, return NaN.
    Otherwise, compute the dihedral angle using calc_dihedral.
    """
    if any(a is None for a in atom_tuple):
        return np.nan
    p1, p2, p3, p4 = atom_tuple
    angle_deg = calc_dihedral(p1, p2, p3, p4)
    if angle_deg is None:
        return np.nan
    return np.float64(angle_deg)

def calc_dihedral(p1, p2, p3, p4):
    """
    Calculate the dihedral angle (in degrees) for four 3D points
    p1, p2, p3, p4 (each a NumPy array [x,y,z]).
    If cross products are too small, return None.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-12 or norm_n2 < 1e-12:
        return None

    cos_angle = np.dot(n1, n2) / (norm_n1 * norm_n2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp
    sign = np.dot(b2, np.cross(n1, n2))
    phi = np.arccos(cos_angle)
    if sign < 0.0:
        phi = -phi
    return np.degrees(phi)


def convert_cif_to_pdb(cif_file):
    """
    Convert an mmCIF file to a temporary PDB file using BioPython.
    Returns the path to the temporary PDB file.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("mmcif_structure", cif_file)

    # Write to a temp PDB
    tmp_handle = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp_handle.close()  # We'll pass its name to PDBIO
    pdb_path = tmp_handle.name

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
    return pdb_path


def calculate_rna_torsions_mdanalysis(pdb_file, chain_id="A", fallback=False):
    """
    Calculate backbone and glycosidic torsion angles (alpha, beta, gamma,
    delta, epsilon, zeta, chi) for an RNA chain using MDAnalysis.
    For .cif files, we convert them to PDB with BioPython first.

    This implementation uses a safe "gather atoms, then compute dihedral" approach,
    ensuring that each torsion angle is computed independently, so we don't rely
    on partially assigned local variables.
    """
    from MDAnalysis import Universe

    # 1) If it's a .cif file, convert to .pdb
    _, ext = os.path.splitext(pdb_file)
    ext = ext.lower()

    using_temp = False
    temp_pdb_path = None
    if ext == ".cif":
        using_temp = True
        temp_pdb_path = convert_cif_to_pdb(pdb_file)
        mdanalysis_file = temp_pdb_path
    else:
        mdanalysis_file = pdb_file

    # 2) Create the Universe
    try:
        u = Universe(mdanalysis_file)
    finally:
        pass

    # Debug: Print all segments and their chain IDs
    print("=== DEBUG: Segments in this Universe ===")
    for seg in u.segments:
        print(f"  Segment segid={seg.segid}, n_res={len(seg.residues)}")

    # Debug: Print any chain info from 'nucleic' selection, in case there's a label mismatch
    all_nucleic = u.select_atoms("nucleic")
    unique_segids = set(a.segment.segid for a in all_nucleic.atoms)
    print(f"=== DEBUG: Found segids in 'nucleic': {unique_segids}")

    # 3) Attempt chain selection
    print("Segments:", u.segments)
    print("Residues:", u.residues)

    chain = u.select_atoms(f"(segid {chain_id}) or (chainID {chain_id})")
    print(f"Selecting chain with chain_id='{chain_id}'... Found {len(chain)} atoms.")

    if len(chain) == 0:
        # Attempt to autodetect among all segids found in nucleic
        candidate_segids = set(a.segment.segid for a in u.select_atoms("nucleic").atoms)
        print(f"No atoms found for chainID={chain_id}. Trying all candidate segids: {candidate_segids}")

        best_seg = None
        best_chain = None
        best_score = float('inf')



        def compute_angle_score(test_chain, csv_file):
            # 1) Load CSV angles
            lines_csv = []
            with open(csv_file, "r") as f_in:
                for line in f_in:
                    if not line.strip() or '====' in line:
                        continue
                    if line.startswith("index_chain") or re.match(r'^\\d', line.strip()):
                        lines_csv.append(line)
            df_str = "".join(lines_csv)
            df = pd.read_csv(StringIO(df_str), sep=",")
            alpha_csv   = df["alpha"].to_numpy(dtype=float)
            beta_csv    = df["beta"].to_numpy(dtype=float)
            gamma_csv   = df["gamma"].to_numpy(dtype=float)
            delta_csv   = df["delta"].to_numpy(dtype=float)
            epsilon_csv = df["epsilon"].to_numpy(dtype=float)
            zeta_csv    = df["zeta"].to_numpy(dtype=float)
            chi_csv     = df["chi"].to_numpy(dtype=float)

            # 2) Compute new angles from test_chain
            # We'll create a Universe with only the test_chain: or just pass to local_calc
            # But local_calc expects a path, so we do an inline approach:
            # Or we can call local_calc on the entire Universe, but restricting to test_chain's residues.
            # For simplicity, let's do a partial approach:
            # We'll rely on your existing logic to gather angles from 'test_chain'.
            # We'll copy/paste the gather logic from within the same file or from custom script.

            # For demonstration, let's call local_calc on a temporary file approach:
            # This might be tricky if we only have an in-memory selection. Alternatively, we can adapt the logic directly.
            # Here we show a pseudo approach:
            new_angles = {
                "alpha": [],
                "beta": [],
                "gamma": [],
                "delta": [],
                "epsilon": [],
                "zeta": [],
                "chi": []
            }
            # We'll gather real angles from the 'test_chain' the same way the main code does:
            all_res = test_chain.residues
            for i, res in enumerate(all_res):
                prev_res = all_res[i - 1] if i > 0 else None
                next_res = all_res[i + 1] if i < len(all_res) - 1 else None
                # we can call gather_atoms, compute_dihedral, etc.
                # For brevity, let's do alpha only. In reality, replicate for all angles:
                from .mdanalysis_torsion_example import (
                    gather_atoms_for_alpha,
                    gather_atoms_for_beta,
                    gather_atoms_for_gamma,
                    gather_atoms_for_delta,
                    gather_atoms_for_epsilon,
                    gather_atoms_for_zeta,
                    gather_atoms_for_chi,
                    compute_dihedral_or_nan
                )
                if prev_res:
                    alpha_tuple = gather_atoms_for_alpha(prev_res, res)
                    alpha_val = compute_dihedral_or_nan(alpha_tuple)
                else:
                    alpha_val = np.nan
                new_angles["alpha"].append(alpha_val)

                beta_tuple = gather_atoms_for_beta(res)
                beta_val = compute_dihedral_or_nan(beta_tuple)
                new_angles["beta"].append(beta_val)

                gamma_tuple = gather_atoms_for_gamma(res)
                gamma_val = compute_dihedral_or_nan(gamma_tuple)
                new_angles["gamma"].append(gamma_val)

                delta_tuple = gather_atoms_for_delta(res)
                delta_val = compute_dihedral_or_nan(delta_tuple)
                new_angles["delta"].append(delta_val)

                if next_res:
                    eps_tuple = gather_atoms_for_epsilon(res, next_res)
                    eps_val = compute_dihedral_or_nan(eps_tuple)
                else:
                    eps_val = np.nan
                new_angles["epsilon"].append(eps_val)

                if next_res:
                    zeta_tuple = gather_atoms_for_zeta(res, next_res)
                    zeta_val = compute_dihedral_or_nan(zeta_tuple)
                else:
                    zeta_val = np.nan
                new_angles["zeta"].append(zeta_val)

                chi_tuple = gather_atoms_for_chi(res)
                chi_val = compute_dihedral_or_nan(chi_tuple)
                new_angles["chi"].append(chi_val)

            alpha_new   = np.array(new_angles["alpha"],   dtype=float)
            beta_new    = np.array(new_angles["beta"],    dtype=float)
            gamma_new   = np.array(new_angles["gamma"],   dtype=float)
            delta_new   = np.array(new_angles["delta"],   dtype=float)
            epsilon_new = np.array(new_angles["epsilon"], dtype=float)
            zeta_new    = np.array(new_angles["zeta"],    dtype=float)
            chi_new     = np.array(new_angles["chi"],     dtype=float)

            # 3) Align/truncate
            n_csv = len(alpha_csv)
            n_new = len(alpha_new)
            n_min = min(n_csv, n_new)
            # 4) compute average absolute difference
            diff_alpha   = np.nanmean(np.abs(alpha_new[:n_min]   - alpha_csv[:n_min]))
            diff_beta    = np.nanmean(np.abs(beta_new[:n_min]    - beta_csv[:n_min]))
            diff_gamma   = np.nanmean(np.abs(gamma_new[:n_min]   - gamma_csv[:n_min]))
            diff_delta   = np.nanmean(np.abs(delta_new[:n_min]   - delta_csv[:n_min]))
            diff_eps     = np.nanmean(np.abs(epsilon_new[:n_min] - epsilon_csv[:n_min]))
            diff_zeta    = np.nanmean(np.abs(zeta_new[:n_min]    - zeta_csv[:n_min]))
            diff_chi     = np.nanmean(np.abs(chi_new[:n_min]     - chi_csv[:n_min]))

            # Combine them:
            overall_score = np.mean([diff_alpha, diff_beta, diff_gamma, diff_delta, diff_eps, diff_zeta, diff_chi])
            return overall_score

        for seg in candidate_segids:
            test_chain = u.select_atoms(f"(segid {seg}) and nucleic")
            if len(test_chain.residues) < 1:
                continue
            score = compute_angle_score(test_chain, csv_file)
            if score < best_score:
                best_score = score
                best_seg = seg
                best_chain = test_chain

        if best_chain is not None:
            print(f"Auto-selected chain segid='{best_seg}' with score={best_score}")
            chain = best_chain
        else:
            if fallback:
                print(f"All segids tested but none chosen. Falling back to all nucleic.")
                chain = u.select_atoms('nucleic')
            else:
                raise ValueError(f"No valid chain found for chainID='{chain_id}' and autodetect failed. Check your PDB/cif labeling.")

    # Extra debug: show residue numbering in the chain
    print("=== DEBUG: Residue numbering in selected chain ===")
    for r in chain.residues:
        print(f"   Residue {r.resname}, resid={r.resid}, segid={r.segid}")

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
        prev_res = residues[i - 1] if i > 0 else None
        next_res = residues[i + 1] if i < len(residues) - 1 else None

        # alpha: gather from prev & cur
        if prev_res:
            alpha_tuple = gather_atoms_for_alpha(prev_res, res)
            alpha_val = compute_dihedral_or_nan(alpha_tuple)
        else:
            alpha_val = np.nan
        torsion_data["alpha"].append(alpha_val)

        # beta
        beta_tuple = gather_atoms_for_beta(res)
        beta_val = compute_dihedral_or_nan(beta_tuple)
        torsion_data["beta"].append(beta_val)

        # gamma
        gamma_tuple = gather_atoms_for_gamma(res)
        gamma_val = compute_dihedral_or_nan(gamma_tuple)
        torsion_data["gamma"].append(gamma_val)

        # delta
        delta_tuple = gather_atoms_for_delta(res)
        delta_val = compute_dihedral_or_nan(delta_tuple)
        torsion_data["delta"].append(delta_val)

        # epsilon
        if next_res:
            epsilon_tuple = gather_atoms_for_epsilon(res, next_res)
            epsilon_val = compute_dihedral_or_nan(epsilon_tuple)
        else:
            epsilon_val = np.nan
        torsion_data["epsilon"].append(epsilon_val)

        # zeta
        if next_res:
            zeta_tuple = gather_atoms_for_zeta(res, next_res)
            zeta_val = compute_dihedral_or_nan(zeta_tuple)
        else:
            zeta_val = np.nan
        torsion_data["zeta"].append(zeta_val)

        # chi: gather from current residue alone
        chi_tuple = gather_atoms_for_chi(res)
        chi_val = compute_dihedral_or_nan(chi_tuple)
        torsion_data["chi"].append(chi_val)

    # Clean up temporary file if used
    if using_temp and temp_pdb_path is not None and os.path.exists(temp_pdb_path):
        os.remove(temp_pdb_path)

    return torsion_data


def main():
    if len(sys.argv) < 2:
        print("Usage: python mdanalysis_torsion_example.py <rna_structure.[pdb|cif]> [chainID]")
        sys.exit(1)

    pdb_path = sys.argv[1]
    chain_id = sys.argv[2] if len(sys.argv) >= 3 else "A"

    torsions = calculate_rna_torsions_mdanalysis(pdb_path, chain_id=chain_id, fallback=True)
    for angle_name, values in torsions.items():
        print(f"{angle_name}: {values}")

if __name__ == "__main__":
    main()