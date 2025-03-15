#!/usr/bin/env python

"""
compare_precomputed_torsions.py

Example script to:
  1) Read a CSV/TSV containing columns: alpha, beta, gamma, delta, epsilon, zeta, chi
  2) Parse an mmCIF structure with MDAnalysis
  3) Recompute those torsions from the .cif file (using the same definitions
     as custom_torsion_example.py, for instance)
  4) Compare the two sets of angles and print the differences

Usage:
  python compare_precomputed_torsions.py my_torsions.csv my_structure.cif [chainID]

Requires: pandas (for CSV) and MDAnalysis
"""

import sys
import numpy as np
import pandas as pd
import MDAnalysis as mda
from mdanalysis_torsion_example import calculate_rna_torsions_mdanalysis  # or custom_torsion_example

def compare_torsions(csv_file, cif_file, chain_id="A"):
    """
    Compare precomputed torsion angles in csv_file with newly computed angles from cif_file.
    
    CSV columns expected: 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi'
    (Plus possibly other columns.)
    
    The CIF is read by MDAnalysis, then we call calculate_rna_torsions_mdanalysis
    to get angles as a dictionary of lists in residue order.

    We'll print out differences for each angle name.
    """
    # 1) Read the data
    # The user's file “1a9n_1_Q” is basically a CSV with a header line containing column names
    # "index_chain,old_nt_resnum,nt_position,nt_name,nt_code,nt_align_code,...,alpha,beta,gamma,delta,epsilon,zeta,chi..."
    # We can specify them or let pandas read it directly.
    #
    # We'll skip any leading lines that don't start with a digit or "index_chain".
    # We'll read them as CSV with comma as separator.
    
    # Option 1: Attempt to read automatically if the file has the header line.
    # If there's any weird lines, let's skip them:
    import re

    # We'll gather lines that have a consistent count of commas
    # and do a quick guess to skip lines with '====' or directory references.
    lines = []
    with open(csv_file, "r") as f_in:
        for line in f_in:
            # skip blank lines or lines that have '===='
            if not line.strip() or '====' in line:
                continue
            # if the line starts with "index_chain" or a digit, assume it's data
            if line.startswith("index_chain") or re.match(r'^\d', line.strip()):
                lines.append(line)

    # Now parse them with pandas
    from io import StringIO
    content_str = "".join(lines)
    df = pd.read_csv(StringIO(content_str), sep=",")
    # We now have a DataFrame with a bunch of columns, including alpha,beta,gamma...
    # Convert them to numpy arrays for easy subtraction
    alpha_csv   = df["alpha"].to_numpy(dtype=float)
    beta_csv    = df["beta"].to_numpy(dtype=float)
    gamma_csv   = df["gamma"].to_numpy(dtype=float)
    delta_csv   = df["delta"].to_numpy(dtype=float)
    epsilon_csv = df["epsilon"].to_numpy(dtype=float)
    zeta_csv    = df["zeta"].to_numpy(dtype=float)
    chi_csv     = df["chi"].to_numpy(dtype=float)

    # 2) Compute angles from the CIF
    # We'll rely on the fallback approach to handle chain selection
    angles_mdanalysis = calculate_rna_torsions_mdanalysis(cif_file, chain_id, fallback=True, csv_file=csv_file)
    # angles_mdanalysis is a dict of lists: angles_mdanalysis["alpha"] -> list of floats
    alpha_new   = np.array(angles_mdanalysis["alpha"],   dtype=float)
    beta_new    = np.array(angles_mdanalysis["beta"],    dtype=float)
    gamma_new   = np.array(angles_mdanalysis["gamma"],   dtype=float)
    delta_new   = np.array(angles_mdanalysis["delta"],   dtype=float)
    epsilon_new = np.array(angles_mdanalysis["epsilon"], dtype=float)
    zeta_new    = np.array(angles_mdanalysis["zeta"],    dtype=float)
    chi_new     = np.array(angles_mdanalysis["chi"],     dtype=float)

    # For a valid comparison, the residue counts must match.
    # If your CSV has a different residue indexing approach,
    # you may need to align them by residue name or ID. We'll do naive same-length compare:
    n_csv = len(alpha_csv)
    n_new = len(alpha_new)
    n_min = min(n_csv, n_new)
    if n_csv != n_new:
        print(f"Warning: CSV has {n_csv} residues, CIF gave {n_new} angles. Truncating to {n_min} for comparison.")
    
    # 3) Differences
    def print_stats(name, arr_csv, arr_new):
        # Subset to n_min
        arr_diff = arr_new[:n_min] - arr_csv[:n_min]
        mean_diff = np.nanmean(arr_diff)
        std_diff  = np.nanstd(arr_diff)
        print(f"{name}: mean diff={mean_diff:.2f} deg, std={std_diff:.2f}, range=({np.nanmin(arr_diff):.2f},{np.nanmax(arr_diff):.2f})")

    print_stats("alpha",   alpha_csv,   alpha_new)
    print_stats("beta",    beta_csv,    beta_new)
    print_stats("gamma",   gamma_csv,   gamma_new)
    print_stats("delta",   delta_csv,   delta_new)
    print_stats("epsilon", epsilon_csv, epsilon_new)
    print_stats("zeta",    zeta_csv,    zeta_new)
    print_stats("chi",     chi_csv,     chi_new)

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_precomputed_torsions.py <my_torsions.csv> <my_structure.cif> [chainID]")
        sys.exit(1)

    csv_file = sys.argv[1]
    cif_file = sys.argv[2]
    chain_id = sys.argv[3] if len(sys.argv) >= 4 else "A"

    compare_torsions(csv_file, cif_file, chain_id=chain_id)

if __name__ == "__main__":
    main()