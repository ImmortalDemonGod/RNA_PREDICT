"""
RNA backbone extraction utilities.
Extracts canonical backbone atom coordinates, checks completeness, and computes internal geometry (bond lengths & angles).
"""

import os
import glob
import numpy as np
import logging

logger = logging.getLogger("rna_predict.utils.rna_backbone_extraction")

CANONICAL_BACKBONE_ORDER = [
    "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"
]

# --- Extraction functions ---
def normalize_atom_name(atom):
    # Normalize atom names for PDB/mmCIF
    return atom.replace('*', "'").strip()

def extract_pdb_backbone_coords(pdb_path, chain_select=None, residue_select=None):
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom = normalize_atom_name(line[12:16])
            resn = line[17:20].strip()
            chain = line[21].strip()
            resi = int(line[22:26])
            if chain_select and chain != chain_select:
                continue
            if residue_select and resi != residue_select:
                continue
            if atom in CANONICAL_BACKBONE_ORDER:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append({'atom': atom, 'coords': [x, y, z]})
    # Sort and check completeness
    coords_sorted = sorted(coords, key=lambda c: CANONICAL_BACKBONE_ORDER.index(c['atom']))
    found = {c['atom'] for c in coords_sorted}
    missing = set(CANONICAL_BACKBONE_ORDER) - found
    if missing:
        logger.warning(f"{os.path.basename(pdb_path)} missing backbone atoms: {missing}")
    return coords_sorted

def extract_cif_backbone_coords(cif_path, chain_select=None, residue_select=None):
    coords = []
    atom_lines = []
    with open(cif_path, 'r') as f:
        lines = f.readlines()
        in_atom = False
        for line in lines:
            if line.strip().startswith('loop_') and '_atom_site.' in lines[lines.index(line)+1]:
                in_atom = True
                continue
            if in_atom and (line.startswith('_') or line.strip() == ''):
                in_atom = False
            if in_atom:
                atom_lines.append(line.strip())
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 11:
            continue
        atom = normalize_atom_name(fields[2])
        chain = fields[4]
        resi = int(fields[5])
        if chain_select and chain != chain_select:
            continue
        if residue_select and resi != residue_select:
            continue
        if atom in CANONICAL_BACKBONE_ORDER:
            x = float(fields[6])
            y = float(fields[7])
            z = float(fields[8])
            coords.append({'atom': atom, 'coords': [x, y, z]})
    coords_sorted = sorted(coords, key=lambda c: CANONICAL_BACKBONE_ORDER.index(c['atom']))
    found = {c['atom'] for c in coords_sorted}
    missing = set(CANONICAL_BACKBONE_ORDER) - found
    if missing:
        logger.warning(f"{os.path.basename(cif_path)} missing backbone atoms: {missing}")
    return coords_sorted

# --- Geometry functions ---
def compute_bond_lengths(coords):
    bond_lengths = []
    for i in range(len(coords) - 1):
        a = np.array(coords[i]['coords'])
        b = np.array(coords[i+1]['coords'])
        bond_len = np.linalg.norm(a - b)
        bond_lengths.append((f"{coords[i]['atom']}-{coords[i+1]['atom']}", round(bond_len, 3)))
    return bond_lengths

def compute_bond_angles(coords):
    bond_angles = []
    for i in range(len(coords) - 2):
        a = np.array(coords[i]['coords'])
        b = np.array(coords[i+1]['coords'])
        c = np.array(coords[i+2]['coords'])
        ba = a - b
        bc = c - b
        ba /= np.linalg.norm(ba)
        bc /= np.linalg.norm(bc)
        cos_angle = np.clip(np.dot(ba, bc), -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        bond_angles.append((f"{coords[i]['atom']}-{coords[i+1]['atom']}-{coords[i+2]['atom']}", round(angle, 2)))
    return bond_angles

# --- Main CLI wrapper ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract canonical RNA backbone coordinates and internal geometry.")
    parser.add_argument('--examples_dir', type=str, required=True, help='Directory with PDB and CIF files to process')
    args = parser.parse_args()
    examples_dir = args.examples_dir

    print("# Canonical backbone coordinates extracted from examples directory\n")
    for pdb_file in glob.glob(os.path.join(examples_dir, '*.pdb')):
        coords = extract_pdb_backbone_coords(pdb_file)
        print(f"PDB: {os.path.basename(pdb_file)}")
        for c in coords:
            print(f"  {c['atom']}: {np.round(c['coords'], 3).tolist()}")
        if len(coords) >= 2:
            print("  Bond lengths:")
            for name, val in compute_bond_lengths(coords):
                print(f"    {name}: {val} Å")
        if len(coords) >= 3:
            print("  Bond angles:")
            for name, val in compute_bond_angles(coords):
                print(f"    {name}: {val}°")
        print()
    for cif_file in glob.glob(os.path.join(examples_dir, '*.cif')):
        coords = extract_cif_backbone_coords(cif_file)
        print(f"CIF: {os.path.basename(cif_file)}")
        for c in coords:
            print(f"  {c['atom']}: {np.round(c['coords'], 3).tolist()}")
        if len(coords) >= 2:
            print("  Bond lengths:")
            for name, val in compute_bond_lengths(coords):
                print(f"    {name}: {val} Å")
        if len(coords) >= 3:
            print("  Bond angles:")
            for name, val in compute_bond_angles(coords):
                print(f"    {name}: {val}°")
        print()

if __name__ == "__main__":
    main()
