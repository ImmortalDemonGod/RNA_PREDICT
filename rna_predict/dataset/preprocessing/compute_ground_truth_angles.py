#!/usr/bin/env python
"""
Script to compute ground truth torsion angles from RNA structure files (PDB/CIF)
using a modular backend (MDAnalysis or DSSR) and save them as PyTorch tensors.

This script:
1. Takes a directory of PDB/CIF files as input
2. Extracts torsion angles using the selected backend (default: MDAnalysis)
3. Converts the angles to radians and formats them as tensors
4. Saves the tensors as .pt files for use in training

Usage:
    uv run rna_predict/dataset/preprocessing/compute_ground_truth_angles.py --input_dir /path/to/pdb_files --output_dir /path/to/output --chain_id A --backend mdanalysis
"""
import argparse
import os
import torch
from rna_predict.dataset.preprocessing.angles import extract_rna_torsions

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ground truth torsion angles from RNA structure files using a modular backend"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PDB/CIF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .pt files")
    parser.add_argument("--chain_id", type=str, default="A", help="Chain ID to extract (default: A)")
    parser.add_argument("--backend", type=str, default="mdanalysis", choices=["mdanalysis", "dssr"], help="Backend to use for extraction")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[DEBUG] Parsed args: {args}")
    os.makedirs(args.output_dir, exist_ok=True)
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith((".pdb", ".cif")):
            continue
        in_path = os.path.join(args.input_dir, fname)
        print(f"[DEBUG] Processing: {in_path} (chain: {args.chain_id}, backend: {args.backend})")
        angles = extract_rna_torsions(in_path, chain_id=args.chain_id, backend=args.backend)
        if angles is not None:
            out_path = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_{args.chain_id}_angles.pt")
            torch.save(torch.from_numpy(angles), out_path)
            print(f"Saved: {out_path}")
        else:
            print(f"Failed: {in_path}")

if __name__ == "__main__":
    main()