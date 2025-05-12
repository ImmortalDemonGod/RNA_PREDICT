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
    """
    Parses command-line arguments for RNA torsion angle extraction.
    
    Returns:
        argparse.Namespace: Parsed arguments including input/output directories, chain ID,
        backend selection, and angle set choice.
    """
    parser = argparse.ArgumentParser(
        description="Compute ground truth torsion angles from RNA structure files using a modular backend"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PDB/CIF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .pt files")
    parser.add_argument("--chain_id", type=str, default="A", help="Chain ID to extract (default: A)")
    parser.add_argument("--backend", type=str, default="mdanalysis", choices=["mdanalysis", "dssr"], help="Backend to use for extraction")
    parser.add_argument("--angle_set", type=str, default="canonical", choices=["canonical", "full"], help="Which angle set to extract: 'canonical' (7) or 'full' (14)")
    return parser.parse_args()

def main():
    """
    Processes RNA structure files to extract and save ground truth torsion angles.
    
    Iterates over PDB or CIF files in the input directory, extracts RNA torsion angles
    for a specified chain and angle set using the selected backend, and saves the
    results as PyTorch tensors in the output directory. Files that fail extraction
    are reported.
    """
    args = parse_args()
    print(f"[DEBUG] Parsed args: {args}")
    # --- Hydra config integration ---
    try:
        from hydra import initialize, compose
        # Use absolute path as per project best practice
        hydra_conf_path = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"
        with initialize(config_path=hydra_conf_path, job_name="compute_ground_truth_angles"):
            cfg = compose(config_name="default")
        config_chain_id = cfg.data.chain_id if hasattr(cfg.data, "chain_id") else None
    except Exception as e:
        print(f"[WARNING] Could not load Hydra config: {e}")
        cfg = None
        config_chain_id = None
    # --- Backend selection logic ---
    cli_backend = args.backend
    cfg_backend = getattr(cfg, 'extraction_backend', None)
    if cfg_backend:
        if cli_backend != cfg_backend:
            print(f"[WARNING] CLI --backend ({cli_backend}) differs from config extraction_backend ({cfg_backend}). Using config value.")
        selected_backend = cfg_backend
    else:
        selected_backend = cli_backend
    print(f"[DEBUG] Using extraction backend: {selected_backend}")
    # --- Chain selection logic ---
    cli_chain_id = args.chain_id
    if config_chain_id is not None:
        if cli_chain_id != config_chain_id:
            print(f"[WARNING] CLI --chain_id ({cli_chain_id}) differs from config chain_id ({config_chain_id}). Using config value.")
        selected_chain_id = config_chain_id
    else:
        selected_chain_id = cli_chain_id
        print(f"[INFO] Using CLI --chain_id: {selected_chain_id}")
    os.makedirs(args.output_dir, exist_ok=True)
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith((".pdb", ".cif")):
            continue
        in_path = os.path.join(args.input_dir, fname)
        # Print available chain IDs in structure
        try:
            from Bio.PDB import MMCIFParser, PDBParser
            import os as _os
            ext = _os.path.splitext(in_path)[1].lower()
            if ext == ".cif":
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("rna", in_path)
            elif ext == ".pdb":
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("rna", in_path)
            else:
                structure = None
            if structure is not None:
                chain_ids = [chain.id for model in structure for chain in model]
                print(f"[DEBUG][compute_ground_truth_angles] Available chain IDs in {in_path}: {chain_ids}")
        except Exception as e:
            print(f"[DEBUG][compute_ground_truth_angles] Could not parse structure to list chain IDs: {e}")
        print(f"[DEBUG] Processing: {in_path} (chain: {selected_chain_id}, backend: {selected_backend}, angle_set: {args.angle_set})")
        angles = extract_rna_torsions(in_path, chain_id=selected_chain_id, backend=selected_backend, angle_set=args.angle_set)
        if angles is not None:
            out_path = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_{selected_chain_id}_angles.pt")
            torch.save(torch.from_numpy(angles), out_path)
            print(f"Saved: {out_path}")
        else:
            print(f"Failed: {in_path}")

if __name__ == "__main__":
    main()