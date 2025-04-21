#!/usr/bin/env python3
"""
Debug script for MP-NeRF NaN issue in test_interface_mpnerf_nan.py
"""

import torch
from omegaconf import OmegaConf
from rna_predict.interface import RNAPredictor

def debug_predict_submission():
    """Debug the predict_submission method with MP-NeRF"""
    # Create the same configuration as in the test
    device = "cpu"
    cfg = OmegaConf.create({
        "device": device,
        "model": {
            "stageB": {
                "torsion_bert": {
                    "model_name_or_path": "sayby/rna_torsionbert",
                    "device": device,
                    "angle_mode": "degrees",
                    "num_angles": 7,
                    "max_length": 512
                }
            },
            "stageC": {
                "enabled": True,
                "method": "mp_nerf",
                "device": device,
                "do_ring_closure": False,
                "place_bases": True,
                "sugar_pucker": "C3'-endo",
                "angle_representation": "degrees",
                "use_metadata": False,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": True,
                "debug_logging": True
            }
        }
    })

    # Initialize the predictor
    predictor = RNAPredictor(cfg)
    
    # Test sequence
    sequence = "ACGUA"
    
    # Debug predict_3d_structure first (which works according to the test)
    print(f"\n[DEBUG] Testing predict_3d_structure for sequence: '{sequence}'")
    result_dict = predictor.predict_3d_structure(sequence)
    coords = result_dict.get("coords")
    print(f"[DEBUG] coords.shape = {coords.shape}")
    print(f"[DEBUG] coords.dim() = {coords.dim()}")
    print(f"[DEBUG] Has NaNs: {torch.isnan(coords).any().item()}")
    
    # Debug predict_submission (which fails according to the test)
    print(f"\n[DEBUG] Testing predict_submission for sequence: '{sequence}'")
    submission_df = predictor.predict_submission(sequence, prediction_repeats=1, residue_atom_choice=0)
    print(f"[DEBUG] submission_df.shape = {submission_df.shape}")
    print(f"[DEBUG] submission_df.columns = {submission_df.columns.tolist()}")
    print(f"[DEBUG] 'residue_index' in columns: {'residue_index' in submission_df.columns}")
    
    # Check the implementation of predict_submission
    print("\n[DEBUG] Analyzing predict_submission implementation")
    result = predictor.predict_3d_structure(sequence)
    coords = result["coords"]
    print(f"[DEBUG] coords.shape = {coords.shape}")
    print(f"[DEBUG] coords.dim() = {coords.dim()}")
    
    # Check if coords.dim() == 2 and coords.shape[0] != len(sequence)
    if coords.dim() == 2 and coords.shape[0] != len(sequence):
        print("[DEBUG] Condition 'coords.dim() == 2 and coords.shape[0] != len(sequence)' is TRUE")
        print("[DEBUG] This triggers the flat format branch in predict_submission")
    else:
        print("[DEBUG] Condition 'coords.dim() == 2 and coords.shape[0] != len(sequence)' is FALSE")
        
    # Try to reshape the coords
    from rna_predict.utils.submission import reshape_coords
    reshaped_coords = reshape_coords(coords, len(sequence))
    print(f"[DEBUG] reshaped_coords.shape = {reshaped_coords.shape}")
    print(f"[DEBUG] reshaped_coords.dim() = {reshaped_coords.dim()}")
    
    # Check if reshaped_coords is the same as coords (indicating reshape didn't change anything)
    if reshaped_coords is coords:
        print("[DEBUG] reshape_coords returned the original tensor (no reshaping occurred)")
    else:
        print("[DEBUG] reshape_coords returned a new tensor (reshaping occurred)")

if __name__ == "__main__":
    debug_predict_submission()
