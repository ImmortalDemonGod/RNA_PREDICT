#!/usr/bin/env python3
"""
Extended test script to verify angle format compatibility between TorsionBERT and run_stageC.

This script tests how run_stageC handles different angle formats with a longer sequence
and more extreme angle values.
"""

import os
import sys
import torch
import math
import numpy as np

# Add the project root to the Python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the functions to test
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor

def print_section(title):
    """Print a section title with formatting."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

def check_for_nans(tensor, name):
    """Check if a tensor contains NaN values and log details."""
    if tensor is None:
        print(f"{name} is None")
        return True
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        print(f"{name} contains {nan_count} NaNs and {inf_count} Infs")
        return True
    return False

def test_angle_format_compatibility_extended():
    """Test how run_stageC handles different angle formats with a longer sequence and extreme angles."""
    print_section("Extended Angle Format Compatibility Test")
    
    # Use a longer sequence
    sequence = "GGCGCUAUGCGCCGAUAGCUAGCU"  # 24 nucleotides
    
    # Initialize result variables
    result_degrees = None
    result_radians = None
    result_sincos = None
    has_nan_degrees = True
    has_nan_radians = True
    has_nan_sincos = True
    
    # 1. Test with extreme angles in degrees
    print("Testing with extreme angles in degrees")
    # Use more extreme angles that are still valid
    degrees_angles = torch.zeros((len(sequence), 7))
    for i in range(len(sequence)):
        # Alternate between extreme positive and negative values
        if i % 2 == 0:
            degrees_angles[i] = torch.tensor([300.0, 180.0, 50.0, 85.0, 180.0, 290.0, 0.0])
        else:
            degrees_angles[i] = torch.tensor([-300.0, -180.0, -50.0, -85.0, -180.0, -290.0, 0.0])
    
    try:
        result_degrees = run_stageC(
            sequence=sequence,
            torsion_angles=degrees_angles,
            method="mp_nerf",
            device="cpu"
        )
        
        has_nan_degrees = check_for_nans(result_degrees["coords"], "degrees result")
        print(f"Extreme degrees format - NaNs: {has_nan_degrees}")
        print(f"Output shape: {result_degrees['coords'].shape}")
    except Exception as e:
        print(f"Error with extreme degrees: {str(e)}")
    
    # 2. Test with extreme angles in radians
    print("\nTesting with extreme angles in radians")
    # Convert to radians but use values outside the normal range
    radians_angles = torch.zeros((len(sequence), 7))
    for i in range(len(sequence)):
        if i % 2 == 0:
            radians_angles[i] = torch.tensor([5.0, 3.14, 0.8, 1.5, 3.14, 5.0, 0.0])
        else:
            radians_angles[i] = torch.tensor([-5.0, -3.14, -0.8, -1.5, -3.14, -5.0, 0.0])
    
    try:
        result_radians = run_stageC(
            sequence=sequence,
            torsion_angles=radians_angles,
            method="mp_nerf",
            device="cpu"
        )
        
        has_nan_radians = check_for_nans(result_radians["coords"], "radians result")
        print(f"Extreme radians format - NaNs: {has_nan_radians}")
        print(f"Output shape: {result_radians['coords'].shape}")
    except Exception as e:
        print(f"Error with extreme radians: {str(e)}")
    
    # 3. Test with extreme sin/cos pairs
    print("\nTesting with extreme sin/cos pairs")
    # Use values outside the normal [-1, 1] range for sin/cos
    sincos_angles = torch.zeros((len(sequence), 7))
    for i in range(len(sequence)):
        if i % 2 == 0:
            sincos_angles[i] = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        else:
            sincos_angles[i] = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
    
    try:
        result_sincos = run_stageC(
            sequence=sequence,
            torsion_angles=sincos_angles,
            method="mp_nerf",
            device="cpu"
        )
        
        has_nan_sincos = check_for_nans(result_sincos["coords"], "sin/cos result")
        print(f"Extreme sin/cos format - NaNs: {has_nan_sincos}")
        print(f"Output shape: {result_sincos['coords'].shape}")
    except Exception as e:
        print(f"Error with extreme sin/cos: {str(e)}")
    
    # Compare the results if all tests succeeded
    print("\nComparing results:")
    if result_degrees is not None and result_radians is not None and result_sincos is not None:
        if not has_nan_degrees and not has_nan_radians and not has_nan_sincos:
            # Calculate mean absolute difference between coordinates
            diff_deg_rad = torch.abs(result_degrees["coords"] - result_radians["coords"]).mean().item()
            diff_deg_sincos = torch.abs(result_degrees["coords"] - result_sincos["coords"]).mean().item()
            
            print(f"Mean absolute difference between degrees and radians: {diff_deg_rad:.6f}")
            print(f"Mean absolute difference between degrees and sin/cos: {diff_deg_sincos:.6f}")
            
            if diff_deg_rad > 1.0 and diff_deg_sincos > 1.0:
                print("\nCONCLUSION: As expected with extreme values, all formats produce significantly different results.")
            elif diff_deg_rad < 0.1 and diff_deg_sincos < 0.1:
                print("\nCONCLUSION: Surprisingly, all formats produce similar results even with extreme values.")
            else:
                print("\nCONCLUSION: Mixed results with extreme values.")
        else:
            print("Cannot compare results due to NaNs.")
    else:
        print("Cannot compare results because one or more tests failed.")

if __name__ == "__main__":
    test_angle_format_compatibility_extended()
