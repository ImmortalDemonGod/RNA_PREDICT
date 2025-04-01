#!/usr/bin/env python3
"""
Test script to verify angle format compatibility between TorsionBERT and run_stageC.

This script tests how run_stageC handles different angle formats:
1. Degrees (expected format)
2. Radians (converted from degrees)
3. Sin/Cos pairs (raw output from TorsionBERT)
"""

import os
import sys
import torch
import math

# Add the project root to the Python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the functions to test
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC

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

def test_angle_format_compatibility():
    """Test how run_stageC handles different angle formats."""
    print_section("Angle Format Compatibility Test")
    
    sequence = "ACGU"
    
    # 1. Test with angles in degrees (expected format)
    print("Testing with angles in degrees (expected format)")
    degrees_angles = torch.tensor([
        [300.0, 180.0, 50.0, 85.0, 180.0, 290.0, 0.0]  # A-form-like in degrees
    ] * len(sequence), dtype=torch.float32)
    
    result_degrees = run_stageC(
        sequence=sequence,
        torsion_angles=degrees_angles,
        method="mp_nerf",
        device="cpu"
    )
    
    has_nan_degrees = check_for_nans(result_degrees["coords"], "degrees result")
    print(f"Degrees format - NaNs: {has_nan_degrees}")
    print(f"Output shape: {result_degrees['coords'].shape}")
    
    # 2. Test with angles in radians (converted from degrees)
    print("\nTesting with angles in radians (converted from degrees)")
    radians_angles = degrees_angles * (torch.pi / 180.0)
    
    result_radians = run_stageC(
        sequence=sequence,
        torsion_angles=radians_angles,
        method="mp_nerf",
        device="cpu"
    )
    
    has_nan_radians = check_for_nans(result_radians["coords"], "radians result")
    print(f"Radians format - NaNs: {has_nan_radians}")
    print(f"Output shape: {result_radians['coords'].shape}")
    
    # 3. Test with sin/cos pairs
    print("\nTesting with sin/cos pairs")
    sincos_angles = torch.zeros((len(sequence), 7))  # 7 angles in sin/cos form (first 7 values)
    for i in range(len(sequence)):
        for j in range(7):
            angle_rad = radians_angles[i, j].item()
            sincos_angles[i, j] = math.sin(angle_rad) if j % 2 == 0 else math.cos(angle_rad)
    
    result_sincos = run_stageC(
        sequence=sequence,
        torsion_angles=sincos_angles,
        method="mp_nerf",
        device="cpu"
    )
    
    has_nan_sincos = check_for_nans(result_sincos["coords"], "sin/cos result")
    print(f"Sin/Cos format - NaNs: {has_nan_sincos}")
    print(f"Output shape: {result_sincos['coords'].shape}")
    
    # Compare the results
    print("\nComparing results:")
    if not has_nan_degrees and not has_nan_radians and not has_nan_sincos:
        # Calculate mean absolute difference between coordinates
        diff_deg_rad = torch.abs(result_degrees["coords"] - result_radians["coords"]).mean().item()
        diff_deg_sincos = torch.abs(result_degrees["coords"] - result_sincos["coords"]).mean().item()
        
        print(f"Mean absolute difference between degrees and radians: {diff_deg_rad:.6f}")
        print(f"Mean absolute difference between degrees and sin/cos: {diff_deg_sincos:.6f}")
        
        if diff_deg_rad < 0.1 and diff_deg_sincos > 1.0:
            print("\nCONCLUSION: As expected, radians format produces similar results to degrees,")
            print("but sin/cos format produces significantly different results.")
        elif diff_deg_rad < 0.1 and diff_deg_sincos < 0.1:
            print("\nCONCLUSION: Surprisingly, all formats produce similar results.")
        else:
            print("\nCONCLUSION: Unexpected differences between formats.")
    else:
        print("Cannot compare results due to NaNs.")

if __name__ == "__main__":
    test_angle_format_compatibility()