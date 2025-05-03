"""
Memory-efficient wrapper for Stage D diffusion.

This script provides a memory-efficient version of the Stage D diffusion process
by applying memory efficiency fixes to reduce memory usage.
"""

import torch
import argparse
from rna_predict.pipeline.stageD.memory_fix import run_stageD_with_memory_fixes

def main():
    """Main entry point for memory-efficient Stage D diffusion."""
    parser = argparse.ArgumentParser(description="Memory-efficient Stage D diffusion")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--mode", type=str, default="inference", choices=["inference", "train"], 
                        help="Mode to run in (inference or train)")
    parser.add_argument("--debug_logging", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Create example data (in a real scenario, this would be loaded from files)
    partial_coords = torch.randn(1, 100, 3)  # [batch, seq_len, 3]
    
    # Create trunk embeddings
    trunk_embeddings = {
        "s_inputs": torch.randn(1, 100, 449),  # [batch, seq_len, 449]
        "s_trunk": torch.randn(1, 100, 384),   # [batch, seq_len, 384]
        "pair": torch.randn(1, 100, 100, 64)   # [batch, seq_len, seq_len, pair_dim]
    }
    
    # Create diffusion config
    diffusion_config = {
        "conditioning": {
            "hidden_dim": 128,
            "num_heads": 8,
            "num_layers": 6,
        },
        "manager": {
            "hidden_dim": 128,
            "num_heads": 8,
            "num_layers": 6,
        },
        "inference": {
            "num_steps": 100,
            "noise_schedule": "linear",
        },
        "c_s_inputs": 449,
        "c_z": 64,
        "c_atom": 128,
        "c_s": 384,
        "c_token": 832,
        "transformer": {"n_blocks": 4, "n_heads": 16}
    }
    
    # Run with memory fixes
    result = run_stageD_with_memory_fixes(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode=args.mode,
        device=args.device
    )
    
    # Print result shape
    if args.mode == "inference":
        if hasattr(args, 'debug_logging') and args.debug_logging:
            print(f"Refined coordinates shape: {result.shape}")
    else:
        x_denoised, loss, sigma = result
        if hasattr(args, 'debug_logging') and args.debug_logging:
            print(f"x_denoised shape: {x_denoised.shape}")
            print(f"loss: {loss.item()}")
            print(f"sigma: {sigma.item()}")

if __name__ == "__main__":
    main() 