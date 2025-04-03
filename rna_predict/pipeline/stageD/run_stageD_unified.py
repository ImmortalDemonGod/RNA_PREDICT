"""
Unified RNA Stage D module with tensor shape compatibility fixes.

This module implements the diffusion-based refinement with built-in fixes
for tensor shape compatibility issues.
"""

import warnings  # Ensure warnings is imported
from typing import Dict, Any, Union, Tuple

import torch

from rna_predict.dataset.dataset_loader import load_rna_data_and_features
from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    InputFeatureEmbedder,
)
from rna_predict.pipeline.stageD.diffusion.components.diffusion_conditioning import DiffusionConditioning # Updated import path
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.tensor_fixes import apply_tensor_fixes


def run_stageD_diffusion(
    partial_coords: torch.Tensor,
    trunk_embeddings: Dict[str, torch.Tensor],
    diffusion_config: Dict[str, Any],
    mode: str = "inference",
    device: str = "cpu",
    input_features: Dict[str, Any] | None = None,  # <-- Add new optional argument
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run Stage D diffusion refinement.

    Args:
        partial_coords: Initial coordinates [B, N_atom, 3]
        trunk_embeddings: Dictionary with trunk embeddings (e.g., s_trunk, pair)
        diffusion_config: Configuration for diffusion components
        mode: Either "inference" or "training"
        device: Device to run on
        input_features: Optional pre-computed input feature dictionary. If provided,
                        internal feature loading and preparation will be skipped.
                        Must contain necessary keys like 'restype', 'atom_to_token_idx', etc.

    Returns:
        If mode == "inference":
            Refined coordinates [B, N_atom, 3]
        If mode == "train":
            Tuple of (x_denoised, x_gt_out, sigma)
    """
    if mode not in ["inference", "train"]:
        raise ValueError(f"Unsupported mode: {mode}. Must be 'inference' or 'train'.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply tensor shape compatibility fixes
    apply_tensor_fixes()

    # --- Feature Preparation ---
    if input_features is None:
        # If features aren't provided, load and prepare them internally
        warnings.warn("input_features not provided, loading from default path.") # Add warning
        atom_feature_dict, token_feature_dict = load_rna_data_and_features(
            "rna_predict/dataset/examples/1a9n_1_R.cif",  # <-- REPLACE WITH YOUR ACTUAL INPUT PATH
            device=device,
            override_num_atoms=partial_coords.shape[1],  # Keep if needed
        )

        # Ensure token_feature_dict has the right dimensions
        for key in token_feature_dict:
            if key == "deletion_mean" and token_feature_dict[key].dim() == 2:
                token_feature_dict[key] = token_feature_dict[key].unsqueeze(-1)

        # Merge token_feature_dict into atom_feature_dict
        for key in ["restype", "profile", "deletion_mean"]:
            if key in token_feature_dict:
                atom_feature_dict[key] = token_feature_dict[key]
        
        # Add fallback for essential keys if loading failed
        if "restype" not in atom_feature_dict:
             warnings.warn("Fallback: Creating dummy 'restype'.")
             num_tokens = trunk_embeddings["s_trunk"].shape[1] # Assuming s_trunk exists
             # Example dummy shape, adjust if needed
             atom_feature_dict["restype"] = torch.zeros(partial_coords.shape[0], num_tokens, 32, device=device)
        if "atom_to_token_idx" not in atom_feature_dict:
             warnings.warn("Fallback: Creating dummy 'atom_to_token_idx'.")
             num_atoms = partial_coords.shape[1]
             atom_feature_dict["atom_to_token_idx"] = torch.arange(num_atoms, device=device).unsqueeze(0)


        prepared_features = atom_feature_dict
    else:
        # Use the provided features directly
        prepared_features = input_features
        # Ensure provided features are on the correct device
        for key, value in prepared_features.items():
            if isinstance(value, torch.Tensor):
                prepared_features[key] = value.to(device)
    # --- End Feature Preparation ---


    # Update diffusion_config to match the expected dimensions (This might be redundant if config is passed correctly)
    # Consider removing this hardcoded update if config is reliable
    # diffusion_config.update({
    #     "c_s_inputs": 384,  # Match the input embedder dimension
    # })

    # Create and initialize the diffusion manager
    diffusion_manager = ProtenixDiffusionManager(
        diffusion_config=diffusion_config,
        device=device,
    )

    # Run diffusion
    if mode == "inference":
        coords = diffusion_manager.multi_step_inference(
            coords_init=partial_coords.to(device), # Ensure coords are on device
            trunk_embeddings=trunk_embeddings, # Manager moves these internally now
            inference_params=diffusion_config.get("inference", {}), # Get inference params from config
            override_input_features=prepared_features, # Use prepared features
        )
    else: # mode == "train"
        # For training mode
        label_dict = {
            "coordinate": partial_coords.to(device),
            "coordinate_mask": torch.ones_like(partial_coords[..., 0], device=device),
        }
        
        # Ensure s_inputs and z_trunk are tensors
        s_inputs = trunk_embeddings.get("s_inputs")
        if s_inputs is None:
            # If s_inputs not provided, it might be generated internally or needs fallback
            # This part might need adjustment based on how s_inputs is handled upstream
             warnings.warn("'s_inputs' not found in trunk_embeddings for training mode. Check upstream logic.")
             # Example fallback: Use s_trunk if dimensions match expected c_s_inputs? Risky.
             # Or generate using InputFeatureEmbedder here? Requires embedder config.
             # For now, raise error or use a placeholder if essential
             raise ValueError("Training mode requires 's_inputs' in trunk_embeddings or generation logic.")

            
        z_trunk = trunk_embeddings.get("pair")
        # Fallback for z_trunk if needed (optional for some models)
        if z_trunk is None:
            warnings.warn("Fallback: Creating dummy 'z_trunk' for training.")
            # Create a zero tensor with the expected shape based on s_trunk and c_z
            c_z = diffusion_config.get("c_z", 32) # Get c_z from config
            s_shape = trunk_embeddings["s_trunk"].shape
            z_shape = (s_shape[0], s_shape[1], s_shape[1], c_z) # B, N_token, N_token, c_z
            z_trunk = torch.zeros(z_shape, device=device, dtype=trunk_embeddings["s_trunk"].dtype)

            
        x_gt_out, x_denoised, sigma = diffusion_manager.train_diffusion_step(
            label_dict=label_dict,
            input_feature_dict=prepared_features, # Use prepared features
            s_inputs=s_inputs.to(device), # Ensure on device
            s_trunk=trunk_embeddings["s_trunk"].to(device), # Ensure on device
            z_trunk=z_trunk.to(device), # Ensure on device
            sampler_params=diffusion_config.get("training", {}).get("sampler_params", {}), # Get sampler params
            N_sample=1, # Assuming N_sample=1 for training step
        )
        # Compute MSE loss between denoised output and ground truth
        loss = torch.mean((x_denoised - x_gt_out) ** 2)
        # Ensure sigma is a scalar by taking mean if it's not already
        if sigma.dim() > 0:
            sigma = torch.mean(sigma)
        # Return the expected tuple format
        return x_denoised, loss, sigma

    return coords


def demo_run_diffusion():
    """
    Demo function to run the diffusion stage with a simple example.
    """
    # Load example data
    data = load_rna_data_and_features("example.pdb")

    # Create partial coordinates
    partial_coords = torch.randn(1, 100, 3)  # [batch, seq_len, 3]

    # Create trunk embeddings
    trunk_embeddings = {
        "s_inputs": torch.randn(1, 100, 128),  # [batch, seq_len, hidden_dim]
        "s_trunk": torch.randn(1, 100, 128),
        "z_trunk": torch.randn(1, 100, 100, 64),  # [batch, seq_len, seq_len, pair_dim]
    }

    # Create diffusion config
    diffusion_config = {
        "conditioning": {"hidden_dim": 128, "num_heads": 8, "num_layers": 6},
        "manager": {"hidden_dim": 128, "num_heads": 8, "num_layers": 6},
        "inference": {"num_steps": 100, "noise_schedule": "linear"},
    }

    # Run diffusion
    refined_coords = run_stageD_diffusion(
        partial_coords,
        trunk_embeddings,
        diffusion_config,
        mode="inference",
        device="cpu",
    )

    print(f"Refined coordinates shape: {refined_coords.shape}")
    return refined_coords
