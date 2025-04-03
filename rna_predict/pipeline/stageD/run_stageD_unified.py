
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

    # Create and initialize the diffusion manager
    diffusion_manager = ProtenixDiffusionManager(
        diffusion_config=diffusion_config,
        device=device,
    )

    # Prepare input features if not provided
    if input_features is None:
        # Build or load the atom-level + token-level features
        atom_feature_dict, token_feature_dict = load_rna_data_and_features(
            "demo_rna_file.cif", device=device, override_num_atoms=partial_coords.shape[1]
        )

        # Fix shape for "deletion_mean" if needed
        if "deletion_mean" in token_feature_dict:
            deletion = token_feature_dict["deletion_mean"]
            expected_tokens = token_feature_dict["restype"].shape[1]
            if deletion.ndim == 2:
                deletion = deletion.unsqueeze(-1)
            if deletion.shape[1] != expected_tokens:
                deletion = deletion[:, :expected_tokens, :]
            atom_feature_dict["deletion_mean"] = deletion

        # Overwrite default coords with partial_coords
        atom_feature_dict["ref_pos"] = partial_coords

        # Merge token-level features so embedder can produce 449-dim
        atom_feature_dict["restype"] = token_feature_dict["restype"]
        atom_feature_dict["profile"] = token_feature_dict["profile"]

        prepared_features = atom_feature_dict
    else:
        # Use the provided features directly
        prepared_features = input_features
        # Ensure provided features are on the correct device
        for key, value in prepared_features.items():
            if isinstance(value, torch.Tensor):
                prepared_features[key] = value.to(device)

    # Run diffusion
    if mode == "inference":
        # Ensure partial_coords has the right shape [B, N_atom, 3]
        if partial_coords.dim() == 4:
            partial_coords = partial_coords.squeeze(1)
        elif partial_coords.dim() == 5:
            # Handle case where we have an extra dimension before N_sample
            partial_coords = partial_coords.squeeze(0).squeeze(0)

        # Ensure trunk embeddings have consistent shapes
        for key in ["s_trunk", "s_inputs"]:
            if key in trunk_embeddings and trunk_embeddings[key] is not None:
                if trunk_embeddings[key].dim() == 4:
                    trunk_embeddings[key] = trunk_embeddings[key].squeeze(1)
                elif trunk_embeddings[key].dim() == 5:
                    trunk_embeddings[key] = trunk_embeddings[key].squeeze(0).squeeze(0)

        if "pair" in trunk_embeddings and trunk_embeddings["pair"] is not None:
            if trunk_embeddings["pair"].dim() == 5:
                trunk_embeddings["pair"] = trunk_embeddings["pair"].squeeze(1)
            elif trunk_embeddings["pair"].dim() == 6:
                trunk_embeddings["pair"] = trunk_embeddings["pair"].squeeze(0).squeeze(0)

        # Set N_sample to 1 in inference params to avoid extra dimensions
        inference_params = diffusion_config.get("inference", {})
        inference_params["N_sample"] = 1

        coords = diffusion_manager.multi_step_inference(
            coords_init=partial_coords.to(device),
            trunk_embeddings=trunk_embeddings,
            inference_params=inference_params,
            override_input_features=prepared_features,
            debug_logging=True,  # Enable debug logging
        )

        # Ensure output has shape [B, N_atom, 3] by squeezing any extra dimensions
        if coords.dim() == 4:
            coords = coords.squeeze(1)  # Remove N_sample dimension when it's 1
        elif coords.dim() == 5:
            # Handle case where we have an extra dimension before N_sample
            coords = coords.squeeze(0).squeeze(0)

        # Final shape check
        if coords.dim() != 3:
            warnings.warn(f"Unexpected output shape: {coords.shape}. Expected [B, N_atom, 3]")
            # Force the shape if possible
            if coords.dim() > 3:
                while coords.dim() > 3:
                    coords = coords.squeeze(0)

        return coords
    else:  # mode == "train"
        # For training mode
        label_dict = {
            "coordinate": partial_coords.to(device),
            "coordinate_mask": torch.ones_like(partial_coords[..., 0], device=device),
        }
        
        # Ensure s_inputs and z_trunk are tensors
        s_inputs = trunk_embeddings.get("s_inputs")
        if s_inputs is None:
            warnings.warn("'s_inputs' not found in trunk_embeddings for training mode. Check upstream logic.")
            raise ValueError("Training mode requires 's_inputs' in trunk_embeddings or generation logic.")
            
        z_trunk = trunk_embeddings.get("pair")
        # Fallback for z_trunk if needed (optional for some models)
        if z_trunk is None:
            warnings.warn("Fallback: Creating dummy 'z_trunk' for training.")
            c_z = diffusion_config.get("c_z", 32)
            s_shape = trunk_embeddings["s_trunk"].shape
            z_shape = (s_shape[0], s_shape[1], s_shape[1], c_z)
            z_trunk = torch.zeros(z_shape, device=device, dtype=trunk_embeddings["s_trunk"].dtype)
            
        x_gt_out, x_denoised, sigma = diffusion_manager.train_diffusion_step(
            label_dict=label_dict,
            input_feature_dict=prepared_features,
            s_inputs=s_inputs.to(device),
            s_trunk=trunk_embeddings["s_trunk"].to(device),
            z_trunk=z_trunk.to(device),
            sampler_params=diffusion_config.get("training", {}).get("sampler_params", {}),
            N_sample=1,
        )

        # Ensure sigma is a scalar by taking mean if it's not already
        if sigma.dim() > 0:
            sigma = torch.mean(sigma)

        # Return the expected tuple format
        return x_denoised, x_gt_out, sigma


def demo_run_diffusion():
    """
    Demo function to run the diffusion stage with a simple example.
    """
    # Load example data
    # Note: load_rna_data_and_features might need adjustment if "example.pdb" isn't a valid path
    # or if the function expects different arguments in a real scenario.
    # For now, we focus on fixing the tensor shape mismatch.
    # data = load_rna_data_and_features("example.pdb") # Commented out as it's not directly related to the shape error

    # Create partial coordinates
    partial_coords = torch.randn(1, 100, 3)  # [batch, seq_len, 3]

    # Create trunk embeddings
    trunk_embeddings = {
        "s_inputs": torch.randn(1, 100, 449),  # <<< MODIFIED: Changed dimension from 128 to 449
        "s_trunk": torch.randn(1, 100, 384),   # Corrected in previous step
        "z_trunk": torch.randn(1, 100, 100, 64),  # [batch, seq_len, seq_len, pair_dim] - Assuming 64 is correct
    }

    # Create diffusion config - Assuming these dimensions are correct for the demo
    # If DiffusionConditioning was initialized differently via ProtenixDiffusionManager
    # using this config, that would be another place to check.
    # But the error points to the default c_s=384 in DiffusionConditioning's init.
    diffusion_config = {
        "conditioning": {"hidden_dim": 128, "num_heads": 8, "num_layers": 6}, # Example values
        "manager": {"hidden_dim": 128, "num_heads": 8, "num_layers": 6},      # Example values
        "inference": {"num_steps": 100, "noise_schedule": "linear"},          # <<< MODIFIED: Changed num_steps back to 100
        # Ensure the actual config used aligns with model expectations if not using defaults
        # Specifically, how ProtenixDiffusionManager uses this config to potentially
        # override DiffusionConditioning's c_s default might be relevant in a real scenario.
        # For this demo fix, we align the data to the default expectation.
    }

    # Run diffusion
    refined_coords = run_stageD_diffusion(
        partial_coords,
        trunk_embeddings,
        diffusion_config,
        mode="inference",
        device="cpu",
        # input_features could be passed here if needed, currently uses internal loading logic
    )

    print(f"Refined coordinates shape: {refined_coords.shape}")
    return refined_coords
