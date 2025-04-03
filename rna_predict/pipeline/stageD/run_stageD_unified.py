"""
Unified RNA Stage D module with tensor shape compatibility fixes.

This module implements the diffusion-based refinement with built-in fixes
for tensor shape compatibility issues.
"""

import warnings  # Ensure warnings is imported

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
    trunk_embeddings: dict,
    diffusion_config: dict,
    mode: str = "inference",
    device: str = "cpu",
):
    """
    Run the diffusion-based refinement stage with tensor shape compatibility fixes.

    Args:
        partial_coords: Partial coordinates tensor
        trunk_embeddings: Dictionary of trunk embeddings
        diffusion_config: Configuration dictionary for diffusion
        mode: Either "inference" or "training"
        device: Device to run on ("cpu" or "cuda")

    Returns:
        Refined coordinates tensor
    """
    if mode not in ["inference", "train"]:
        raise ValueError(f"Unsupported mode: {mode}. Must be 'inference' or 'train'.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply tensor shape compatibility fixes
    apply_tensor_fixes()

    # Initialize diffusion components
    diffusion_conditioning = DiffusionConditioning(
        **diffusion_config.get("conditioning", {})
    ).to(device)

    # Correctly pass the main diffusion_config dictionary and device
    diffusion_manager = ProtenixDiffusionManager(
        diffusion_config=diffusion_config, device=device
    )
    # Note: .to(device) is likely handled internally by ProtenixDiffusionManager

    # --- Re-introduced Feature Preparation Logic ---
    # 2) Build or load the atom-level + token-level features
    #    Using placeholder logic as exact feature loading might depend on context
    #    This part needs careful review based on actual data loading requirements.
    # atom_feature_dict = {} # Placeholder - Initialized by loader now
    # token_feature_dict = {} # Placeholder - Initialized by loader now
    # Example: Load features if needed (replace with actual loading)
    # --- UNCOMMENTED AND PATH UPDATED ---
    atom_feature_dict, token_feature_dict = load_rna_data_and_features(
        "rna_predict/dataset/examples/1a9n_1_R.cif",  # <-- REPLACE WITH YOUR ACTUAL INPUT PATH
        device=device,
        override_num_atoms=partial_coords.shape[1],  # Keep if needed
    )
    # --- END UNCOMMENT ---

    # Ensure atom_to_token_idx is now present after loading
    if "atom_to_token_idx" not in atom_feature_dict:
        raise ValueError("atom_to_token_idx is still missing after loading features!")

    # Ensure essential keys exist even if loading fails or is skipped
    if "ref_pos" not in atom_feature_dict:
        atom_feature_dict["ref_pos"] = partial_coords.to(device)  # Use input coords
    if "restype" not in token_feature_dict:
        # Create dummy restype based on trunk_embeddings token dim if possible
        num_tokens = trunk_embeddings.get(
            "sing", trunk_embeddings.get("s_trunk")
        ).shape[1]
        token_feature_dict["restype"] = torch.zeros(
            partial_coords.shape[0], num_tokens, 32, device=device
        )  # Dummy restype

    # 3) Fix shape for "deletion_mean" if needed (assuming it comes from token_feature_dict)
    if "deletion_mean" in token_feature_dict:
        deletion = token_feature_dict["deletion_mean"]
        expected_tokens = token_feature_dict["restype"].shape[1]
        if deletion.ndim == 2:
            deletion = deletion.unsqueeze(-1)
        if deletion.shape[1] != expected_tokens:
            deletion = deletion[:, :expected_tokens, :]
        atom_feature_dict["deletion_mean"] = deletion  # Add to atom features

    # 4) Overwrite default coords with partial_coords
    atom_feature_dict["ref_pos"] = partial_coords.to(device)

    # 5) Merge necessary token-level features into atom_feature_dict
    atom_feature_dict["restype"] = token_feature_dict["restype"]
    if "profile" in token_feature_dict:  # Profile might be optional
        # Correct indentation for the line below
        atom_feature_dict["profile"] = token_feature_dict["profile"]

    # 6) If trunk_embeddings lacks "s_trunk", fallback to "sing"
    if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
        trunk_embeddings["s_trunk"] = trunk_embeddings.get("sing")
        if trunk_embeddings["s_trunk"] is None:
            raise ValueError(
                "Trunk embeddings must contain either 's_trunk' or 'sing'."
            )

    # 7) Use InputFeatureEmbedder to produce s_inputs (dimension depends on embedder config)
    #    Make sure embedder config matches diffusion_config expectations if needed
    embedder_config = diffusion_config.get(
        "embedder", {"c_atom": 128, "c_atompair": 16, "c_token": 384}
    )  # Example config
    embedder = InputFeatureEmbedder(**embedder_config).to(device)
    # Ensure all required features for embedder are in atom_feature_dict
    # Add dummy features if necessary for the embedder call
    required_embedder_keys = [
        "ref_pos",
        "restype",
        "profile",
        "deletion_mean",
    ]  # Example keys
    for key in required_embedder_keys:
        if key not in atom_feature_dict:
            # Create dummy data based on expected shapes or raise error
            warnings.warn(
                f"Missing feature '{key}' for InputFeatureEmbedder. Creating dummy."
            )
            # Example dummy creation - needs refinement based on actual embedder needs
            if key == "profile":
                atom_feature_dict[key] = torch.zeros_like(atom_feature_dict["restype"])
            elif key == "deletion_mean":
                atom_feature_dict[key] = torch.zeros_like(
                    atom_feature_dict["restype"][..., 0:1]
                )
            # Add more cases as needed

    s_inputs = embedder(
        atom_feature_dict, inplace_safe=False, chunk_size=None
    )  # Generate s_inputs

    # 8) Store generated s_inputs in trunk_embeddings if not already present
    if "s_inputs" not in trunk_embeddings or trunk_embeddings["s_inputs"] is None:
        # Correct indentation for the line below
        trunk_embeddings["s_inputs"] = s_inputs
    # --- End Re-introduced Logic ---

    # Run diffusion using the prepared features and embeddings
    if mode == "inference":
        coords = diffusion_manager.multi_step_inference(
            coords_init=partial_coords,
            trunk_embeddings=trunk_embeddings,  # Now contains s_trunk and s_inputs
            inference_params=diffusion_config.get("inference", {}),
            override_input_features=atom_feature_dict,  # Pass the prepared features
        )

        # Ensure the output has the correct batch dimension of 1
        if coords.ndim > 3:
            # If there are extra dimensions, squeeze them
            while coords.ndim > 3 and coords.shape[0] == 1:
                coords = coords.squeeze(0)
        elif coords.ndim == 2:
            # If missing batch dimension, add it
            coords = coords.unsqueeze(0)

        # Ensure batch dimension is 1
        if coords.shape[0] != 1:
            coords = coords[:1]  # Take only the first batch element

        return coords
    else:  # mode == "train"
        # Ensure label_dict is correctly formed for training
        label_dict = {
            "coordinate": partial_coords.to(device),
            "coordinate_mask": torch.ones_like(partial_coords[..., 0], device=device),
        }
        x_gt_out, x_denoised, sigma = diffusion_manager.train_diffusion_step(
            label_dict=label_dict,
            input_feature_dict=atom_feature_dict,  # Pass prepared features
            s_inputs=trunk_embeddings["s_inputs"],  # Use prepared s_inputs
            s_trunk=trunk_embeddings["s_trunk"],  # Use prepared s_trunk
            z_trunk=trunk_embeddings.get("pair"),
            sampler_params=diffusion_config.get("training", {}).get(
                "sampler_params", {}
            ),
            N_sample=1,
        )
        # Calculate loss properly - reduce spatial dimensions but preserve batch
        # x_denoised and x_gt_out have shape [B, N_sample, N_atom, 3]
        # First reduce spatial dimensions (N_atom, 3)
        loss = (x_denoised - x_gt_out).pow(2).mean(dim=(-1, -2))
        # Then reduce sample dimension if present
        if loss.dim() > 1:
            loss = loss.mean(dim=1)
        # Finally ensure scalar
        if loss.dim() > 0:
            loss = loss.mean()

        # Ensure sigma is a scalar
        if sigma.dim() > 0:
            sigma = sigma.mean()

        return x_denoised, loss, sigma


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
