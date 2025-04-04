"""
Unified RNA Stage D module with tensor shape compatibility fixes.

This module implements the diffusion-based refinement with built-in fixes
for tensor shape compatibility issues.
"""

import warnings  # Ensure warnings is imported
from typing import Dict, Any, Union, Tuple

import torch
import torch.nn.functional as F # Added for loss calculation

from rna_predict.dataset.dataset_loader import load_rna_data_and_features
from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    InputFeatureEmbedder,
)
from rna_predict.pipeline.stageD.diffusion.components.diffusion_conditioning import DiffusionConditioning # Updated import path
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.tensor_fixes import apply_tensor_fixes


def validate_and_fix_shapes(
    partial_coords: torch.Tensor,
    trunk_embeddings: Dict[str, torch.Tensor],
    input_features: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Validate and fix tensor shapes to ensure compatibility.

    Args:
        partial_coords: Initial coordinates [B, N_atom, 3]
        trunk_embeddings: Dictionary with trunk embeddings
        input_features: Dictionary of input features

    Returns:
        Tuple of (fixed_partial_coords, fixed_trunk_embeddings, fixed_input_features)
    """
    # Fix partial_coords shape
    if partial_coords.dim() == 4:
        partial_coords = partial_coords.squeeze(1)
    elif partial_coords.dim() == 5:
        partial_coords = partial_coords.squeeze(0).squeeze(0)

    # Get batch size and number of atoms
    batch_size = partial_coords.shape[0]
    num_atoms = partial_coords.shape[1]

    # Fix trunk embeddings
    fixed_trunk_embeddings = {}
    for key, value in trunk_embeddings.items():
        if value is None:
            fixed_trunk_embeddings[key] = None # Keep None values
            continue

        temp_value = value # Work with a temporary variable

        if key in ["s_trunk", "s_inputs", "sing"]: # Added 'sing'
            if temp_value.dim() == 4:
                temp_value = temp_value.squeeze(1)
            elif temp_value.dim() == 5:
                temp_value = temp_value.squeeze(0).squeeze(0)
            # Ensure batch size matches
            if temp_value.shape[0] != batch_size:
                 warnings.warn(f"Adjusting batch size for {key} from {temp_value.shape[0]} to {batch_size}")
                 temp_value = temp_value[:batch_size]
            # Ensure sequence length matches num_atoms (assuming 1-1 mapping for simplicity here)
            # A more robust check might use atom_to_token_idx max value + 1
            if temp_value.shape[1] != num_atoms:
                 warnings.warn(f"Adjusting sequence length for {key} from {temp_value.shape[1]} to {num_atoms}")
                 temp_value = temp_value[:, :num_atoms]

            fixed_trunk_embeddings[key] = temp_value

        elif key == "pair":
            if temp_value.dim() == 5:
                temp_value = temp_value.squeeze(1)
            elif temp_value.dim() == 6:
                temp_value = temp_value.squeeze(0).squeeze(0)
            # Ensure batch size matches
            if temp_value.shape[0] != batch_size:
                 warnings.warn(f"Adjusting batch size for {key} from {temp_value.shape[0]} to {batch_size}")
                 temp_value = temp_value[:batch_size]
            # Ensure sequence lengths match num_atoms
            if temp_value.shape[1] != num_atoms or temp_value.shape[2] != num_atoms:
                 warnings.warn(f"Adjusting sequence lengths for {key} from ({temp_value.shape[1]}, {temp_value.shape[2]}) to ({num_atoms}, {num_atoms})")
                 temp_value = temp_value[:, :num_atoms, :num_atoms]

            fixed_trunk_embeddings[key] = temp_value
        else: # Keep other keys as is
             fixed_trunk_embeddings[key] = temp_value


    # Fix input features
    fixed_input_features = {}
    for key, value in input_features.items():
        if isinstance(value, torch.Tensor):
            temp_value = value # Work with temporary variable

            # Ensure batch dimension exists if missing and expected
            if temp_value.dim() == 1 and key in ["atom_to_token_idx", "ref_space_uid"]:
                 temp_value = temp_value.unsqueeze(0) # Add batch dim
            elif temp_value.dim() == 2 and key in ["ref_pos", "ref_charge", "ref_element", "ref_atom_name_chars", "ref_mask", "restype", "profile", "deletion_mean", "sing"]: # Added sing
                 temp_value = temp_value.unsqueeze(0) # Add batch dim

            # Now adjust batch size and sequence/atom length if tensor has at least 2 dims
            if temp_value.ndim >= 2:
                if temp_value.shape[0] != batch_size:
                     warnings.warn(f"Adjusting batch size for input_feature '{key}' from {temp_value.shape[0]} to {batch_size}")
                     temp_value = temp_value[:batch_size]

                # Adjust sequence/atom dimension (usually dim 1 after batch)
                if key in ["atom_to_token_idx", "ref_pos", "ref_space_uid", "ref_charge", "ref_element", "ref_atom_name_chars", "ref_mask", "deletion_mean", "sing"]: # Added sing
                     if temp_value.shape[1] != num_atoms:
                          warnings.warn(f"Adjusting atom dimension for input_feature '{key}' from {temp_value.shape[1]} to {num_atoms}")
                          temp_value = temp_value[:, :num_atoms]
                elif key in ["restype", "profile"]: # Token-level features
                     if temp_value.shape[1] != num_atoms: # Assuming N_token == N_atom here
                          warnings.warn(f"Adjusting token dimension for input_feature '{key}' from {temp_value.shape[1]} to {num_atoms}")
                          temp_value = temp_value[:, :num_atoms]

            # Specific shape adjustments
            if key == "atom_to_token_idx" and temp_value.ndim > 2:
                 temp_value = temp_value.squeeze(-1) # Ensure [B, N_atom]

            fixed_input_features[key] = temp_value
        else:
            fixed_input_features[key] = value

    # Ensure ref_pos uses the potentially fixed partial_coords
    fixed_input_features["ref_pos"] = partial_coords

    return partial_coords, fixed_trunk_embeddings, fixed_input_features


def run_stageD_diffusion(
    partial_coords: torch.Tensor,
    trunk_embeddings: Dict[str, torch.Tensor],
    diffusion_config: Dict[str, Any],
    mode: str = "inference",
    device: str = "cpu",
    input_features: Dict[str, Any] | None = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run Stage D diffusion refinement.

    Args:
        partial_coords: Initial coordinates [B, N_atom, 3]
        trunk_embeddings: Dictionary with trunk embeddings (e.g., s_trunk, pair)
                          This dictionary might be modified in-place to cache 's_inputs'.
        diffusion_config: Configuration for diffusion components
        mode: Either "inference" or "training"
        device: Device to run on
        input_features: Optional pre-computed input feature dictionary

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

    # Store reference to the original dictionary for potential updates
    original_trunk_embeddings_ref = trunk_embeddings

    # Apply tensor shape compatibility fixes
    apply_tensor_fixes() # <<< RESTORED

    # Create and initialize the diffusion manager
    diffusion_manager = ProtenixDiffusionManager(
        diffusion_config=diffusion_config,
        device=device,
    )

    # Prepare input features if not provided (basic fallback)
    if input_features is None:
         warnings.warn("input_features not provided, using basic fallback based on partial_coords.")
         N = partial_coords.shape[1]
         # --- Corrected Fallback Input Features ---
         input_features = {
              "atom_to_token_idx": torch.arange(N, device=device).unsqueeze(0),
              "ref_pos": partial_coords.to(device), # Use provided coords
              "ref_space_uid": torch.arange(N, device=device).unsqueeze(0),
              "ref_charge": torch.zeros(1, N, 1, device=device),
              "ref_element": torch.zeros(1, N, 128, device=device), # Assuming c_atom=128
              "ref_atom_name_chars": torch.zeros(1, N, 256, device=device), # Assuming size
              "ref_mask": torch.ones(1, N, 1, device=device),
              "restype": torch.zeros(1, N, 32, device=device), # Assuming size
              "profile": torch.zeros(1, N, 32, device=device), # Assuming size
              "deletion_mean": torch.zeros(1, N, 1, device=device),
              "sing": torch.zeros(1, N, diffusion_config.get("c_s_inputs", 449), device=device), # Use config or default
         }
         # --- End Correction ---


    # Validate and fix shapes - NOTE: This returns potentially modified copies
    # Pass the original trunk_embeddings ref to validate_and_fix_shapes
    partial_coords, trunk_embeddings_internal, input_features = validate_and_fix_shapes(
        partial_coords, original_trunk_embeddings_ref, input_features
    )

    # Run diffusion
    if mode == "inference":
        # Set N_sample to 1 in inference params to avoid extra dimensions
        inference_params = diffusion_config.get("inference", {})
        inference_params["N_sample"] = 1

        # Pass the internal (potentially fixed) copy to the manager
        coords = diffusion_manager.multi_step_inference(
            coords_init=partial_coords.to(device),
            trunk_embeddings=trunk_embeddings_internal,
            inference_params=inference_params,
            override_input_features=input_features,
            debug_logging=True, # Keep debug logging for now
        )

        # Debug print after the call
        print(f"[DEBUG] After inference call: 's_inputs' in internal? {'s_inputs' in trunk_embeddings_internal}, 's_inputs' in original? {'s_inputs' in original_trunk_embeddings_ref}")

        # Update the original trunk_embeddings dict with cached s_inputs if it was added
        # Check the internal dict used by multi_step_inference
        if "s_inputs" in trunk_embeddings_internal and "s_inputs" not in original_trunk_embeddings_ref:
             print("[DEBUG] Copying cached 's_inputs' back to original dictionary.") # DEBUG PRINT
             original_trunk_embeddings_ref["s_inputs"] = trunk_embeddings_internal["s_inputs"]


        return coords
    else:  # mode == "train"
        # For training mode
        label_dict = {
            "coordinate": partial_coords.to(device),
            "coordinate_mask": torch.ones_like(partial_coords[..., 0], device=device),
        }

        # Ensure s_inputs and z_trunk are tensors within the internal copy
        s_inputs = trunk_embeddings_internal.get("s_inputs")
        if s_inputs is None:
            # Try to get s_inputs from input_features (using 'sing' as fallback key)
            s_inputs = input_features.get("sing")
            if s_inputs is None:
                warnings.warn("'s_inputs' not found in trunk_embeddings or input_features ('sing') for training mode. Using fallback.")
                # Create a fallback s_inputs with the right shape
                n_tokens = trunk_embeddings_internal["s_trunk"].shape[1]
                # --- Corrected Config Access ---
                conditioning_config_nested = diffusion_config.get("conditioning", {})
                c_s_inputs_dim = diffusion_config.get("c_s_inputs", conditioning_config_nested.get("c_s_inputs", 449))
                # --- End Correction ---
                s_inputs = torch.zeros((1, n_tokens, c_s_inputs_dim), device=device)

            # Update the internal copy
            trunk_embeddings_internal["s_inputs"] = s_inputs
            # Also update the original reference if it wasn't there initially
            if "s_inputs" not in original_trunk_embeddings_ref:
                print("[DEBUG] Copying generated 's_inputs' back to original dictionary (train mode).") # DEBUG PRINT
                original_trunk_embeddings_ref["s_inputs"] = s_inputs


        z_trunk = trunk_embeddings_internal.get("pair")
        # Fallback for z_trunk if needed
        if z_trunk is None:
            warnings.warn("Fallback: Creating dummy 'z_trunk' for training.")
            n_tokens = trunk_embeddings_internal["s_trunk"].shape[1]
            # --- Corrected Config Access ---
            conditioning_config_nested = diffusion_config.get("conditioning", {})
            c_z_dim = diffusion_config.get("c_z", conditioning_config_nested.get("c_z", 128))
            # --- End Correction ---
            z_trunk = torch.zeros((1, n_tokens, n_tokens, c_z_dim), device=device)
            # Update the internal copy
            trunk_embeddings_internal["pair"] = z_trunk
            # Also update the original reference if it wasn't there initially
            if "pair" not in original_trunk_embeddings_ref:
                 print("[DEBUG] Copying generated 'pair' back to original dictionary (train mode).") # DEBUG PRINT
                 original_trunk_embeddings_ref["pair"] = z_trunk


        # Run training step using the internal copy
        x_denoised_tuple = diffusion_manager.train_diffusion_step(
            label_dict=label_dict,
            input_feature_dict=input_features,
            s_inputs=trunk_embeddings_internal["s_inputs"], # Use from internal copy
            s_trunk=trunk_embeddings_internal["s_trunk"], # Use from internal copy
            z_trunk=trunk_embeddings_internal["pair"],    # Use from internal copy
            sampler_params={"sigma_data": diffusion_config["sigma_data"]},
            N_sample=1,
        )

        # Unpack the results - x_gt_augment, x_denoised, sigma
        x_gt_augment, x_denoised, sigma = x_denoised_tuple
        # Ensure sigma is a scalar tensor
        if sigma.dim() > 0:
            sigma = sigma.mean().squeeze()  # Take mean and remove all dimensions
        return x_denoised, sigma, x_gt_augment


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
        "c_s_inputs": 449, # Added for fallback creation
        "c_z": 64,         # Added for fallback creation
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
