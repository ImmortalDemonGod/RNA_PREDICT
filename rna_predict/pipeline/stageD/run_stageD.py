import torch
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager
# Import the loader utility for RNA features
from rna_predict.dataset.dataset_loader import load_rna_data_and_features
# Import the Protenix InputFeatureEmbedder for computing s_inputs with dimension 449
from protenix.model.modules.embedders import InputFeatureEmbedder

def run_stageD_diffusion(
    partial_coords: torch.Tensor,
    trunk_embeddings: dict,
    diffusion_config: dict,
    mode: str = "inference",
    device: str = "cpu"
):
    """
    Stage D entry:
      - partial_coords: [B, N_atom, 3]
      - trunk_embeddings: dict with "sing" [B,N_token,c_s], "pair" [B,N_token,N_token,c_z], etc.
      - diffusion_config: diffusion hyperparameters.
      - mode: "inference" or "train"
      - device: "cpu" or "cuda"

    Returns:
      if inference => final coordinates,
      if train => (x_denoised, loss, sigma)
    """
    manager = ProtenixDiffusionManager(diffusion_config, device=device)
    
    # Load real input feature dictionaries: atom-level and token-level features.
    atom_feature_dict, token_feature_dict = load_rna_data_and_features("demo_rna_file.cif", device=device)
    # Overwrite ref_pos in the atom_feature_dict with the provided partial coordinates
    atom_feature_dict["ref_pos"] = partial_coords

    # Ensure trunk_embeddings has the key "s_trunk" required by multi_step_inference.
    if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
        trunk_embeddings["s_trunk"] = trunk_embeddings.get("sing")

    # Instantiate the InputFeatureEmbedder to compute s_inputs with correct dimension (449)
    embedder = InputFeatureEmbedder(c_atom=128, c_atompair=16, c_token=384)
    # Call the modified forward method that only processes atom-level features.
    a = embedder.forward_atom_only(atom_feature_dict, inplace_safe=False)
    
    # Now manually concatenate the separate token-level features to form the final s_inputs.
    # Expected shapes:
    # a: [1, N_token, 384]
    # restype: [1, N_token, 32]
    # profile: [1, N_token, 32]
    # deletion_mean: [1, N_token, 1]
    s_inputs = torch.cat(
        [a, token_feature_dict["restype"], token_feature_dict["profile"], token_feature_dict["deletion_mean"]],
        dim=-1
    )
    
    s_trunk = trunk_embeddings["s_trunk"]
    z_trunk = trunk_embeddings.get("pair", None)

    if mode == "inference":
        inference_params = {"num_steps": 20, "sigma_max": 1.0, "N_sample": 1}
        coords_final = manager.multi_step_inference(
            coords_init=partial_coords,
            trunk_embeddings={
                "s_trunk": s_trunk,
                "pair": z_trunk
            },
            inference_params=inference_params,
            override_input_features=atom_feature_dict  # Pass only atom-level features
        )
        return coords_final

    elif mode == "train":
        sampler_params = {"p_mean": -1.2, "p_std": 1.0, "sigma_data": 16.0}
        
        # Build label dictionary for training
        label_dict = {
            "coordinate": partial_coords,
            "coordinate_mask": torch.ones_like(partial_coords[..., 0])  # No mask applied
        }
        
        x_gt_out, x_denoised, sigma = manager.train_diffusion_step(
            label_dict=label_dict,
            input_feature_dict=atom_feature_dict,  # Pass only atom-level features
            s_inputs=s_inputs,      # Now the dimension is 449 after concatenation
            s_trunk=s_trunk,        # dimension 384
            z_trunk=z_trunk,
            sampler_params=sampler_params,
            N_sample=1
        )
        loss = (x_denoised - x_gt_out).pow(2).mean()
        return x_denoised, loss, sigma

    else:
        raise ValueError(f"Unsupported mode: {mode}")
    

def demo_run_diffusion():
    """
    Demonstrates Stage D diffusion usage with partial coordinates and trunk embeddings
    for global refinement.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example revised config ensuring c_token is a multiple of n_heads
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 768,  # 768 is divisible by n_heads=16
        "transformer": {
            "n_blocks": 4,
            "n_heads": 16
        }
    }

    # OPTIONAL: Quick check for config validity
    if "transformer" in diffusion_config:
        n_heads = diffusion_config["transformer"].get("n_heads", 16)
        c_token = diffusion_config.get("c_token", 768)
        if c_token % n_heads != 0:
            raise ValueError(f"Invalid config: c_token={c_token} not divisible by n_heads={n_heads}.")

    # Suppose partial_coords is from Stage C or generated randomly
    partial_coords = torch.randn(1, 10, 3, device=device)
    trunk_embeddings = {
        "sing": torch.randn(1, 10, 384, device=device),
        "pair": torch.randn(1, 10, 10, 32, device=device)
    }

    coords_final = run_stageD_diffusion(
        partial_coords,
        trunk_embeddings,
        diffusion_config,
        mode="inference",
        device=device
    )
    print("[Diffusion Demo] coords_final shape:", coords_final.shape)

    # Training mode demonstration
    x_denoised, loss, sigma = run_stageD_diffusion(
        partial_coords,
        trunk_embeddings,
        diffusion_config,
        mode="train",
        device=device
    )
    print("[Diffusion Demo] x_denoised shape:", x_denoised.shape)
    print("[Diffusion Demo] loss:", loss.item())
    print("[Diffusion Demo] sigma:", sigma.item())

if __name__ == "__main__":
    demo_run_diffusion()