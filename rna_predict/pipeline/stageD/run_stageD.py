import torch
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager
# Import the loader utility for RNA features
from rna_predict.dataset.dataset_loader import load_rna_data_and_features
# Import the Protenix InputFeatureEmbedder for computing s_inputs with dimension 449
from rna_predict.pipeline.stageA.input_embedding.current.embedders import InputFeatureEmbedder
import snoop

@snoop
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
    atom_feature_dict, token_feature_dict = load_rna_data_and_features(
        "demo_rna_file.cif",
        device=device,
        override_num_atoms=partial_coords.shape[1]
    )

    # Check or fix shape of deletion_mean
    if "deletion_mean" in token_feature_dict:
        deletion = token_feature_dict["deletion_mean"]
        expected_tokens = token_feature_dict["restype"].shape[1]
        if deletion.shape[1] != expected_tokens:
            raise ValueError(f"deletion_mean middle dimension {deletion.shape[1]} != {expected_tokens}")
        # If deletion_mean is not the correct shape, fix or skip it
        if deletion.shape[1] == expected_tokens:
            # If deletion_mean is not the correct shape, fix or skip it
            if deletion.shape[1] == expected_tokens:
                atom_feature_dict["deletion_mean"] = deletion
            else:
                print("[WARN] skipping 'deletion_mean' because shape is", deletion.shape, "expected", (deletion.shape[0], expected_tokens, 1))
        else:
            print("[WARN] skipping 'deletion_mean' because shape is", deletion.shape, "expected", (deletion.shape[0], expected_tokens, 1))

    # Overwrite ref_pos in the atom_feature_dict with the provided partial coordinates
    atom_feature_dict["ref_pos"] = partial_coords

    # Merge token-level features into atom_feature_dict so that the embedder can produce a 449-dim output.
    atom_feature_dict["restype"] = token_feature_dict["restype"]
    atom_feature_dict["profile"] = token_feature_dict["profile"]
    
    # Force deletion_mean to have shape (batch, num_tokens, 1)
    deletion = token_feature_dict["deletion_mean"]
    expected_tokens = token_feature_dict["restype"].shape[1]  # expected token count
    if deletion.ndim == 2:
        deletion = deletion.unsqueeze(-1)
    if deletion.shape[1] != expected_tokens:
        deletion = deletion[:, :expected_tokens, :]
    atom_feature_dict["deletion_mean"] = deletion
 
    # Ensure trunk_embeddings has the key "s_trunk" required by multi_step_inference.
    if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
        trunk_embeddings["s_trunk"] = trunk_embeddings.get("sing")
 
    # Instantiate the standard InputFeatureEmbedder to produce a 449-dim output
    embedder = InputFeatureEmbedder(c_atom=128, c_atompair=16, c_token=384)
    s_inputs = embedder(atom_feature_dict, inplace_safe=False, chunk_size=None)

    # Store the 449-dim s_inputs in trunk_embeddings so that multi_step_inference
    # does not fall back to s_trunk (which is only 384).
    trunk_embeddings["s_inputs"] = s_inputs
    s_inputs = embedder(atom_feature_dict, inplace_safe=False, chunk_size=None)
    
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
            override_input_features=atom_feature_dict,  # Pass only atom-level features
            debug_logging=True
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
        # Now we unify shape to 832 (divisible by 16)
        "c_token": 832,
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