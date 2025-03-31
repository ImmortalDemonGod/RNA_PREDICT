import snoop
import torch

from rna_predict.dataset.dataset_loader import load_rna_data_and_features
from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    InputFeatureEmbedder,
)
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)


# @snoop
def run_stageD_diffusion(
    partial_coords: torch.Tensor,
    trunk_embeddings: dict,
    diffusion_config: dict,
    mode: str = "inference",
    device: str = "cpu",
):
    """
    Stage D entry function that orchestrates the final diffusion-based refinement.

    Args:
        partial_coords (torch.Tensor): [B, N_atom, 3], partial or initial coordinates
        trunk_embeddings (dict): dictionary typically containing
           - "sing" or "s_trunk": shape [B, N_token, 384]
           - "pair": [B, N_token, N_token, c_z]
           - optionally "s_inputs": [B, N_token, 449]
        diffusion_config (dict): hyperparameters for building the diffusion module
        mode (str): "inference" or "train"
        device (str): "cpu" or "cuda" device

    Returns:
        If mode=="inference": A tensor of final coordinates.
        If mode=="train": (x_denoised, loss, sigma).
    """

    # 1) Build the diffusion manager
    manager = ProtenixDiffusionManager(diffusion_config, device=device)

    # 2) Build or load the atom-level + token-level features
    atom_feature_dict, token_feature_dict = load_rna_data_and_features(
        "demo_rna_file.cif", device=device, override_num_atoms=partial_coords.shape[1]
    )

    # 3) Fix shape for "deletion_mean" if needed
    if "deletion_mean" in token_feature_dict:
        deletion = token_feature_dict["deletion_mean"]
        expected_tokens = token_feature_dict["restype"].shape[1]
        if deletion.ndim == 2:
            deletion = deletion.unsqueeze(-1)
        if deletion.shape[1] != expected_tokens:
            deletion = deletion[:, :expected_tokens, :]
        atom_feature_dict["deletion_mean"] = deletion

    # 4) Overwrite default coords with partial_coords
    atom_feature_dict["ref_pos"] = partial_coords

    # 5) Merge token-level features so embedder can produce 449-dim
    atom_feature_dict["restype"] = token_feature_dict["restype"]
    atom_feature_dict["profile"] = token_feature_dict["profile"]

    # 6) If trunk_embeddings lacks "s_trunk", fallback to "sing"
    if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
        trunk_embeddings["s_trunk"] = trunk_embeddings.get("sing")

    # 7) Use InputFeatureEmbedder to produce 449-dim single embedding
    embedder = InputFeatureEmbedder(c_atom=128, c_atompair=16, c_token=384)
    s_inputs = embedder(atom_feature_dict, inplace_safe=False, chunk_size=None)

    # 8) Store it in trunk_embeddings so multi_step_inference can find "s_inputs"
    trunk_embeddings["s_inputs"] = s_inputs

    # 9) Handle inference vs. train
    if mode == "inference":
        inference_params = {"num_steps": 20, "sigma_max": 1.0, "N_sample": 1}
        coords_final = manager.multi_step_inference(
            coords_init=partial_coords,
            trunk_embeddings=trunk_embeddings,  # includes "s_inputs"
            inference_params=inference_params,
            override_input_features=atom_feature_dict,
            debug_logging=True,
        )
        return coords_final

    elif mode == "train":
        # Create label_dict for single-step diffusion training
        label_dict = {
            "coordinate": partial_coords,  # ground truth
            "coordinate_mask": torch.ones_like(partial_coords[..., 0]),  # no mask
        }
        sampler_params = {"p_mean": -1.2, "p_std": 1.0, "sigma_data": 16.0}

        # Grab trunk embeddings
        s_trunk = trunk_embeddings["s_trunk"]
        z_trunk = trunk_embeddings.get("pair", None)

        x_gt_out, x_denoised, sigma = manager.train_diffusion_step(
            label_dict=label_dict,
            input_feature_dict=atom_feature_dict,
            s_inputs=s_inputs,  # shape (batch, num_tokens, 449)
            s_trunk=s_trunk,  # shape (batch, num_tokens, 384)
            z_trunk=z_trunk,
            sampler_params=sampler_params,
            N_sample=1,
        )
        loss = (x_denoised - x_gt_out).pow(2).mean()
        return x_denoised, loss, sigma

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def demo_run_diffusion():
    """
    Demonstrates Stage D usage with partial coordinates and trunk embeddings
    for a final global refinement pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FIX: Add c_s_inputs to match the actual shape of s_inputs (384)
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "c_s_inputs": 384,  # This is the key fix - match the actual shape
        "transformer": {"n_blocks": 4, "n_heads": 16},
    }

    # Suppose partial_coords is from StageC
    partial_coords = torch.randn(1, 10, 3, device=device)
    trunk_embeddings = {
        "sing": torch.randn(1, 10, 384, device=device),
        "pair": torch.randn(1, 10, 10, 32, device=device),
    }

    # Inference
    coords_final = run_stageD_diffusion(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="inference",
        device=device,
    )
    print("[Diffusion Demo] coords_final shape:", coords_final.shape)

    # Training
    x_denoised, loss, sigma = run_stageD_diffusion(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="train",
        device=device,
    )
    print("[Diffusion Demo] x_denoised shape:", x_denoised.shape)
    print("[Diffusion Demo] loss:", loss.item())
    print("[Diffusion Demo] sigma:", sigma.item())


if __name__ == "__main__":
    demo_run_diffusion()