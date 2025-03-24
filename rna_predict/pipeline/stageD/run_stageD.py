import torch
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager

def run_stageD_diffusion(
    partial_coords: torch.Tensor,
    trunk_embeddings: dict,
    diffusion_config: dict,
    mode: str = "inference",
    device: str = "cpu"
):
    """
    Stage D entry:
      - partial_coords: [B, N, 3]
      - trunk_embeddings: dict with "sing",[B,N,c_token], "pair",[B,N,N,c_pair], etc.
      - diffusion_config: e.g. { "c_x":3, "c_s":384, "c_z":128, "num_layers":4 }
      - mode: "inference" or "train"
      - device: "cpu" or "cuda"

    If inference => returns final coords
    If train => returns (x_denoised, loss, sigma)
    """
    manager = ProtenixDiffusionManager(diffusion_config, device=device)

    if mode == "inference":
        inference_params = {"num_steps":20, "sigma_max":1.0}
        coords_final = manager.multi_step_inference(
            coords_init=partial_coords,
            trunk_embeddings=trunk_embeddings,
            inference_params=inference_params
        )
        return coords_final

    elif mode == "train":
        sampler_params = {"p_mean": -1.2, "p_std": 1.0}
        x_gt_out, x_denoised, sigma = manager.train_diffusion_step(
            x_gt=partial_coords,
            trunk_embeddings=trunk_embeddings,
            sampler_params=sampler_params
        )
        loss = (x_denoised - x_gt_out).pow(2).mean()
        return x_denoised, loss, sigma

    else:
        raise ValueError(f"Unsupported mode: {mode}")
    

def demo_run_diffusion():
    """
    Demonstrates Stage D diffusion usage with partial coords and trunk embeddings
    for global refinement.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suppose partial_coords is from Stage C, or random
    partial_coords = torch.randn(1, 10, 3, device=device)
    trunk_embeddings = {
        "sing": torch.randn(1, 10, 384, device=device),
        "pair": torch.randn(1, 10, 10, 32, device=device)
    }
    diffusion_config = {
        "c_s": 384,    # single embedding dimension
        "c_z": 32,     # pair embedding dimension
        "num_layers": 4
    }

    # 1) Inference mode
    coords_final = run_stageD_diffusion(
        partial_coords,
        trunk_embeddings,
        diffusion_config,
        mode="inference",
        device=device
    )
    print("[Diffusion Demo] coords_final shape:", coords_final.shape)

    # 2) Training mode
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
# Expected output:
# [Diffusion Demo] coords_final shape: torch.Size([1, 10, 3])
# [Diffusion Demo] x_denoised shape: torch.Size([1, 10, 3])
# [Diffusion Demo] loss: 1.0
# [Diffusion Demo] sigma: 1.0