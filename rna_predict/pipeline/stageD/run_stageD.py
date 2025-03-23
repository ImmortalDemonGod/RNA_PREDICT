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