import torch
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.generator import (
    sample_diffusion_training,
    sample_diffusion
)

class ProtenixDiffusionManager:
    """
    Wraps the Protenix DiffusionModule, providing single-step (train) or
    multi-step (inference) bridging for Stage D in the pipeline.
    """

    def __init__(self, diffusion_config: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.diffusion_module = DiffusionModule(**diffusion_config).to(self.device)

    def train_diffusion_step(self, x_gt: torch.Tensor, trunk_embeddings: dict, sampler_params: dict):
        """
        Single-step training using sample_diffusion_training.
        x_gt: (B, N, 3)
        trunk_embeddings: e.g. { "sing": [B,N,c_token], "pair": [B,N,N,c_pair] }
        sampler_params: e.g. {"p_mean": -1.2, "p_std":1.0}

        Returns: (x_gt_out, x_denoised, sigma)
        """
        x_gt = x_gt.to(self.device)
        for k, v in trunk_embeddings.items():
            trunk_embeddings[k] = v.to(self.device) if v is not None else None

        x_gt_out, x_denoised, sigma = sample_diffusion_training(
            x_gt=x_gt,
            trunk_sing=trunk_embeddings.get("sing", None),
            trunk_pair=trunk_embeddings.get("pair", None),
            diffusion_module=self.diffusion_module,
            noise_sampler_params=sampler_params,
            device=self.device
        )
        return x_gt_out, x_denoised, sigma

    def multi_step_inference(self, coords_init: torch.Tensor, trunk_embeddings: dict, inference_params: dict):
        """
        multi-step inference using sample_diffusion.
        coords_init: [B, N, 3]
        trunk_embeddings: e.g. {"sing": [B,N,c_token], "pair": [B,N,N,c_pair]}
        inference_params: e.g. {"num_steps":20,"sigma_max":1.0}

        Returns final denoised coords: [B, N, 3]
        """
        coords_init = coords_init.to(self.device)
        for k, v in trunk_embeddings.items():
            trunk_embeddings[k] = v.to(self.device) if v is not None else None

        coords_final = sample_diffusion(
            x_init=coords_init,
            trunk_sing=trunk_embeddings.get("sing", None),
            trunk_pair=trunk_embeddings.get("pair", None),
            diffusion_module=self.diffusion_module,
            device=self.device,
            **inference_params
        )
        return coords_final

    def custom_manual_loop(self, x_gt: torch.Tensor, trunk_embeddings: dict, sigma: float):
        """
        Optional manual approach: user controls noise, calls diffusion_module forward
        """
        x_gt = x_gt.to(self.device)
        x_noisy = x_gt + torch.randn_like(x_gt) * sigma
        x_denoised = self.diffusion_module(
            x_noisy,
            sigma=sigma,
            trunk_sing=trunk_embeddings.get("sing", None),
            trunk_pair=trunk_embeddings.get("pair", None)
        )
        return x_noisy, x_denoised