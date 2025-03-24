import torch
from protenix.model.modules.diffusion import DiffusionModule  # type: ignore
from protenix.model.generator import (  # type: ignore
    sample_diffusion_training,
    sample_diffusion,
    TrainingNoiseSampler
)

class ProtenixDiffusionManager:
    """
    Wraps the Protenix DiffusionModule, providing single-step (train) or
    multi-step (inference) bridging for Stage D in the pipeline.
    """

    def __init__(self, diffusion_config: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.diffusion_module = DiffusionModule(**diffusion_config).to(self.device)

    def train_diffusion_step(
        self,
        label_dict: dict,
        input_feature_dict: dict,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        sampler_params: dict,
        N_sample: int = 1,
        diffusion_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fix to pass all required arguments to sample_diffusion_training.
        We create the noise_sampler from sampler_params and treat self.diffusion_module
        as the denoise_net. Then we call sample_diffusion_training with the correct signature.
        Returns x_gt_augment, x_denoised, sigma.
        """

        noise_sampler = TrainingNoiseSampler(
            p_mean=sampler_params.get("p_mean", -1.2),
            p_std=sampler_params.get("p_std", 1.5),
            sigma_data=sampler_params.get("sigma_data", 16.0),
        )
        x_gt_augment, x_denoised, sigma = sample_diffusion_training(
            noise_sampler=noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            N_sample=N_sample,
            diffusion_chunk_size=diffusion_chunk_size,
        )
        return x_gt_augment, x_denoised, sigma
    
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