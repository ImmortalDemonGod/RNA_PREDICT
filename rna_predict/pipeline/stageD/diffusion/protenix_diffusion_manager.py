import torch
import torch.nn as nn
from typing import Optional, Dict
import snoop

from rna_predict.pipeline.stageD.diffusion.diffusion import DiffusionModule
from rna_predict.pipeline.stageD.diffusion.generator import (
    sample_diffusion_training,
    sample_diffusion,
    TrainingNoiseSampler,
)


class ProtenixDiffusionManager:
    """
    Updated manager that no longer uses bridging layers;
    s_trunk and s_inputs must have correct dimensions upstream.
    """

    @snoop
    def __init__(self, diffusion_config: dict, device: str = "cpu"):
        # Ensure "initialization" is never None
        if "initialization" not in diffusion_config or diffusion_config["initialization"] is None:
            diffusion_config["initialization"] = {}

        self.device = torch.device(device)

        # No more forced check for c_token in [832, 833].
        # If needed, we can store c_token from config, but do not enforce dimension.
        # e.g. self.c_token = diffusion_config.get("c_token", 768)

        # Build the diffusion module
        self.diffusion_module = DiffusionModule(**diffusion_config).to(self.device)

        # Remove bridging layers entirely
        # self.trunk_bridge_s = None
        # self.trunk_bridge_i = None

    @snoop
    def train_diffusion_step(
        self,
        label_dict: Dict[str, torch.Tensor],
        input_feature_dict: Dict[str, torch.Tensor],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        sampler_params: dict,
        N_sample: int = 1,
        diffusion_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step diffusion used in training mode.
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

    @snoop
    def multi_step_inference(
        self,
        coords_init: torch.Tensor,
        trunk_embeddings: dict,
        inference_params: dict,
        override_input_features: dict = None,
        debug_logging: bool = False,
    ):
        """
        Multi-step diffusion sampling with no bridging.
        We rely on s_trunk.shape[-1] + s_inputs.shape[-1]
        matching the sum of c_s and c_s_inputs in the diffusion code (e.g. 384 + 449 = 833).
        """
        device = self.device
        coords_init = coords_init.to(device)

        # Move trunk embeddings to device
        for k, v in trunk_embeddings.items():
            if v is not None:
                trunk_embeddings[k] = v.to(device)

        # Must have valid s_trunk
        if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
            available = {
                k: (v.shape if isinstance(v, torch.Tensor) else v)
                for k, v in trunk_embeddings.items()
            }
            raise ValueError(
                f"StageD diffusion requires a non-empty 's_trunk'. Found only: {available}"
            )

        # Possibly override
        if override_input_features is not None:
            input_feature_dict = override_input_features
        else:
            input_feature_dict = {
                "atom_to_token_idx": torch.zeros((1, 0), device=device)
            }
        # Instead of falling back to s_trunk, we now require a valid 's_inputs' tensor:
        s_inputs = trunk_embeddings.get("s_inputs", trunk_embeddings.get("sing"))
        if not isinstance(s_inputs, torch.Tensor):
            raise ValueError(
                "No valid 's_inputs' found in trunk_embeddings. "
                "Please supply the 449-dim InputFeatureEmbedder output under 's_inputs' or 'sing'."
            )
        z_trunk = trunk_embeddings.get("pair", None)

        # If no separate s_inputs, fallback to s_trunk
        if s_inputs is None:
            s_inputs = s_trunk

        if debug_logging:
            print(f"[DEBUG] s_trunk shape: {s_trunk.shape}")
            print(f"[DEBUG] s_inputs shape: {s_inputs.shape}")

        # Create noise schedule
        num_steps = inference_params.get("num_steps", 20)
        noise_schedule = torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)
        N_sample = inference_params.get("N_sample", 1)

        coords_final = sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            noise_schedule=noise_schedule,
            N_sample=N_sample,
        )

        return coords_final

    def custom_manual_loop(self, x_gt: torch.Tensor, trunk_embeddings: dict, sigma: float):
        """
        Optional function showing manual usage of the diffusion module directly.
        """
        x_gt = x_gt.to(self.device)
        x_noisy = x_gt + torch.randn_like(x_gt) * sigma
        x_denoised = self.diffusion_module(
            x_noisy,
            sigma=sigma,
            trunk_sing=trunk_embeddings.get("sing", None),
            trunk_pair=trunk_embeddings.get("pair", None),
        )
        return x_noisy, x_denoised