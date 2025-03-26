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
    Manager that handles training steps or multi-step inference for diffusion.
    Expects trunk_embeddings to contain:
      - "s_trunk"
      - optionally "s_inputs"
      - "pair"
    """

    @snoop
    def __init__(self, diffusion_config: dict, device: str = "cpu"):
        # Ensure we have an "initialization" key
        if "initialization" not in diffusion_config or diffusion_config["initialization"] is None:
            diffusion_config["initialization"] = {}

        self.device = torch.device(device)
        self.diffusion_module = DiffusionModule(**diffusion_config).to(self.device)

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
        diffusion_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a single-step training pass with random noise injection.
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
        Multi-step diffusion-based inference with fallback logic for s_inputs
        and dimension expansion for multi-sample.
        """
        device = self.device
        coords_init = coords_init.to(device)

        # Ensure trunk embeddings are on device
        for k, v in trunk_embeddings.items():
            if isinstance(v, torch.Tensor):
                trunk_embeddings[k] = v.to(device)

        # We require "s_trunk"
        if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
            available = {k: (v.shape if isinstance(v, torch.Tensor) else v)
                         for k, v in trunk_embeddings.items()}
            raise ValueError(
                f"StageD multi_step_inference requires a valid 's_trunk'. Found: {available}"
            )

        # Attempt to get s_inputs
        s_inputs = trunk_embeddings.get("s_inputs", trunk_embeddings.get("sing"))
        if not isinstance(s_inputs, torch.Tensor):
            if override_input_features is not None:
                from rna_predict.pipeline.stageA.input_embedding.current.embedders import InputFeatureEmbedder
                embedder = InputFeatureEmbedder(c_atom=128, c_atompair=16, c_token=384)
                s_inputs = embedder(override_input_features, inplace_safe=False, chunk_size=None)
            else:
                raise ValueError(
                    "No valid 's_inputs' found in trunk_embeddings. "
                    "Please supply the 449-dim InputFeatureEmbedder output under 's_inputs' or 'sing', "
                    "or provide override_input_features so we can build it."
                )

        z_trunk = trunk_embeddings.get("pair", None)

        if debug_logging:
            print(f"[DEBUG] s_trunk shape: {trunk_embeddings['s_trunk'].shape}")
            if isinstance(s_inputs, torch.Tensor):
                print(f"[DEBUG] s_inputs shape: {s_inputs.shape}")
            else:
                print("[DEBUG] s_inputs is invalid or None!")
            if z_trunk is not None:
                print(f"[DEBUG] z_trunk shape: {z_trunk.shape}")

        # Prepare the input features
        if override_input_features is not None:
            input_feature_dict = override_input_features
        else:
            # minimal fallback if none provided
            input_feature_dict = {"atom_to_token_idx": torch.zeros((1, 0), device=device)}

        N_sample = inference_params.get("N_sample", 1)
        # Expand shape logic
        if "atom_to_token_idx" in input_feature_dict:
            atom_idx = input_feature_dict["atom_to_token_idx"]
            # If shape is [B, N_atom], we want [B, N_sample, N_atom] if multi-sample > 1
            if atom_idx.dim() == 1:
                atom_idx = atom_idx.unsqueeze(0)
            B = atom_idx.shape[0]
            if atom_idx.dim() == 2 and N_sample > 1:
                atom_idx = atom_idx.unsqueeze(1).expand(B, N_sample, atom_idx.shape[-1])
            input_feature_dict["atom_to_token_idx"] = atom_idx

        if N_sample > 1:
            # s_trunk => [B, N_token, c_s] -> [B, N_sample, N_token, c_s]
            if trunk_embeddings["s_trunk"].dim() == 3:
                trunk_embeddings["s_trunk"] = trunk_embeddings["s_trunk"].unsqueeze(1)\
                                              .expand(-1, N_sample, -1, -1)

            # s_inputs => [B, N_token, 449] -> [B, N_sample, N_token, 449]
            if isinstance(s_inputs, torch.Tensor) and s_inputs.dim() == 3:
                s_inputs = s_inputs.unsqueeze(1).expand(-1, N_sample, -1, -1)

            # z_trunk => [B, N_token, N_token, c_z] -> [B, N_sample, N_token, N_token, c_z]
            if z_trunk is not None and z_trunk.dim() == 4:
                z_trunk = z_trunk.unsqueeze(1).expand(-1, N_sample, -1, -1, -1)

            # coords_init => [B, N_atom, 3] -> [B, N_sample, N_atom, 3]
            if coords_init.dim() == 3:
                coords_init = coords_init.unsqueeze(1).expand(-1, N_sample, -1, -1)

            # Also expand ref_pos in input_feature_dict if shape [B, N_atom, 3]
            if "ref_pos" in input_feature_dict:
                rp = input_feature_dict["ref_pos"]
                if rp.dim() == 3:
                    input_feature_dict["ref_pos"] = rp.unsqueeze(1).expand(-1, N_sample, -1, -1)

        # Overwrite updated references
        trunk_embeddings["s_inputs"] = s_inputs
        trunk_embeddings["pair"] = z_trunk

        # Build a linear noise schedule
        num_steps = inference_params.get("num_steps", 20)
        noise_schedule = torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)

        coords_final = sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=trunk_embeddings["s_trunk"],
            z_trunk=z_trunk,
            noise_schedule=noise_schedule,
            N_sample=N_sample,
        )

        return coords_final

    def custom_manual_loop(self, x_gt: torch.Tensor, trunk_embeddings: dict, sigma: float):
        """
        Optional direct usage demonstration: single forward pass
        """
        x_gt = x_gt.to(self.device)
        x_noisy = x_gt + torch.randn_like(x_gt) * sigma
        x_denoised = self.diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=torch.tensor([sigma], device=self.device),
            input_feature_dict={},
            s_inputs=trunk_embeddings.get("s_inputs"),
            s_trunk=trunk_embeddings.get("s_trunk"),
            z_trunk=trunk_embeddings.get("pair"),
        )
        return x_noisy, x_denoised