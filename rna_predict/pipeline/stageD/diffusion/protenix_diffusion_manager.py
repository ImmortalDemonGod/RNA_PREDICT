from typing import Dict, Optional

import torch

from rna_predict.pipeline.stageD.diffusion.components.diffusion_module import DiffusionModule # Updated import path
from rna_predict.pipeline.stageD.diffusion.generator import (
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_training,
)


class ProtenixDiffusionManager:
    """
    Manager that handles training steps or multi-step inference for diffusion.
    Expects trunk_embeddings to contain:
      - "s_trunk"
      - optionally "s_inputs"
      - "pair"
    """

    # @snoop
    def __init__(self, diffusion_config: dict, device: str = "cpu"):
        # Ensure we have an "initialization" key
        if (
            "initialization" not in diffusion_config
            or diffusion_config["initialization"] is None
        ):
            diffusion_config["initialization"] = {}

        self.device = torch.device(device)
        self.diffusion_module = DiffusionModule(**diffusion_config).to(self.device)

    # @snoop
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

    # @snoop
    def multi_step_inference(
        self,
        coords_init: torch.Tensor,
        trunk_embeddings: dict,
        inference_params: dict,
        override_input_features: dict | None = None,
        debug_logging: bool = False,
    ) -> torch.Tensor:
        """
        Multi-step diffusion-based inference. Handles shape expansions for single or multiple samples.

        Typical usage:
        - coords_init: [B, N_atom, 3] or [B, 1, N_atom, 3]
        - trunk_embeddings: must contain 's_trunk'.
          optional: 's_inputs' (else fallback to 'sing')
                    'pair'
        - override_input_features: if needed to build or unify shape for multi-sample expansions
        - inference_params: includes "N_sample" (# samples) and "num_steps" for noise schedule
        """
        device = self.device
        coords_init = coords_init.to(device)

        # Move trunk embeddings to device
        for k, v in trunk_embeddings.items():
            if isinstance(v, torch.Tensor):
                trunk_embeddings[k] = v.to(device)

        # Must have s_trunk
        if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
            raise ValueError(
                "StageD multi_step_inference requires a valid 's_trunk' in trunk_embeddings."
            )

        # Attempt to get s_inputs or fallback
        s_inputs = trunk_embeddings.get("s_inputs", trunk_embeddings.get("sing"))
        z_trunk = trunk_embeddings.get("pair", None)

        if debug_logging:
            print(f"[DEBUG] s_trunk shape: {trunk_embeddings['s_trunk'].shape}")
            if isinstance(s_inputs, torch.Tensor):
                print(f"[DEBUG] s_inputs shape: {s_inputs.shape}")
            else:
                print("[DEBUG] s_inputs is invalid or None!")
            if z_trunk is not None:
                print(f"[DEBUG] z_trunk shape: {z_trunk.shape}")

        # Prepare the input features (atom_to_token_idx, ref_pos, etc.)
        if override_input_features is not None:
            input_feature_dict = override_input_features
        else:
            # minimal fallback if none provided
            input_feature_dict = {
                "atom_to_token_idx": torch.zeros((1, 0), device=device)
            }

        # Determine how many samples we want
        N_sample = inference_params.get("N_sample", 1)

        # Possibly unify shapes in atom_to_token_idx if we forcibly unsqueeze s_trunk for single-sample
        # e.g. if s_trunk is [B,1,N_token,c_s], we want atom_idx => [B,1,N_atom]
        if "atom_to_token_idx" in input_feature_dict:
            atom_idx = input_feature_dict["atom_to_token_idx"]
            # example check: if trunk is 4D with trunk_embeddings["s_trunk"].shape[1] == 1
            # then unify shape of atom_idx accordingly
            if (
                atom_idx.dim() == 2
                and trunk_embeddings["s_trunk"].dim() == 4
                and trunk_embeddings["s_trunk"].shape[1] == 1
            ):
                atom_idx = atom_idx.unsqueeze(1)  # => [B,1,N_atom]

            elif (
                atom_idx.dim() == 2
                and s_inputs is not None
                and s_inputs.dim() == 4
                and s_inputs.shape[1] == 1
            ):
                atom_idx = atom_idx.unsqueeze(1)

            input_feature_dict["atom_to_token_idx"] = atom_idx

        # If we truly want multiple samples (N_sample>1), expand shapes further
        if N_sample > 1:
            # Expand s_trunk => [B,N_sample,N_token,c_s]
            st = trunk_embeddings["s_trunk"]
            if st.dim() == 3:
                trunk_embeddings["s_trunk"] = st.unsqueeze(1).expand(
                    -1, N_sample, -1, -1
                )

            # Expand s_inputs => [B,N_sample,N_token,449]
            if isinstance(s_inputs, torch.Tensor) and s_inputs.dim() == 3:
                s_inputs = s_inputs.unsqueeze(1).expand(-1, N_sample, -1, -1)

            # Expand pair => [B,N_sample,N_token,N_token,c_z]
            if z_trunk is not None and z_trunk.dim() == 4:
                z_trunk = z_trunk.unsqueeze(1).expand(-1, N_sample, -1, -1, -1)

            # Expand coords_init => [B,N_sample,N_atom,3]
            if coords_init.dim() == 3:
                coords_init = coords_init.unsqueeze(1).expand(-1, N_sample, -1, -1)

        # Overwrite updated references
        trunk_embeddings["s_inputs"] = s_inputs
        trunk_embeddings["pair"] = z_trunk

        # Build a simple linear noise schedule from 1.0 down to 0.0
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
            inplace_safe=False,  # or True if memory is tight
            attn_chunk_size=None,  # you can set chunk sizes if needed
        )

        # Remove extra dimension when N_sample=1 to match expected shape [B, N_atom, 3]
        if coords_final.ndim == 4 and coords_final.shape[1] == 1:
            coords_final = coords_final.squeeze(1)

        return coords_final

    def custom_manual_loop(
        self, x_gt: torch.Tensor, trunk_embeddings: dict, sigma: float
    ):
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
