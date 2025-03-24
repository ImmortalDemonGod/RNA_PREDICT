import torch
from protenix.model.modules.diffusion import DiffusionModule  # type: ignore
from protenix.model.generator import (  # type: ignore
    sample_diffusion_training,
    sample_diffusion,
    TrainingNoiseSampler
)
import snoop

class ProtenixDiffusionManager:
    """
    Wraps the Protenix DiffusionModule, providing single-step (train) or
    multi-step (inference) bridging for Stage D in the pipeline.
    """

    @snoop
    def __init__(self, diffusion_config: dict, device: str = "cpu"):
        # Ensure "initialization" is never None
        if "initialization" not in diffusion_config or diffusion_config["initialization"] is None:
            diffusion_config["initialization"] = {}

        self.device = torch.device(device)
        self.diffusion_module = DiffusionModule(**diffusion_config).to(self.device)

    @snoop
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
        Pass all required arguments to sample_diffusion_training.
        Creates a noise sampler from sampler_params and uses self.diffusion_module as the denoise_net.
        Returns: x_gt_augment, x_denoised, sigma.
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
        override_input_features: dict = None
    ):
        """
        If override_input_features is provided, use that dictionary instead of a dummy one.
        This prevents the 'index out of bounds' error when using an empty input_feature_dict.
        """
        device = self.device
        coords_init = coords_init.to(device)

        # Move trunk embeddings to the correct device
        for k, v in trunk_embeddings.items():
            trunk_embeddings[k] = v.to(device) if v is not None else None

        # Diagnostic: Check that 's_trunk' exists and is non-empty
        if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
            available = {k: (v.shape if isinstance(v, torch.Tensor) else v) for k, v in trunk_embeddings.items()}
            raise ValueError(f"StageD diffusion requires a non-empty 's_trunk' in trunk_embeddings, but it was not found. Available keys: {available}")

        # Use the provided input_feature_dict if available; otherwise fallback to a dummy dict
        if override_input_features is not None:
            input_feature_dict = override_input_features
        else:
            input_feature_dict = {
                "atom_to_token_idx": torch.zeros((1, 0), device=device)
            }

        s_inputs = trunk_embeddings.get("sing", trunk_embeddings.get("s_inputs", None))
        s_trunk = trunk_embeddings["s_trunk"]
        z_trunk = trunk_embeddings.get("pair", None)

        # Create a simple noise schedule
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
            N_sample=N_sample
            # Optionally pass other advanced arguments if needed
        )

        return coords_final

    def custom_manual_loop(self, x_gt: torch.Tensor, trunk_embeddings: dict, sigma: float):
        """
        Optional manual approach: user controls noise and calls diffusion_module forward.
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