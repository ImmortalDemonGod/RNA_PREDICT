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
    
    @snoop
    def multi_step_inference(self, coords_init: torch.Tensor, trunk_embeddings: dict, inference_params: dict):
        """
        Adjust this method so we pass valid arguments to sample_diffusion.
        We do not currently use coords_init as the initial coordinate inside sample_diffusion,
        because sample_diffusion starts from random noise. If you want to incorporate
        coords_init as a real starting point, you'd need to modify sample_diffusion's logic.
        """
        device = self.device
        coords_init = coords_init.to(device)

        # Move trunk embeddings to the correct device
        for k, v in trunk_embeddings.items():
            trunk_embeddings[k] = v.to(device) if v is not None else None

        # Construct a minimal input_feature_dict for sample_diffusion
        input_feature_dict = {
            "atom_to_token_idx": torch.zeros((1, 0), device=device),
            # Additional fields or real token mapping can be placed here
        }

        # For demonstration, pick s_inputs, s_trunk, and z_trunk from trunk_embeddings
        s_inputs = trunk_embeddings.get("sing", torch.empty((1, 0), device=device))
        s_trunk = trunk_embeddings.get("s_trunk", torch.empty((1, 0), device=device))
        z_trunk = trunk_embeddings.get("pair", torch.empty((1, 1, 0), device=device))

        # Add check for empty s_trunk dimension
        if s_trunk.size(1) == 0:
            raise ValueError(f"StageD diffusion requires a non-empty 's_trunk' in trunk_embeddings. Received s_trunk with shape: {s_trunk.shape}")

        # Create a simple linear noise schedule or pull from InferenceNoiseScheduler
        num_steps = inference_params.get("num_steps", 20)
        noise_schedule = torch.linspace(1.0, 0.0, steps=num_steps+1, device=device)

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