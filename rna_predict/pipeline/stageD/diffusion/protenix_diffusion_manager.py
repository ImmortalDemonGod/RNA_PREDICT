import warnings
import logging
from typing import Dict, Optional, Any
from omegaconf import OmegaConf, DictConfig
import torch

from rna_predict.pipeline.stageD.diffusion.components.diffusion_module import (
    DiffusionModule,
)
from rna_predict.pipeline.stageD.diffusion.generator import (
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_training,
)

# Initialize logger for Stage D diffusion manager
logger = logging.getLogger("rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager")


class ProtenixDiffusionManager:
    """
    Manager that handles training steps or multi-step inference for diffusion.
    Expects trunk_embeddings to contain:
      - "s_trunk"
      - optionally "s_inputs"
      - "pair"

    Uses standard Hydra configuration (DictConfig) with stageD.diffusion group.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the Diffusion Manager using Hydra configuration.
        Reads all parameters from cfg.stageD.diffusion group.

        Args:
            cfg: Hydra configuration object. Expected structure:
                 stageD:
                   diffusion:
                     device: "cpu"
                     # DiffusionModule args
                     c_atom: 128
                     transformer: { ... }
                     atom_encoder: { ... }
                     atom_decoder: { ... }
                     sigma_data: 16.0
                     # Noise schedule args
                     s_max: 160.0
                     # Sampler args
                     sampler: {p_mean: -1.2, ...}
                     # Inference args
                     inference: {num_steps: 50, sampling: {num_samples: 1}, ...}
        """
        if not OmegaConf.is_config(cfg):
            raise ValueError("Config must be a Hydra DictConfig")

        if "stageD" not in cfg:
            raise ValueError("Config missing required 'stageD' group")

        if "diffusion" not in cfg.stageD:
            raise ValueError("Config missing required 'diffusion' group in stageD")

        self.cfg = cfg
        stage_cfg = cfg.stageD.diffusion
        self.device = torch.device(stage_cfg.device)

        # Extract inference parameters for test compatibility
        inference_cfg = stage_cfg.get("inference", OmegaConf.create({}))
        self.num_inference_steps = inference_cfg.get("num_steps", 2)
        self.temperature = inference_cfg.get("temperature", 1.0)

        # Extract arguments for DiffusionModule
        expected_keys = [
            "sigma_data", "c_atom", "c_atompair", "c_token", "c_s", "c_z",
            "c_s_inputs", "c_noise_embedding", "atom_encoder", "transformer",
            "atom_decoder", "blocks_per_ckpt", "use_fine_grained_checkpoint",
            "initialization"
        ]

        diffusion_module_args = {}
        for key in expected_keys:
            value = stage_cfg.get(key, None)
            if value is not None:
                diffusion_module_args[key] = value

        # Handle memory configuration
        mem_cfg = stage_cfg.get("memory", OmegaConf.create({}))
        use_ckpt = mem_cfg.get("use_checkpointing", False)
        blocks_per_ckpt = mem_cfg.get("blocks_per_ckpt") if use_ckpt else None
        use_fine_grained = mem_cfg.get("use_fine_grained_checkpoint", False) if blocks_per_ckpt else False

        diffusion_module_args.update({
            "blocks_per_ckpt": blocks_per_ckpt,
            "use_fine_grained_checkpoint": use_fine_grained
        })

        debug_logging = self.cfg.stageD.diffusion.get("debug_logging", False)
        if debug_logging:
            logger.debug(f"Initializing ProtenixDiffusionManager with config: {self.cfg.stageD.diffusion}")

        try:
            self.diffusion_module = DiffusionModule(**diffusion_module_args).to(self.device)
        except TypeError as e:
            logger.error(f"Error initializing DiffusionModule: {e}")
            logger.error(f"Config provided: {diffusion_module_args}")
            raise

    # #@snoop
    def train_diffusion_step(
        self,
        label_dict: Dict[str, torch.Tensor],
        input_feature_dict: Dict[str, torch.Tensor],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        # Remove backward compatibility args, rely solely on self.cfg
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a single-step training pass with random noise injection.
        Sampler parameters are read from the stored Hydra config (self.cfg.stageD.diffusion.sampler).
        """
        if not OmegaConf.is_config(self.cfg): # Check if running in Hydra mode
             raise RuntimeError("train_diffusion_step requires Hydra config (cfg) for parameters.")

        stage_cfg = self.cfg.stageD.diffusion
        # Use .get for sampler params with defaults if section/keys might be missing
        sampler_cfg = stage_cfg.get("sampler", OmegaConf.create({})) # Default to empty if missing

        # Get specific sampler params, falling back to defaults if not in config
        p_mean = sampler_cfg.get("p_mean", -1.2)
        p_std = sampler_cfg.get("p_std", 1.5)
        # sigma_data now expected under stage_cfg directly
        sigma_data = stage_cfg.get("sigma_data", 16.0)
        N_sample = sampler_cfg.get("N_sample", 1)
        diffusion_chunk_size = stage_cfg.get("diffusion_chunk_size") # Can be None


        noise_sampler = TrainingNoiseSampler(
            p_mean=p_mean,
            p_std=p_std,
            sigma_data=sigma_data,
        )

        x_gt_augment, x_denoised, sigma = sample_diffusion_training(
            noise_sampler=noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            N_sample=N_sample, # Pass value derived from config
            diffusion_chunk_size=diffusion_chunk_size, # Pass value derived from config
        )
        return x_gt_augment, x_denoised, sigma

    #@snoop
    def multi_step_inference(
        self,
        coords_init: torch.Tensor,
        trunk_embeddings: Dict[str, Any],
        override_input_features: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Multi-step diffusion-based inference. Handles shape expansions.
        Inference parameters are read from the stored Hydra config (self.cfg.stageD.diffusion).
        """
        if not OmegaConf.is_config(self.cfg):
            raise ValueError("multi_step_inference requires Hydra config")

        stage_cfg = self.cfg.stageD.diffusion
        inference_cfg = stage_cfg.get("inference", OmegaConf.create({}))
        noise_schedule_cfg = stage_cfg.get("noise_schedule", OmegaConf.create({}))
        debug_logging = stage_cfg.get("debug_logging", False)

        device = self.device
        coords_init = coords_init.to(device)

        # Move trunk embeddings to device
        processed_trunk_embeddings = {}
        for k, v in trunk_embeddings.items():
            if isinstance(v, torch.Tensor):
                processed_trunk_embeddings[k] = v.to(device)
            else:
                processed_trunk_embeddings[k] = v
        trunk_embeddings = processed_trunk_embeddings

        # Must have s_trunk
        if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
            raise ValueError("StageD multi_step_inference requires a valid 's_trunk' in trunk_embeddings.")

        # Get s_inputs from trunk_embeddings
        s_inputs = trunk_embeddings.get("s_inputs")
        if s_inputs is None and override_input_features is not None:
            s_inputs = override_input_features.get("s_inputs")

        # Create fallback s_inputs if not available
        if s_inputs is None:
            logger.warning("'s_inputs' not found in trunk_embeddings or override_input_features. Creating fallback.")
            # Get dimensions from s_trunk
            s_trunk_shape = trunk_embeddings["s_trunk"].shape
            batch_size = s_trunk_shape[0]
            n_tokens = s_trunk_shape[1]
            # Default dimension for s_inputs is 449, but can be configured
            c_s_inputs_dim = stage_cfg.get("c_s_inputs", 449)
            # Create zero tensor with appropriate shape
            s_inputs = torch.zeros((batch_size, n_tokens, c_s_inputs_dim), device=device)

        # Get z_trunk or create a fallback
        z_trunk = trunk_embeddings.get("pair")
        if z_trunk is None:
            logger.warning("'pair' embedding not found in trunk_embeddings. Creating fallback.")
            # Get dimensions from s_trunk
            s_trunk_shape = trunk_embeddings["s_trunk"].shape
            batch_size = s_trunk_shape[0]
            n_tokens = s_trunk_shape[1]
            # Default dimension for z_trunk is 128, but can be configured
            c_z_dim = stage_cfg.get("c_z", 128)
            # Create zero tensor with appropriate shape
            z_trunk = torch.zeros((batch_size, n_tokens, n_tokens, c_z_dim), device=device)

        # Validate tensor shapes
        if z_trunk.shape[1] != trunk_embeddings["s_trunk"].shape[1] or z_trunk.shape[2] != trunk_embeddings["s_trunk"].shape[1]:
            raise RuntimeError(f"shape mismatch between z_trunk {z_trunk.shape} and s_trunk {trunk_embeddings['s_trunk'].shape}")

        if debug_logging:
            logger.debug(f"[multi_step_inference] coords_init shape: {coords_init.shape}")
            logger.debug(f"[multi_step_inference] s_trunk shape: {trunk_embeddings['s_trunk'].shape}")
            if isinstance(s_inputs, torch.Tensor):
                logger.debug(f"[multi_step_inference] s_inputs shape: {s_inputs.shape}")
            if z_trunk is not None:
                logger.debug(f"[multi_step_inference] z_trunk shape: {z_trunk.shape}")

        # Prepare input features
        if override_input_features is not None:
            input_feature_dict = override_input_features
        else:
            n_atoms = coords_init.shape[1]
            batch_size = coords_init.shape[0]
            input_feature_dict = {
                "atom_to_token_idx": torch.arange(n_atoms, device=device).long().unsqueeze(0).expand(batch_size, -1)
            }

        # Get sampling parameters
        sampling_cfg = inference_cfg.get("sampling", OmegaConf.create({}))
        N_sample = sampling_cfg.get("num_samples", 1)

        # Shape unification for atom_idx
        if "atom_to_token_idx" in input_feature_dict:
            atom_idx = input_feature_dict["atom_to_token_idx"]
            if (atom_idx.dim() == 2 and trunk_embeddings["s_trunk"].dim() == 4 and
                    trunk_embeddings["s_trunk"].shape[1] == 1):
                atom_idx = atom_idx.unsqueeze(1)
            elif (atom_idx.dim() == 2 and s_inputs is not None and
                    s_inputs.dim() == 4 and s_inputs.shape[1] == 1):
                atom_idx = atom_idx.unsqueeze(1)
            input_feature_dict["atom_to_token_idx"] = atom_idx

        # Get noise schedule
        num_steps = inference_cfg.get("num_steps", 50)
        schedule_type = noise_schedule_cfg.get("schedule_type", "linear")
        if schedule_type == "linear":
            noise_schedule = torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)
        else:
            warnings.warn(f"Using default linear noise schedule. Unknown type: {schedule_type}")
            noise_schedule = torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)

        # Get inference parameters
        inplace_safe = inference_cfg.get("inplace_safe", False)
        attn_chunk_size = stage_cfg.get("attn_chunk_size")

        if debug_logging:
            logger.debug("[multi_step_inference] Before sample_diffusion:")
            logger.debug(f"  coords_init shape: {coords_init.shape}")
            logger.debug(f"  s_trunk shape: {trunk_embeddings['s_trunk'].shape}")

        coords_final = sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=trunk_embeddings["s_trunk"],
            z_trunk=z_trunk,
            noise_schedule=noise_schedule,
            N_sample=N_sample,
            inplace_safe=inplace_safe,
            attn_chunk_size=attn_chunk_size,
        )

        if debug_logging:
            logger.debug(f"[multi_step_inference] coords_final shape: {coords_final.shape}")

        # Handle N_sample=1 case
        if N_sample == 1:
            n_atoms_inferred = coords_final.shape[2] if coords_final.ndim > 2 else -1
            if (coords_final.ndim == 4 and coords_final.shape[1] == n_atoms_inferred and
                    coords_final.shape[2] == n_atoms_inferred):
                warnings.warn(f"Unexpected shape {coords_final.shape} for N_sample=1")
                coords_final = coords_final[:, 0, :, :]
            else:
                sample_dim_index = 1
                if (coords_final.ndim > sample_dim_index and
                        coords_final.shape[sample_dim_index] == 1):
                    coords_final = coords_final.squeeze(sample_dim_index)

        if debug_logging:
            logger.debug(f"[multi_step_inference] Final coords shape: {coords_final.shape}")

        return coords_final

    # custom_manual_loop remains largely unchanged as it calls diffusion_module directly
    def custom_manual_loop(
        self, x_gt: torch.Tensor, trunk_embeddings: dict, sigma: float
    ):
        """
        Optional direct usage demonstration: single forward pass
        """
        x_gt = x_gt.to(self.device)
        x_noisy = x_gt + torch.randn_like(x_gt) * sigma
        # Ensure diffusion_module exists (it should if __init__ ran correctly)
        if not hasattr(self, 'diffusion_module'):
             raise RuntimeError("Diffusion module not initialized. Call __init__ first.")

        x_denoised = self.diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=torch.tensor([sigma], device=self.device),
            input_feature_dict={}, # Provide minimal dict
            s_inputs=trunk_embeddings.get("s_inputs"),
            s_trunk=trunk_embeddings.get("s_trunk"),
            z_trunk=trunk_embeddings.get("pair"),
        )
        return x_noisy, x_denoised
