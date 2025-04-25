# RNA_PREDICT Stage D Diffusion Manager
# -------------------------------------------------------
# All configuration for Stage D must be accessed via Hydra config groups (cfg.model.stageD and subgroups).
# No ad-hoc or duplicate configs allowed. See defaults.yaml for config group references.
# Strict Hydra best practices enforced project-wide.

import logging
import os
import warnings
from typing import Any, Dict, Optional
import psutil
import snoop
import torch
from omegaconf import DictConfig, OmegaConf

from rna_predict.pipeline.stageD.diffusion.components.diffusion_module import (
    DiffusionModule,
)
from rna_predict.pipeline.stageD.diffusion.generator import (
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_training,
)
from rna_predict.pipeline.stageD.diffusion.context_objects import (
    DiffusionStepInput,
    FeaturePreparationContext,
    EmbeddingContext,
)
from rna_predict.pipeline.stageD.diffusion.utils.config_utils import (
    validate_stageD_config,
    parse_diffusion_module_args,
)

# Initialize logger for Stage D diffusion manager
logger = logging.getLogger(
    "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager"
)


# --- Robust config extraction for Hydra and test layouts ---
def _get_stageD_diffusion_cfg(cfg):
    # Try Hydra (OmegaConf) nested structure
    if hasattr(cfg, "model") and hasattr(cfg.model, "stageD") and hasattr(cfg.model.stageD, "diffusion"):
        return cfg.model.stageD.diffusion
    # Try dict/test structure
    if isinstance(cfg, dict):
        if "stageD" in cfg and "diffusion" in cfg["stageD"]:
            return cfg["stageD"]["diffusion"]
        if "diffusion" in cfg:
            return cfg["diffusion"]
    # Try OmegaConf DictConfig with keys
    if hasattr(cfg, 'keys'):
        if 'stageD' in cfg and 'diffusion' in cfg['stageD']:
            return cfg['stageD']['diffusion']
        if 'diffusion' in cfg:
            return cfg['diffusion']
    raise ValueError("Could not find Stage D diffusion config in provided configuration.")


class DiffusionManagerConfig:
    def __init__(self, device, num_inference_steps, temperature, diffusion_module_args, debug_logging):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.temperature = temperature
        self.diffusion_module_args = diffusion_module_args
        self.debug_logging = debug_logging

    @classmethod
    def from_hydra_cfg(cls, cfg: DictConfig):
        validate_stageD_config(cfg)

        # PATCH: Always look under model.stageD
        stageD_cfg = cfg.model.stageD
        if "diffusion" in stageD_cfg:
            stage_cfg = stageD_cfg.diffusion
        elif "stageD" in stageD_cfg and "diffusion" in stageD_cfg.stageD:
            stage_cfg = stageD_cfg.stageD.diffusion
        else:
            raise ValueError(
                "[UNIQUE-ERR-STAGED-DIFFUSION-ACCESS] Cannot access diffusion config under model.stageD"
            )

        device = torch.device(stage_cfg.device)
        inference_cfg = stage_cfg.get("inference", OmegaConf.create({}))
        num_inference_steps = inference_cfg.get("num_steps", 2)
        temperature = inference_cfg.get("temperature", 1.0)
        diffusion_module_args = parse_diffusion_module_args(stage_cfg)
        debug_logging = stage_cfg.get("debug_logging", False)
        return cls(
            device,
            num_inference_steps,
            temperature,
            diffusion_module_args,
            debug_logging,
        )


class ProtenixDiffusionManager(torch.nn.Module):
    """
    Manager that handles training steps or multi-step inference for diffusion.
    Expects trunk_embeddings to contain:
      - "s_trunk"
      - optionally "s_inputs"
      - "pair"

    Uses standard Hydra configuration (DictConfig) with stageD.diffusion group.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        print("[MEMORY-LOG][StageD] Initializing ProtenixDiffusionManager")
        process = psutil.Process(os.getpid())
        print(
            f"[MEMORY-LOG][StageD] Memory usage: {process.memory_info().rss / 1e6:.2f} MB"
        )
        """
        Initializes the Diffusion Manager using Hydra configuration.
        Reads all parameters from cfg.stageD.diffusion group.
        """
        self.cfg = cfg
        self.config = DiffusionManagerConfig.from_hydra_cfg(cfg)
        self.device = self.config.device
        self.num_inference_steps = self.config.num_inference_steps
        self.temperature = self.config.temperature
        self.debug_logging = self.config.debug_logging

        # --- Respect Hydra init_from_scratch flag ---
        init_from_scratch = False
        # Support both flat and nested config
        if hasattr(cfg, "init_from_scratch"):
            init_from_scratch = cfg.init_from_scratch
        elif hasattr(cfg, "diffusion") and hasattr(cfg.diffusion, "init_from_scratch"):
            init_from_scratch = cfg.diffusion.init_from_scratch
        if init_from_scratch:
            logger.info(
                "[StageD] Initializing DiffusionModule from scratch (no checkpoint loaded)"
            )
            self.diffusion_module = DiffusionModule(
                **self.config.diffusion_module_args
            ).to(self.device)
        else:
            try:
                self.diffusion_module = DiffusionModule(
                    **self.config.diffusion_module_args
                ).to(self.device)
            except TypeError as e:
                logger.error(f"Error initializing DiffusionModule: {e}")
                logger.error(f"Config provided: {self.config.diffusion_module_args}")
                raise
        print("[MEMORY-LOG][StageD] After super().__init__")
        print(
            f"[MEMORY-LOG][StageD] Memory usage: {process.memory_info().rss / 1e6:.2f} MB"
        )

    def _get_sampler_params(self, stage_cfg: DictConfig):
        sampler_cfg = stage_cfg.get("sampler", OmegaConf.create({}))
        model_arch = stage_cfg["model_architecture"]
        return {
            "p_mean": sampler_cfg.get("p_mean", -1.2),
            "p_std": sampler_cfg.get("p_std", 1.5),
            "sigma_data": model_arch["sigma_data"],
            "N_sample": sampler_cfg.get("N_sample", 1),
            "diffusion_chunk_size": stage_cfg.get("diffusion_chunk_size"),
        }

    def train_diffusion_step(
        self,
        step_input: DiffusionStepInput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a single-step training pass with random noise injection.
        Sampler parameters are read from the stored Hydra config (self.cfg.stageD.diffusion.sampler).
        """
        if not OmegaConf.is_config(self.cfg):
            raise RuntimeError(
                "train_diffusion_step requires Hydra config (cfg) for parameters."
            )
        stage_cfg = self.cfg.model.stageD.diffusion
        sampler_params = self._get_sampler_params(stage_cfg)
        noise_sampler = TrainingNoiseSampler(
            p_mean=sampler_params["p_mean"],
            p_std=sampler_params["p_std"],
            sigma_data=sampler_params["sigma_data"],
        )

        step_input = step_input.to_device(self.device)

        x_gt_augment, x_denoised, sigma = sample_diffusion_training(
            noise_sampler=noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=step_input.label_dict,
            input_feature_dict=step_input.input_feature_dict,
            s_inputs=step_input.s_inputs,
            s_trunk=step_input.s_trunk,
            z_trunk=step_input.z_trunk,
            N_sample=sampler_params["N_sample"],
            diffusion_chunk_size=sampler_params["diffusion_chunk_size"],
        )
        return x_gt_augment, x_denoised, sigma

    def _move_trunk_embeddings_to_device(self, ctx: EmbeddingContext) -> Dict[str, Any]:
        return ctx.move_to_device().trunk_embeddings

    def _get_s_inputs(self, ctx: EmbeddingContext) -> torch.Tensor:
        return ctx.get_s_inputs()

    def _get_z_trunk(self, ctx: EmbeddingContext) -> torch.Tensor:
        return ctx.get_z_trunk()

    def _prepare_input_features(self, ctx: FeaturePreparationContext) -> Dict[str, Any]:
        return ctx.prepare()

    def _validate_trunk_embeddings(self, trunk_embeddings):
        if "s_trunk" not in trunk_embeddings or trunk_embeddings["s_trunk"] is None:
            raise ValueError(
                "StageD multi_step_inference requires a valid 's_trunk' in trunk_embeddings."
            )

    def _get_noise_schedule(self, num_steps, schedule_type, device):
        if schedule_type == "linear":
            return torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)
        warnings.warn(
            f"Using default linear noise schedule. Unknown type: {schedule_type}"
        )
        return torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)

    def _log_shapes(self, debug_logging, coords_init, s_trunk, s_inputs, z_trunk):
        if debug_logging:
            logger.debug(
                f"[multi_step_inference] coords_init shape: {coords_init.shape}"
            )
            logger.debug(f"[multi_step_inference] s_trunk shape: {s_trunk.shape}")
            if isinstance(s_inputs, torch.Tensor):
                logger.debug(f"[multi_step_inference] s_inputs shape: {s_inputs.shape}")
            if z_trunk is not None:
                logger.debug(f"[multi_step_inference] z_trunk shape: {z_trunk.shape}")

    def _log_final_coords(self, debug_logging, coords_final):
        if debug_logging:
            logger.debug(
                f"[multi_step_inference] Final coords shape: {coords_final.shape}"
            )

    def _postprocess_coords(self, coords_final, N_sample, debug_logging):
        if N_sample != 1:
            return coords_final
        if self._is_unexpected_shape(coords_final):
            coords_final = self._handle_unexpected_shape(coords_final)
        else:
            coords_final = self._handle_expected_shape(coords_final)
        return coords_final

    def _is_unexpected_shape(self, coords_final):
        # Handle multi-sample case with shape [B, N_sample, N_atoms, 3]
        if coords_final.ndim == 4 and coords_final.shape[3] == 3:
            # This is the expected shape for multi-sample output
            # We'll handle this in _handle_expected_shape
            return False

        # Original check for unexpected shape
        n_atoms_inferred = coords_final.shape[2] if coords_final.ndim > 2 else -1
        return (
            coords_final.ndim == 4
            and coords_final.shape[1] == n_atoms_inferred
            and coords_final.shape[2] == n_atoms_inferred
        )

    def _handle_unexpected_shape(self, coords_final):
        warnings.warn(f"Unexpected shape {coords_final.shape} for N_sample=1")
        return coords_final[:, 0, :, :]

    def _handle_expected_shape(self, coords_final):
        sample_dim_index = 1

        # Handle multi-sample case with shape [B, N_sample, N_atoms, 3]
        if coords_final.ndim == 4 and coords_final.shape[3] == 3:
            # For multi-sample output, we keep the sample dimension
            # but we'll log it for debugging
            print(f"[DEBUG][PATCHED] Found multi-sample coords with shape {coords_final.shape}")
            return coords_final

        # Original handling for standard case
        if (
            coords_final.ndim > sample_dim_index
            and coords_final.shape[sample_dim_index] == 1
        ):
            coords_final = coords_final.squeeze(sample_dim_index)
        return coords_final

    @snoop
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
        # --- Robust config extraction ---
        try:
            stage_cfg = _get_stageD_diffusion_cfg(self.cfg)
        except Exception as e:
            logger.error(f"[CONFIG-ERROR][StageD] {e}")
            raise
        # --- PATCH: Handle extra 'diffusion' nesting in config ---
        inference_cfg = stage_cfg.get("inference", None)
        if (
            inference_cfg is None
            and "diffusion" in stage_cfg
            and isinstance(stage_cfg["diffusion"], dict)
        ):
            logger.warning(
                "[HYDRA-CONF-WARN][StageD] Detected extra 'diffusion' nesting in config. Falling back to stage_cfg['diffusion']['inference']."
            )
            inference_cfg = stage_cfg["diffusion"].get(
                "inference", OmegaConf.create({})
            )
        elif inference_cfg is None:
            inference_cfg = OmegaConf.create({})
        sampling_cfg = inference_cfg.get("sampling", OmegaConf.create({}))
        N_sample = sampling_cfg.get("num_samples", 1)
        if "num_steps" not in inference_cfg:
            # --- SYSTEMATIC DEBUGGING: Print config state before setting default ---
            logger.warning(
                "[HYDRA-CONF-DEBUG][StageD] 'num_steps' missing from inference config. Using default value of 2."
            )
            logger.debug(
                f"[HYDRA-CONF-DEBUG][StageD] stage_cfg type: {type(stage_cfg)}"
            )
            logger.debug(
                f"[HYDRA-CONF-DEBUG][StageD] stage_cfg keys: {list(stage_cfg.keys()) if hasattr(stage_cfg, 'keys') else stage_cfg}"
            )
            logger.debug(
                f"[HYDRA-CONF-DEBUG][StageD] inference_cfg type: {type(inference_cfg)}"
            )
            logger.debug(f"[HYDRA-CONF-DEBUG][StageD] inference_cfg: {inference_cfg}")

            # Set default value for num_steps
            inference_cfg["num_steps"] = 2
        num_steps = inference_cfg["num_steps"]
        logger.info(
            f"[HYDRA-CONF-DEBUG][StageD] Using num_steps from config: {num_steps}"
        )
        noise_schedule_cfg = stage_cfg.get("noise_schedule", OmegaConf.create({}))
        debug_logging = stage_cfg.get("debug_logging", False)
        device = self.device
        coords_init = coords_init.to(device)
        emb_ctx = EmbeddingContext(
            trunk_embeddings, override_input_features, stage_cfg, device
        )
        trunk_embeddings = self._move_trunk_embeddings_to_device(emb_ctx)
        self._validate_trunk_embeddings(trunk_embeddings)
        emb_ctx.trunk_embeddings = trunk_embeddings
        s_inputs = self._get_s_inputs(emb_ctx)
        z_trunk = self._get_z_trunk(emb_ctx)

        # Ensure s_trunk and s_inputs have compatible dimensions
        s_trunk = trunk_embeddings["s_trunk"]

        # If s_trunk has 5 dimensions [B, 1, N_sample, N_res, C] but s_inputs has 4 [B, N_sample, N_res, C]
        if s_trunk.dim() == 5 and s_inputs.dim() == 4:
            logger.info(f"[StageD] Reshaping s_trunk from 5D {s_trunk.shape} to 4D to match s_inputs {s_inputs.shape}")
            s_trunk = s_trunk.squeeze(1)  # Remove the extra dimension at index 1
            trunk_embeddings["s_trunk"] = s_trunk
            logger.info(f"[StageD] After reshaping, s_trunk shape: {s_trunk.shape}")

        # --- Assert and log z_trunk shape ---
        logger.info(f"[StageD] z_trunk shape at entry: {z_trunk.shape}")
        expected_ndim = 4  # [B, N_res, N_res, C] or [B, N_sample, N_res, N_res, C] if sample dim present

        # Handle multi-sample case with extra dimensions
        if z_trunk.dim() == 6:  # [B, 1, N_sample, N_res, N_res, C]
            # This is likely a case where we have an extra dimension from the bridging process
            # Reshape to [B, N_sample, N_res, N_res, C] by removing the extra dimension
            logger.info(f"[StageD] Reshaping 6D z_trunk with shape {z_trunk.shape} to 5D")
            z_trunk = z_trunk.squeeze(1)  # Remove the extra dimension at index 1
            logger.info(f"[StageD] After reshaping, z_trunk shape: {z_trunk.shape}")

        if z_trunk.dim() not in (4, 5):
            raise ValueError(f"[StageD] z_trunk must be 4D or 5D (residue-level pair tensor), got shape {z_trunk.shape}")

        # --- If atom-level pairs are needed, bridge using residue_atom_bridge ---
        if stage_cfg.get("require_atom_level_pairs", False):
            from rna_predict.pipeline.stageD.diffusion.bridging.residue_atom_bridge import _process_pair_embedding
            # You must provide residue_atom_map (List[List[int]]) and key (e.g., 'pair')
            # Example: residue_atom_map = ... (get from context or input features)
            # z_trunk = _process_pair_embedding(z_trunk, residue_atom_map, self.debug_logging, key="pair")
            logger.info("[StageD] Bridging z_trunk from residue-level to atom-level pairs using _process_pair_embedding...")
            # Uncomment and set up the following line as needed:
            # z_trunk = _process_pair_embedding(z_trunk, residue_atom_map, self.debug_logging, key="pair")
            pass  # TODO: Provide residue_atom_map and call bridging here

        self._log_shapes(
            debug_logging, coords_init, trunk_embeddings["s_trunk"], s_inputs, z_trunk
        )
        ctx = FeaturePreparationContext(
            coords_init, override_input_features, device, trunk_embeddings, s_inputs
        )
        input_feature_dict = self._prepare_input_features(ctx)
        schedule_type = noise_schedule_cfg.get("schedule_type", "linear")
        noise_schedule = self._get_noise_schedule(num_steps, schedule_type, device)
        if debug_logging:
            logger.debug("[multi_step_inference] Before sample_diffusion:")
            logger.debug(f"  coords_init shape: {coords_init.shape}")
            logger.debug(f"  s_trunk shape: {trunk_embeddings['s_trunk'].shape}")
            logger.debug(
                f"  noise_schedule (len={len(noise_schedule)}): {noise_schedule}"
            )
            logger.debug(f"  num_steps from config: {num_steps}")
            logger.debug(f"  schedule_type from config: {schedule_type}")
        inplace_safe = inference_cfg.get("inplace_safe", False)
        attn_chunk_size = stage_cfg.get("attn_chunk_size")
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
        coords_final = self._postprocess_coords(coords_final, N_sample, debug_logging)
        # If output shape is [B, 1, N_atoms, 3] and N_sample==1, squeeze the sample dimension
        if (
            N_sample == 1 and isinstance(coords_final, torch.Tensor)
            and coords_final.ndim == 4 and coords_final.shape[1] == 1 and coords_final.shape[-1] == 3
        ):
            coords_final = coords_final.squeeze(1)
            logger.info(f"[PATCHED] Squeezed singleton sample dimension: new shape {coords_final.shape}")
        self._log_final_coords(debug_logging, coords_final)
        return coords_final

    def _manual_forward_pass(
        self, x_gt: torch.Tensor, trunk_embeddings: dict, sigma: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_gt = x_gt.to(self.device)
        x_noisy = x_gt + torch.randn_like(x_gt) * sigma
        if not hasattr(self, "diffusion_module"):
            raise RuntimeError("Diffusion module not initialized. Call __init__ first.")
        x_denoised = self.diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=torch.tensor([sigma], device=self.device),
            input_feature_dict={},
            s_inputs=trunk_embeddings.get("s_inputs"),
            s_trunk=trunk_embeddings.get("s_trunk"),
            z_trunk=trunk_embeddings.get("pair"),
        )
        return x_noisy, x_denoised

    def custom_manual_loop(
        self, x_gt: torch.Tensor, trunk_embeddings: dict, sigma: float
    ):
        """
        Optional direct usage demonstration: single forward pass
        """
        return self._manual_forward_pass(x_gt, trunk_embeddings, sigma)
