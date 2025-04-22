import warnings
import logging
from typing import Dict, Optional, Any
from omegaconf import OmegaConf, DictConfig
import torch
from dataclasses import dataclass
import torch.nn as nn
import os, psutil

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


@dataclass
class DiffusionStepInput:
    label_dict: Dict[str, torch.Tensor]
    input_feature_dict: Dict[str, torch.Tensor]
    s_inputs: torch.Tensor
    s_trunk: torch.Tensor
    z_trunk: torch.Tensor

    @classmethod
    def from_args(cls, label_dict, input_feature_dict, s_inputs, s_trunk, z_trunk):
        # Accepts 5 arguments for backward compatibility, but encourages use of a dict or context object in future
        # TODO: Refactor all usages to pass a single context or dict to further reduce argument count and silence CodeScene
        return cls(label_dict, input_feature_dict, s_inputs, s_trunk, z_trunk)

    def to_device(self, device):
        """
        Moves all tensors (including nested in dicts/lists/tuples) to the specified device.
        Unique Error: [UNIQUE-ERR-DIFFSTEPINPUT-DEVICE-001] If you see this error, a non-tensor object in a nested dict could not be moved to device.
        """
        def _move_to_device(obj, device):
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: _move_to_device(v, device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_move_to_device(v, device) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(_move_to_device(v, device) for v in obj)
            else:
                return obj

        try:
            self.label_dict = _move_to_device(self.label_dict, device)
            self.input_feature_dict = _move_to_device(self.input_feature_dict, device)
            self.s_inputs = self.s_inputs.to(device)
            self.s_trunk = self.s_trunk.to(device)
            self.z_trunk = self.z_trunk.to(device)
        except Exception as e:
            raise RuntimeError(f"[UNIQUE-ERR-DIFFSTEPINPUT-DEVICE-001] Failed to move DiffusionStepInput to device: {e}")
        return self

    def as_tuple(self):
        return (self.label_dict, self.input_feature_dict, self.s_inputs, self.s_trunk, self.z_trunk)

    def __iter__(self):
        return iter(self.as_tuple())

    def __getitem__(self, idx):
        return self.as_tuple()[idx]

    def __len__(self):
        return 5

    def __repr__(self):
        return f"DiffusionStepInput(label_dict=..., input_feature_dict=..., s_inputs=..., s_trunk=..., z_trunk=...)"


@dataclass
class FeaturePreparationContext:
    coords_init: torch.Tensor
    override_input_features: Optional[Dict[str, Any]]
    device: torch.device
    trunk_embeddings: Dict[str, Any]
    s_inputs: torch.Tensor

    @property
    def batch_size(self):
        return self.coords_init.shape[0]

    @property
    def n_atoms(self):
        return self.coords_init.shape[1]

    @property
    def s_trunk_shape(self):
        return self.trunk_embeddings["s_trunk"].shape

    def fallback_input_feature_dict(self):
        return {
            "atom_to_token_idx": torch.arange(self.n_atoms, device=self.device).long().unsqueeze(0).expand(self.batch_size, -1)
        }

    def get_atom_idx(self, input_feature_dict):
        return input_feature_dict["atom_to_token_idx"]

    def update_atom_idx(self, input_feature_dict, atom_idx):
        input_feature_dict["atom_to_token_idx"] = atom_idx
        return input_feature_dict

    def is_expand_needed(self, atom_idx):
        return (atom_idx.dim() == 2 and self.trunk_embeddings["s_trunk"].dim() == 4 and self.trunk_embeddings["s_trunk"].shape[1] == 1)

    def is_expand_needed_s_inputs(self, atom_idx):
        return (atom_idx.dim() == 2 and self.s_inputs is not None and self.s_inputs.dim() == 4 and self.s_inputs.shape[1] == 1)

    def expand_atom_idx(self, atom_idx):
        return atom_idx.unsqueeze(1)

    def prepare(self):
        if self.override_input_features is not None:
            input_feature_dict = self.override_input_features
        else:
            input_feature_dict = self.fallback_input_feature_dict()
        if "atom_to_token_idx" in input_feature_dict:
            atom_idx = self.get_atom_idx(input_feature_dict)
            if self.is_expand_needed(atom_idx):
                atom_idx = self.expand_atom_idx(atom_idx)
            elif self.is_expand_needed_s_inputs(atom_idx):
                atom_idx = self.expand_atom_idx(atom_idx)
            input_feature_dict = self.update_atom_idx(input_feature_dict, atom_idx)
        return input_feature_dict

    def __repr__(self):
        return f"FeaturePreparationContext(coords_init=..., override_input_features=..., device=..., trunk_embeddings=..., s_inputs=...)"


@dataclass
class EmbeddingContext:
    trunk_embeddings: Dict[str, Any]
    override_input_features: Optional[Dict[str, Any]] = None
    stage_cfg: Optional[DictConfig] = None
    device: Optional[torch.device] = None

    def move_to_device(self):
        processed = {}
        for k, v in self.trunk_embeddings.items():
            if isinstance(v, torch.Tensor):
                processed[k] = v.to(self.device)
            else:
                processed[k] = v
        self.trunk_embeddings = processed
        return self

    def get_s_inputs(self):
        s_inputs = self.trunk_embeddings.get("s_inputs")
        if s_inputs is None and self.override_input_features is not None:
            s_inputs = self.override_input_features.get("s_inputs")
        if s_inputs is None:
            logger.warning("'s_inputs' not found in trunk_embeddings or override_input_features. Creating fallback.")
            s_trunk_shape = self.trunk_embeddings["s_trunk"].shape
            batch_size = s_trunk_shape[0]
            n_tokens = s_trunk_shape[1]
            # Use config-driven c_s_inputs
            c_s_inputs_dim = self.stage_cfg.get("c_s_inputs", None)
            if c_s_inputs_dim is None:
                raise ValueError("Config missing c_s_inputs for fallback s_inputs shape.")
            s_inputs = torch.zeros((batch_size, n_tokens, c_s_inputs_dim), device=self.device)
        return s_inputs

    def get_z_trunk(self):
        z_trunk = self.trunk_embeddings.get("pair")
        if z_trunk is None:
            logger.warning("'pair' embedding not found in trunk_embeddings. Creating fallback.")
            s_trunk_shape = self.trunk_embeddings["s_trunk"].shape
            batch_size = s_trunk_shape[0]
            n_tokens = s_trunk_shape[1]
            c_z_dim = self.stage_cfg.get("c_z", 128)
            z_trunk = torch.zeros((batch_size, n_tokens, n_tokens, c_z_dim), device=self.device)
        return z_trunk


@dataclass
class DiffusionManagerConfig:
    device: torch.device
    num_inference_steps: int
    temperature: float
    diffusion_module_args: dict
    debug_logging: bool

    @classmethod
    def from_hydra_cfg(cls, cfg: DictConfig):
        cls._validate_config(cfg)

        # Handle both single and double nesting of stageD
        if "diffusion" in cfg.stageD:
            # Single nesting: cfg.stageD.diffusion
            stage_cfg = cfg.stageD.diffusion
        elif "stageD" in cfg.stageD and "diffusion" in cfg.stageD.stageD:
            # Double nesting: cfg.stageD.stageD.diffusion
            stage_cfg = cfg.stageD.stageD.diffusion
        else:
            # This should never happen due to _validate_config, but as a safeguard
            raise ValueError("[UNIQUE-ERR-STAGED-DIFFUSION-ACCESS] Cannot access diffusion config")

        device = torch.device(stage_cfg.device)
        inference_cfg = stage_cfg.get("inference", OmegaConf.create({}))
        num_inference_steps = inference_cfg.get("num_steps", 2)
        temperature = inference_cfg.get("temperature", 1.0)
        diffusion_module_args = cls._parse_diffusion_module_args(stage_cfg)
        debug_logging = stage_cfg.get("debug_logging", False)
        return cls(device, num_inference_steps, temperature, diffusion_module_args, debug_logging)

    @staticmethod
    def _validate_config(cfg: DictConfig):
        if not OmegaConf.is_config(cfg):
            raise ValueError("[UNIQUE-ERR-STAGED-CONFIG-TYPE] Config must be a Hydra DictConfig")
        if "stageD" not in cfg:
            raise ValueError("[UNIQUE-ERR-STAGED-MISSING] Config missing required 'stageD' group")

        # Handle both single and double nesting of stageD
        if "diffusion" in cfg.stageD:
            # Single nesting: cfg.stageD.diffusion
            return
        elif "stageD" in cfg.stageD and "diffusion" in cfg.stageD.stageD:
            # Double nesting: cfg.stageD.stageD.diffusion
            return
        else:
            # Neither structure found
            raise ValueError("[UNIQUE-ERR-STAGED-DIFFUSION-MISSING] Config missing required 'diffusion' group in stageD. Available keys: " +
                           str(list(cfg.stageD.keys())))

    @staticmethod
    def _parse_diffusion_module_args(stage_cfg: DictConfig):
        """
        Extract all required model dimensions and architectural parameters from the config,
        strictly using config values and never hardcoded fallbacks.
        """
        # Instrument: print the full config for debugging
        print("[DEBUG][_parse_diffusion_module_args] stage_cfg:", stage_cfg)

        # If 'diffusion' key exists, descend into it
        if "diffusion" in stage_cfg:
            print("[DEBUG][_parse_diffusion_module_args] Descending into 'diffusion' section of config.")
            base_cfg = stage_cfg["diffusion"]
        else:
            base_cfg = stage_cfg

        def get_nested(cfg, keys, default=None):
            for k in keys:
                if hasattr(cfg, k):
                    cfg = getattr(cfg, k)
                elif isinstance(cfg, dict) and k in cfg:
                    cfg = cfg[k]
                else:
                    return default
            return cfg

        diffusion_module_args = {}
        # Top-level parameters
        diffusion_module_args["sigma_data"] = get_nested(base_cfg, ["sigma_data"])
        # Model sizes from model_architecture
        arch_cfg = get_nested(base_cfg, ["model_architecture"])
        required_model_keys = ["c_token", "c_s", "c_z", "c_s_inputs", "c_atom", "c_noise_embedding"]
        if arch_cfg is not None:
            for key in required_model_keys:
                val = get_nested(arch_cfg, [key])
                if val is None:
                    raise ValueError(f"Required model_architecture field '{key}' missing in config!")
                diffusion_module_args[key] = val
        else:
            raise ValueError("model_architecture section missing in config!")
        # c_atompair: try to get from atom_encoder or raise error
        atom_encoder_cfg = get_nested(base_cfg, ["atom_encoder"])
        if atom_encoder_cfg is not None and "c_out" in atom_encoder_cfg:
            diffusion_module_args["c_atompair"] = atom_encoder_cfg["c_out"]
        else:
            raise ValueError("c_atompair (atom_encoder.c_out) missing in config!")
        # atom_encoder, atom_decoder, transformer
        for subkey in ["atom_encoder", "atom_decoder", "transformer"]:
            subcfg = get_nested(base_cfg, [subkey])
            if subcfg is not None:
                diffusion_module_args[subkey] = dict(subcfg) if hasattr(subcfg, 'items') else dict(subcfg)
            else:
                raise ValueError(f"Required config section '{subkey}' missing in config!")
        # Optional blocks_per_ckpt, use_fine_grained_checkpoint, initialization
        for key in ["blocks_per_ckpt", "use_fine_grained_checkpoint", "initialization"]:
            val = get_nested(base_cfg, [key])
            if val is not None:
                diffusion_module_args[key] = val
        # Memory optimization
        mem_cfg = get_nested(base_cfg, ["memory"], {})
        use_ckpt = mem_cfg.get("use_checkpointing", False)
        blocks_per_ckpt = mem_cfg.get("blocks_per_ckpt") if use_ckpt else None
        use_fine_grained = mem_cfg.get("use_fine_grained_checkpoint", False) if blocks_per_ckpt else False
        diffusion_module_args.update({
            "blocks_per_ckpt": blocks_per_ckpt,
            "use_fine_grained_checkpoint": use_fine_grained
        })
        print("[DEBUG][_parse_diffusion_module_args] Final module args:", diffusion_module_args)
        return diffusion_module_args


class ProtenixDiffusionManager(nn.Module):
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
        print(f"[MEMORY-LOG][StageD] Memory usage: {process.memory_info().rss / 1e6:.2f} MB")
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
        if hasattr(cfg, 'init_from_scratch'):
            init_from_scratch = cfg.init_from_scratch
        elif hasattr(cfg, 'diffusion') and hasattr(cfg.diffusion, 'init_from_scratch'):
            init_from_scratch = cfg.diffusion.init_from_scratch
        if init_from_scratch:
            logger.info("[StageD] Initializing DiffusionModule from scratch (no checkpoint loaded)")
            self.diffusion_module = DiffusionModule(**self.config.diffusion_module_args).to(self.device)
        else:
            try:
                self.diffusion_module = DiffusionModule(**self.config.diffusion_module_args).to(self.device)
            except TypeError as e:
                logger.error(f"Error initializing DiffusionModule: {e}")
                logger.error(f"Config provided: {self.config.diffusion_module_args}")
                raise
        print("[MEMORY-LOG][StageD] After super().__init__")
        print(f"[MEMORY-LOG][StageD] Memory usage: {process.memory_info().rss / 1e6:.2f} MB")

    def _get_sampler_params(self, stage_cfg: DictConfig):
        sampler_cfg = stage_cfg.get("sampler", OmegaConf.create({}))
        return {
            "p_mean": sampler_cfg.get("p_mean", -1.2),
            "p_std": sampler_cfg.get("p_std", 1.5),
            "sigma_data": stage_cfg.get("sigma_data", 16.0),
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
            raise RuntimeError("train_diffusion_step requires Hydra config (cfg) for parameters.")
        stage_cfg = self.cfg.stageD.diffusion
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
            raise ValueError("StageD multi_step_inference requires a valid 's_trunk' in trunk_embeddings.")

    def _get_noise_schedule(self, num_steps, schedule_type, device):
        if schedule_type == "linear":
            return torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)
        warnings.warn(f"Using default linear noise schedule. Unknown type: {schedule_type}")
        return torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)

    def _log_shapes(self, debug_logging, coords_init, s_trunk, s_inputs, z_trunk):
        if debug_logging:
            logger.debug(f"[multi_step_inference] coords_init shape: {coords_init.shape}")
            logger.debug(f"[multi_step_inference] s_trunk shape: {s_trunk.shape}")
            if isinstance(s_inputs, torch.Tensor):
                logger.debug(f"[multi_step_inference] s_inputs shape: {s_inputs.shape}")
            if z_trunk is not None:
                logger.debug(f"[multi_step_inference] z_trunk shape: {z_trunk.shape}")

    def _log_final_coords(self, debug_logging, coords_final):
        if debug_logging:
            logger.debug(f"[multi_step_inference] Final coords shape: {coords_final.shape}")

    def _postprocess_coords(self, coords_final, N_sample, debug_logging):
        if N_sample != 1:
            return coords_final
        if self._is_unexpected_shape(coords_final):
            coords_final = self._handle_unexpected_shape(coords_final)
        else:
            coords_final = self._handle_expected_shape(coords_final)
        return coords_final

    def _is_unexpected_shape(self, coords_final):
        n_atoms_inferred = coords_final.shape[2] if coords_final.ndim > 2 else -1
        return (
            coords_final.ndim == 4 and coords_final.shape[1] == n_atoms_inferred and coords_final.shape[2] == n_atoms_inferred
        )

    def _handle_unexpected_shape(self, coords_final):
        warnings.warn(f"Unexpected shape {coords_final.shape} for N_sample=1")
        return coords_final[:, 0, :, :]

    def _handle_expected_shape(self, coords_final):
        sample_dim_index = 1
        if (coords_final.ndim > sample_dim_index and coords_final.shape[sample_dim_index] == 1):
            coords_final = coords_final.squeeze(sample_dim_index)
        return coords_final

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
        emb_ctx = EmbeddingContext(trunk_embeddings, override_input_features, stage_cfg, device)
        trunk_embeddings = self._move_trunk_embeddings_to_device(emb_ctx)
        self._validate_trunk_embeddings(trunk_embeddings)
        emb_ctx.trunk_embeddings = trunk_embeddings
        s_inputs = self._get_s_inputs(emb_ctx)
        z_trunk = self._get_z_trunk(emb_ctx)
        if z_trunk.shape[1] != trunk_embeddings["s_trunk"].shape[1] or z_trunk.shape[2] != trunk_embeddings["s_trunk"].shape[1]:
            raise RuntimeError(f"shape mismatch between z_trunk {z_trunk.shape} and s_trunk {trunk_embeddings['s_trunk'].shape}")
        self._log_shapes(debug_logging, coords_init, trunk_embeddings["s_trunk"], s_inputs, z_trunk)
        ctx = FeaturePreparationContext(coords_init, override_input_features, device, trunk_embeddings, s_inputs)
        input_feature_dict = self._prepare_input_features(ctx)
        sampling_cfg = inference_cfg.get("sampling", OmegaConf.create({}))
        N_sample = sampling_cfg.get("num_samples", 1)
        num_steps = inference_cfg.get("num_steps", 50)
        schedule_type = noise_schedule_cfg.get("schedule_type", "linear")
        noise_schedule = self._get_noise_schedule(num_steps, schedule_type, device)
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
        coords_final = self._postprocess_coords(coords_final, N_sample, debug_logging)
        self._log_final_coords(debug_logging, coords_final)
        return coords_final

    def _manual_forward_pass(self, x_gt: torch.Tensor, trunk_embeddings: dict, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        x_gt = x_gt.to(self.device)
        x_noisy = x_gt + torch.randn_like(x_gt) * sigma
        if not hasattr(self, 'diffusion_module'):
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
