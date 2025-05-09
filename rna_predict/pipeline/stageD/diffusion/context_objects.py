# Context and Data Classes for Stage D Diffusion
# -------------------------------------------------------
# Contains dataclasses and context objects used throughout the Stage D diffusion pipeline.

from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
from omegaconf import DictConfig
import logging

logger = logging.getLogger("rna_predict.pipeline.stageD.diffusion.context_objects")

@dataclass
class DiffusionStepInput:
    label_dict: Dict[str, torch.Tensor]
    input_feature_dict: Dict[str, torch.Tensor]
    s_inputs: torch.Tensor
    s_trunk: torch.Tensor
    z_trunk: torch.Tensor

    @classmethod
    def from_args(cls, label_dict, input_feature_dict, s_inputs, s_trunk, z_trunk):
        return cls(label_dict, input_feature_dict, s_inputs, s_trunk, z_trunk)

    def to_device(self, device):
        try:
            self.s_inputs = self.s_inputs.to(device)
            self.s_trunk = self.s_trunk.to(device)
            self.z_trunk = self.z_trunk.to(device)
        except Exception as e:
            raise RuntimeError(
                f"[UNIQUE-ERR-DIFFSTEPINPUT-DEVICE-001] Failed to move DiffusionStepInput to device: {e}"
            )
        return self

    def as_tuple(self):
        return (
            self.label_dict,
            self.input_feature_dict,
            self.s_inputs,
            self.s_trunk,
            self.z_trunk,
        )

    def __iter__(self):
        return iter(self.as_tuple())

    def __getitem__(self, idx):
        return self.as_tuple()[idx]

    def __len__(self):
        return 5

    def __repr__(self):
        return "DiffusionStepInput(label_dict=..., input_feature_dict=..., s_inputs=..., s_trunk=..., z_trunk=...)"

@dataclass
class FeaturePreparationContext:
    coords_init: torch.Tensor
    override_input_features: Optional[Dict[str, Any]]
    device: torch.device
    trunk_embeddings: Dict[str, Any]
    s_inputs: torch.Tensor
    max_atoms: int = 4096  # default fallback, but should always be set from cfg

    def __post_init__(self):
        logger.debug(f"[INSTRUMENT][FeaturePreparationContext] max_atoms={self.max_atoms}, coords_init.shape={self.coords_init.shape}")

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
        # Use config-driven max_atoms
        atom_idx = torch.arange(self.max_atoms, device=self.device).long().unsqueeze(0).expand(self.batch_size, -1)
        logger.debug(f"[INSTRUMENT][FeaturePreparationContext] fallback_input_feature_dict: max_atoms={self.max_atoms}, atom_idx.shape={atom_idx.shape}")
        return {
            "atom_to_token_idx": atom_idx
        }

    def get_atom_idx(self, input_feature_dict):
        return input_feature_dict["atom_to_token_idx"]

    def is_expand_needed(self, atom_idx):
        # Check if atom_idx needs to be expanded to match s_trunk dimensions
        # Case 1: atom_idx is [B, N_atoms] but s_trunk is [B, N_sample, N_res, C]
        if atom_idx.dim() == 2 and self.trunk_embeddings["s_trunk"].dim() == 4:
            return True
        # Case 2: atom_idx is [B, N_sample, N_atoms] but s_trunk is [B, N_sample, N_res, C]
        # and they have different N_sample dimensions
        elif atom_idx.dim() == 3 and self.trunk_embeddings["s_trunk"].dim() == 4:
            return atom_idx.shape[1] != self.trunk_embeddings["s_trunk"].shape[1]
        return False

    def is_expand_needed_s_inputs(self, atom_idx):
        return (
            atom_idx.dim() == 2
            and self.s_inputs is not None
            and self.s_inputs.dim() == 4
            and self.s_inputs.shape[1] == 1
        )

    def expand_atom_idx(self, atom_idx):
        # If atom_idx is [B, N_atoms], expand to [B, N_sample, N_atoms]
        if atom_idx.dim() == 2 and self.trunk_embeddings["s_trunk"].dim() == 4:
            n_samples = self.trunk_embeddings["s_trunk"].shape[1]
            return atom_idx.unsqueeze(1).expand(-1, n_samples, -1)
        # If atom_idx is [B, N_sample_old, N_atoms], reshape to [B, N_sample_new, N_atoms]
        elif atom_idx.dim() == 3 and self.trunk_embeddings["s_trunk"].dim() == 4:
            n_samples = self.trunk_embeddings["s_trunk"].shape[1]
            # Create a new tensor with the correct shape
            # Take the first sample and expand it to the required number of samples
            new_atom_idx = atom_idx[:, 0:1, :].expand(-1, n_samples, -1)
            return new_atom_idx
        # Default case: just add a sample dimension
        return atom_idx.unsqueeze(1)

    def update_atom_idx(self, input_feature_dict, atom_idx):
        input_feature_dict["atom_to_token_idx"] = atom_idx
        return input_feature_dict

    def prepare(self):
        input_feature_dict = self.override_input_features or self.fallback_input_feature_dict()
        if "atom_to_token_idx" in input_feature_dict:
            atom_idx = self.get_atom_idx(input_feature_dict)
            if self.is_expand_needed(atom_idx):
                atom_idx = self.expand_atom_idx(atom_idx)
            elif self.is_expand_needed_s_inputs(atom_idx):
                atom_idx = self.expand_atom_idx(atom_idx)
            input_feature_dict = self.update_atom_idx(input_feature_dict, atom_idx)
        return input_feature_dict

    def __repr__(self):
        return "FeaturePreparationContext(coords_init=..., override_input_features=..., device=..., trunk_embeddings=..., s_inputs=...)"

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
        # SYSTEMATIC DEBUGGING: Print trunk_embeddings keys before accessing 's_inputs'
        logger.debug(f"[DEBUG][get_s_inputs] trunk_embeddings keys: {list(self.trunk_embeddings.keys())}")
        s_inputs = self.trunk_embeddings.get("s_inputs")
        # Determine expected c_s_inputs from config or nested feature_dimensions
        if hasattr(self.stage_cfg, "c_s_inputs"):
            expected_c = getattr(self.stage_cfg, "c_s_inputs")
        elif hasattr(self.stage_cfg, "feature_dimensions"):
            fd = getattr(self.stage_cfg, "feature_dimensions")
            expected_c = getattr(fd, "c_s_inputs", None) if not isinstance(fd, dict) else fd.get("c_s_inputs")
        elif isinstance(self.stage_cfg, dict) and "feature_dimensions" in self.stage_cfg:
            expected_c = self.stage_cfg["feature_dimensions"].get("c_s_inputs")
        else:
            expected_c = None
        if expected_c is not None and s_inputs is not None and s_inputs.dim() >= 1 and s_inputs.shape[-1] != expected_c:
            logger.warning(f"[StageD][HydraConf] Dropping s_inputs with channels {s_inputs.shape[-1]} != config c_s_inputs {expected_c}")
            s_inputs = None
        if s_inputs is None and self.override_input_features is not None:
            logger.debug(f"[DEBUG][get_s_inputs] override_input_features keys: {list(self.override_input_features.keys()) if self.override_input_features is not None else None}")
            s_inputs = self.override_input_features.get("s_inputs")
            # Drop override s_inputs if channel dim mismatches config
            if expected_c is not None and s_inputs is not None and s_inputs.dim() >= 1 and s_inputs.shape[-1] != expected_c:
                logger.warning(f"[StageD][HydraConf] Dropping override s_inputs channels {s_inputs.shape[-1]} != config c_s_inputs {expected_c}")
                s_inputs = None
        if s_inputs is None:
            logger.warning(
                "'s_inputs' not found in trunk_embeddings or override_input_features. Creating fallback."
            )
            # SYSTEMATIC DEBUGGING: Print trunk_embeddings before fallback
            logger.debug(f"[DEBUG][get_s_inputs] trunk_embeddings before fallback: {self.trunk_embeddings}")
            if "s_trunk" not in self.trunk_embeddings:
                logger.error("[ERROR][get_s_inputs] 's_trunk' missing from trunk_embeddings. Cannot create fallback s_inputs.")
                raise KeyError("'s_trunk' missing from trunk_embeddings. Cannot create fallback s_inputs.")
            s_trunk_shape = self.trunk_embeddings["s_trunk"].shape
            batch_size = s_trunk_shape[0]
            n_tokens = s_trunk_shape[1]
            # Use expected_c from config (possibly nested in feature_dimensions)
            c_s_inputs_dim = expected_c
            if c_s_inputs_dim is None:
                raise KeyError("Cannot determine c_s_inputs for fallback s_inputs from config.")
            s_inputs = torch.zeros(
                (batch_size, n_tokens, c_s_inputs_dim), device=self.device
            )
        else:
            # Handle multi-sample case with extra dimensions
            if s_inputs.dim() == 5:  # [B, 1, N_sample, N_res, C]
                # This is likely a case where we have an extra dimension from the bridging process
                # Reshape to [B, N_sample, N_res, C] by removing the extra dimension
                logger.info(f"[EmbeddingContext] Reshaping 5D s_inputs with shape {s_inputs.shape} to 4D")
                s_inputs = s_inputs.squeeze(1)  # Remove the extra dimension at index 1
                logger.info(f"[EmbeddingContext] After reshaping, s_inputs shape: {s_inputs.shape}")
        return s_inputs

    def get_z_trunk(self):
        from rna_predict.pipeline.stageD.run_stageD import log_mem
        z_trunk = self.trunk_embeddings.get("pair")
        if z_trunk is None:
            logger.warning(
                "'pair' embedding not found in trunk_embeddings. Creating fallback."
            )
            s_trunk_shape = self.trunk_embeddings["s_trunk"].shape
            batch_size = s_trunk_shape[0]
            n_tokens = s_trunk_shape[1]
            c_z_dim = self.stage_cfg["model_architecture"]["c_z"]
            z_trunk = torch.zeros(
                (batch_size, n_tokens, n_tokens, c_z_dim), device=self.device
            )
            logger.debug(f"[EmbeddingContext] Created fallback z_trunk with shape: {z_trunk.shape}")
            log_mem("After z_trunk fallback allocation")
        else:
            logger.debug(f"[EmbeddingContext] Initial z_trunk shape: {z_trunk.shape}")
            # Handle multi-sample case with extra dimensions
            if z_trunk.dim() == 6:  # [B, 1, N_sample, N_res, N_res, C]
                logger.info(f"[EmbeddingContext] Reshaping 6D z_trunk with shape {z_trunk.shape} to 5D")
                z_trunk = z_trunk.squeeze(1)  # Remove the extra dimension at index 1
                logger.info(f"[EmbeddingContext] After reshaping, z_trunk shape: {z_trunk.shape}")
                log_mem("After z_trunk reshape (6D->5D)")
            # Patch: If z_trunk is 3D, expand to 4D by adding batch dim
            elif z_trunk.dim() == 3:  # [N_res, N_res, C]
                logger.warning(f"[EmbeddingContext] z_trunk is 3D, expanding to 4D. Original shape: {z_trunk.shape}")
                z_trunk = z_trunk.unsqueeze(0)  # [1, N_res, N_res, C]
                logger.info(f"[EmbeddingContext] After expanding, z_trunk shape: {z_trunk.shape}")
                log_mem("After z_trunk expand (3D->4D)")
        # Always log memory after get_z_trunk returns
        log_mem("After get_z_trunk return")
        return z_trunk
