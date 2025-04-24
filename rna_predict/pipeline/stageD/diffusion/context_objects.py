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
            "atom_to_token_idx": torch.arange(self.n_atoms, device=self.device)
            .long()
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        }

    def get_atom_idx(self, input_feature_dict):
        return input_feature_dict["atom_to_token_idx"]

    def is_expand_needed(self, atom_idx):
        return (
            atom_idx.dim() == 2
            and self.trunk_embeddings["s_trunk"].dim() == 4
            and self.trunk_embeddings["s_trunk"].shape[1] == 1
        )

    def is_expand_needed_s_inputs(self, atom_idx):
        return (
            atom_idx.dim() == 2
            and self.s_inputs is not None
            and self.s_inputs.dim() == 4
            and self.s_inputs.shape[1] == 1
        )

    def expand_atom_idx(self, atom_idx):
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
        if s_inputs is None and self.override_input_features is not None:
            logger.debug(f"[DEBUG][get_s_inputs] override_input_features keys: {list(self.override_input_features.keys()) if self.override_input_features is not None else None}")
            s_inputs = self.override_input_features.get("s_inputs")
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
            c_s_inputs_dim = self.stage_cfg["model_architecture"]["c_s_inputs"]
            s_inputs = torch.zeros(
                (batch_size, n_tokens, c_s_inputs_dim), device=self.device
            )
        return s_inputs

    def get_z_trunk(self):
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
        return z_trunk
