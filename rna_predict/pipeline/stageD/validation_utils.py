"""
Validation utilities for Stage D pipeline.
Extracted from run_stageD.py to reduce file size and improve cohesion.
"""
from typing import Any, Dict, Optional
from omegaconf import DictConfig
import torch

def validate_run_stageD_inputs(
    cfg: DictConfig,
    coords: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    s_inputs: torch.Tensor,
    input_feature_dict: Dict[str, Any],
    atom_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Validates inputs and config for run_stageD. Raises ValueError on errors."""
    # Validate that the required configuration sections exist
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageD"):
        raise ValueError("Configuration must contain model.stageD section")
    # Defensive check: s_trunk must be residue-level at entry
    if atom_metadata is not None and "residue_indices" in atom_metadata:
        n_atoms = len(atom_metadata["residue_indices"])
        n_residues = (
            len(input_feature_dict.get("sequence", []))
            if input_feature_dict.get("sequence", None) is not None
            else None
        )
        # If sequence not available, estimate from atom count and atoms per residue
        _ = n_atoms // n_residues if n_residues else None
        if s_trunk.shape[1] == n_atoms:
            raise ValueError(
                "[RUNSTAGED ERROR][UNIQUE_CODE_003] s_trunk is atom-level at entry to run_stageD; upstream code must pass residue-level embeddings."
            )
