"""
Bridging and shape/feature validation utilities for Stage D pipeline.
Extracted from run_stageD.py for code quality and modularity.
"""
from typing import Any, Dict, Optional
import torch

def check_and_bridge_embeddings(
    trunk_embeddings: Dict[str, torch.Tensor],
    features: Dict[str, torch.Tensor],
    input_feature_dict: Dict[str, Any],
    coords: torch.Tensor,
    atom_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Defensive shape checks and bridging for atom count and feature mismatches.
    Mutates trunk_embeddings and features in-place as needed.
    Raises ValueError if mismatches are not bridgeable.
    """
    n_atoms = (
        coords.shape[1] if hasattr(coords, "shape") and len(coords.shape) > 1 else None
    )
    err_prefix = "[ERR-STAGED-SHAPE-001]"
    if n_atoms is not None:
        skip_keys = ["atom_metadata", "sequence", "ref_space_uid"]
        # Check trunk_embeddings
        for key, value in list(trunk_embeddings.items()):
            if key in skip_keys or not isinstance(value, torch.Tensor):
                continue
            if value.shape[1] != n_atoms:
                if key in ["s_trunk", "s_inputs", "pair"]:
                    continue
                raise ValueError(
                    f"{err_prefix} trunk_embeddings['{key}'] atom dim ({value.shape[1]}) != n_atoms ({n_atoms})"
                )
        # Check features
        for key, value in list(features.items()):
            if key in skip_keys or not isinstance(value, torch.Tensor):
                continue
            if value.shape[1] != n_atoms:
                if key in ["s_trunk", "s_inputs", "pair"]:
                    continue
                raise ValueError(
                    f"{err_prefix} features['{key}'] atom dim ({value.shape[1]}) != n_atoms ({n_atoms})"
                )
    # Copy all features to input_feature_dict to ensure they're available to the diffusion model
    for key, value in features.items():
        if key not in input_feature_dict:
            input_feature_dict[key] = value
