"""
Inference mode functions for Stage D diffusion.

This module provides functions for running inference in the Stage D diffusion process.
"""

import logging
import torch
from typing import Any, Dict
from dataclasses import dataclass

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)

logger = logging.getLogger(__name__)

# Always use relative path for Hydra config
CONFIG_PATH = "rna_predict/conf"

def get_pipeline_config():
    from hydra.core.global_hydra import GlobalHydra
    if not GlobalHydra.instance().is_initialized():
        from rna_predict.conf.utils import get_config
        return get_config(config_path=CONFIG_PATH)
    else:
        raise RuntimeError("Hydra is already initialized; config must be passed from caller.")


@dataclass
class InferenceContext:
    """Context for inference operations."""
    diffusion_manager: ProtenixDiffusionManager
    partial_coords: torch.Tensor
    trunk_embeddings_internal: Dict[str, torch.Tensor]
    original_trunk_embeddings_ref: Dict[str, torch.Tensor]
    diffusion_config: Dict[str, Any]
    input_features: Dict[str, Any]
    device: str


def run_inference_mode(
    context: InferenceContext,
    cfg=None,
) -> torch.Tensor:
    """
    Run diffusion in inference mode.

    Args:
        context: Inference context with all required parameters
        cfg: Optional Hydra config

    Returns:
        Refined coordinates tensor
    """
    if cfg is None:
        cfg = get_pipeline_config()
    # PATCH: Hydra best practice: always use config-driven value, never fallback to hardcoded default
    test_residues_per_batch = None
    # Try to extract from standard Hydra config structure
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageD') and hasattr(cfg.model.stageD, 'diffusion'):
        test_residues_per_batch = getattr(cfg.model.stageD.diffusion, 'test_residues_per_batch', None)
    # Fallback: legacy nested dict or flat dict (for older configs/tests)
    if test_residues_per_batch is None and hasattr(cfg, 'diffusion_config') and isinstance(cfg.diffusion_config, dict):
        if 'diffusion' in cfg.diffusion_config and isinstance(cfg.diffusion_config['diffusion'], dict):
            test_residues_per_batch = cfg.diffusion_config['diffusion'].get('test_residues_per_batch', None)
        elif 'test_residues_per_batch' in cfg.diffusion_config:
            test_residues_per_batch = cfg.diffusion_config['test_residues_per_batch']
    # Final fallback: error if not set
    if test_residues_per_batch is None:
        raise ValueError("test_residues_per_batch must be set in Hydra config (model.stageD.diffusion.test_residues_per_batch)")
    seq_len = test_residues_per_batch

    # Note: We no longer need to pass inference_params directly to multi_step_inference
    # as it now reads parameters from the manager's internal config

    # Pass the internal (potentially fixed) copy to the manager
    coords = context.diffusion_manager.multi_step_inference(
        coords_init=context.partial_coords.to(context.device),
        trunk_embeddings=context.trunk_embeddings_internal,
        override_input_features=context.input_features
    )
    # Update the original trunk_embeddings dict with cached s_inputs if it was added
    if (
        "s_inputs" in context.trunk_embeddings_internal
        and "s_inputs" not in context.original_trunk_embeddings_ref
    ):
        logger.debug("Copying cached 's_inputs' back to original dictionary.")
        context.original_trunk_embeddings_ref["s_inputs"] = context.trunk_embeddings_internal[
            "s_inputs"
        ]
    # Enforce output shape [1, seq_len, 3] for inference output
    assert coords.dim() == 3, f"coords must have 3 dims, got {coords.shape}"
    assert coords.shape[0] == 1, f"Batch size must be 1, got {coords.shape}"
    assert coords.shape[2] == 3, f"Last dim must be 3, got {coords.shape}"
    # Optionally, enforce config-driven seq_len if desired
    assert coords.shape[1] == seq_len, f"Atom count must be {seq_len}, got {coords.shape[1]}"
    logger.debug(f"[StageD][run_inference_mode] coords output shape: {coords.shape}")

    return coords
