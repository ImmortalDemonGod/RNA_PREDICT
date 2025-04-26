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

##@snoop
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
    # Print config structure for debugging
    print(f"[DEBUG][CONFIG STRUCTURE] cfg type: {type(cfg)}; keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else dir(cfg)}")
    # Try to extract from standard Hydra config structure
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageD') and hasattr(cfg.model.stageD, 'diffusion'):
        test_residues_per_batch = getattr(cfg.model.stageD.diffusion, 'test_residues_per_batch', None)
    # Try flat config (as in current pipeline)
    if test_residues_per_batch is None and hasattr(cfg, 'test_residues_per_batch'):
        test_residues_per_batch = getattr(cfg, 'test_residues_per_batch', None)
    # Try one more fallback: direct dict
    if test_residues_per_batch is None and isinstance(cfg, dict):
        test_residues_per_batch = cfg.get('test_residues_per_batch', None)
    # Final fallback: use a default value if not set
    if test_residues_per_batch is None:
        # Try to get from diffusion_config if available
        if hasattr(cfg, 'diffusion_config') and isinstance(cfg.diffusion_config, dict):
            test_residues_per_batch = cfg.diffusion_config.get('test_residues_per_batch', 25)
        else:
            # Use a default value as last resort
            test_residues_per_batch = 25
        print(f"[DEBUG][PATCHED] Using default test_residues_per_batch={test_residues_per_batch}")
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
    # Enforce output shape [1, seq_len, 3] or [1, N_sample, seq_len, 3] for inference output
    if coords.dim() == 4:
        # Handle multi-sample case: [1, N_sample, seq_len, 3]
        print(f"[DEBUG][PATCHED] Found 4D coords with shape {coords.shape}. Selecting first sample.")
        assert coords.shape[0] == 1, f"Batch size must be 1, got {coords.shape}"
        assert coords.shape[3] == 3, f"Last dim must be 3, got {coords.shape}"
        # Select the first sample for compatibility with downstream code
        coords = coords[:, 0, :, :]
        print(f"[DEBUG][PATCHED] After selecting first sample, coords shape: {coords.shape}")
    else:
        # Standard case: [1, seq_len, 3]
        assert coords.dim() == 3, f"coords must have 3 dims, got {coords.shape}"
        assert coords.shape[0] == 1, f"Batch size must be 1, got {coords.shape}"
        assert coords.shape[2] == 3, f"Last dim must be 3, got {coords.shape}"
    # Patch: check atom count matches input_features['atom_metadata'] if present
    atom_count = None
    if 'atom_metadata' in context.input_features:
        atom_metadata = context.input_features['atom_metadata']
        if 'atom_type' in atom_metadata:
            if hasattr(atom_metadata['atom_type'], 'shape'):
                atom_count = atom_metadata['atom_type'].shape[0]
            elif isinstance(atom_metadata['atom_type'], list):
                atom_count = len(atom_metadata['atom_type'])
            else:
                atom_count = coords.shape[1]
        elif 'residue_indices' in atom_metadata:
            # Use residue_indices as fallback
            atom_count = len(atom_metadata['residue_indices'])
        else:
            atom_count = coords.shape[1]
    else:
        atom_count = coords.shape[1]
    print(f"[DEBUG][PATCHED] Checking atom count: coords.shape[1]={coords.shape[1]}, atom_count={atom_count}, seq_len={seq_len}")
    assert coords.shape[1] == atom_count, f"Atom count mismatch: expected {atom_count}, got {coords.shape[1]} (seq_len={seq_len})"
    if atom_count != seq_len:
        print(f"[WARN][PATCHED] Atom count ({atom_count}) != residue-level seq_len ({seq_len}). This is expected for atom-level output.")
    logger.debug(f"[StageD][run_inference_mode] coords output shape: {coords.shape}")

    return coords
