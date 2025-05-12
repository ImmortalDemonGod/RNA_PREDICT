"""
Config validation and flattening utilities for Stage D pipeline.
Extracted from run_stageD.py for cohesion and Hydra best practices.
"""
from typing import Any, Dict, Union
from omegaconf import DictConfig

def validate_and_extract_stageD_config(cfg: DictConfig):
    """Validates and extracts the Stage D config. Raises ValueError if missing."""
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageD"):
        raise ValueError("Configuration must contain model.stageD section")
    return cfg.model.stageD


def flatten_stageD_config_to_dict(stage_cfg: DictConfig) -> Union[Dict[str, Any], list[Any], str, Any]:
    """Flattens the Stage D config (OmegaConf) to a plain dict for the diffusion model, omitting tensor fields."""
    import omegaconf
    # Use OmegaConf.to_container to convert to dict
    # Exclude tensor fields (handled separately)
    return omegaconf.OmegaConf.to_container(stage_cfg, resolve=True)


# --- CONFIG & VALIDATION UTILS (moved from run_stageD.py) ---

def _print_config_debug(cfg):
    """
    Print key config structure for debugging. Refactored to minimize nesting and improve readability.
    """
    print("[CONFIG DEBUG] Top-level keys:", list(cfg.keys()) if hasattr(cfg, 'keys') else dir(cfg))
    for group in ["model", "stageD", "diffusion"]:
        val = cfg.get(group, None) if hasattr(cfg, 'get') else getattr(cfg, group, None)
        if val is not None:
            print(f"[CONFIG DEBUG] {group} keys:", list(val.keys()) if hasattr(val, 'keys') else dir(val))


def _validate_and_extract_stageD_config(cfg):
    """
    Validate and extract stageD config. Split into helpers to reduce complexity.
    """
    stage_cfg = _extract_stageD(cfg)
    _validate_required_stageD_params(stage_cfg)
    return stage_cfg


def _extract_stageD(cfg):
    if hasattr(cfg, "model") and hasattr(cfg.model, "stageD"):
        return cfg.model.stageD
    elif hasattr(cfg, "stageD"):
        return cfg.stageD
    else:
        raise ValueError("Missing stageD section in config.")


def _validate_required_stageD_params(stage_cfg):
    """Validate required parameters for Stage D, with fallbacks for missing values."""
    import logging
    log = logging.getLogger(__name__)

    # Define required keys and default values
    required_with_defaults = {
        "diffusion": {"enabled": True, "mode": "inference", "device": "cpu"},
        "model_architecture": {"c_s": 384, "c_z": 128, "c_s_inputs": 449, "c_atom": 128, "c_noise_embedding": 32, "sigma_data": 16.0},
        "mode": "inference",
        "device": "cpu"
    }

    from omegaconf import OmegaConf

    # Check for missing keys and provide defaults
    for key, default_value in required_with_defaults.items():
        if not hasattr(stage_cfg, key):
            log.warning(f"stageD config missing required key: {key}, using default value")
            setattr(stage_cfg, key, OmegaConf.create(default_value))


def _get_debug_logging(stage_cfg):
    return getattr(stage_cfg, "debug_logging", False)


def _validate_and_extract_test_data_cfg(cfg):
    # This function is already simple; no refactor needed for complexity.
    sequence_str = getattr(cfg, "test_sequence", "ACGU")
    if not hasattr(cfg, "atoms_per_residue"):
        raise ValueError("Config missing required 'atoms_per_residue' field. Please check your Hydra config.")
    atoms_per_residue = cfg.atoms_per_residue
    return sequence_str, atoms_per_residue


def _extract_diffusion_dims(stage_cfg):
    model_arch = getattr(stage_cfg, "model_architecture", None)
    if model_arch is None and isinstance(stage_cfg, dict):
        model_arch = stage_cfg.get("model_architecture", None)
    c_s = getattr(model_arch, "c_s", None) if hasattr(model_arch, "c_s") else model_arch.get("c_s", None) if isinstance(model_arch, dict) else None
    c_s_inputs = getattr(model_arch, "c_s_inputs", None) if hasattr(model_arch, "c_s_inputs") else model_arch.get("c_s_inputs", None) if isinstance(model_arch, dict) else None
    c_z = getattr(model_arch, "c_z", None) if hasattr(model_arch, "c_z") else model_arch.get("c_z", None) if isinstance(model_arch, dict) else None
    return c_s, c_s_inputs, c_z
