"""
Configuration utilities for Stage D diffusion.

This module provides functions for handling configuration in the Stage D diffusion process.
"""

import torch
from typing import Any, Dict
from .config_types import DiffusionConfig
from omegaconf import OmegaConf


def get_embedding_dimension(
    diffusion_config: DiffusionConfig, key: str, default_value: int
) -> int:
    """
    Get embedding dimension from diffusion config with fallback.

    Args:
        diffusion_config: DiffusionConfig structured config
        key: Key to look for in config.diffusion_config
        default_value: Default value if key not found

    Returns:
        Embedding dimension
    """
    conditioning_config = diffusion_config.diffusion_config.get("conditioning", {})
    return diffusion_config.diffusion_config.get(key, conditioning_config.get(key, default_value))


def create_fallback_input_features(
    partial_coords: torch.Tensor, diffusion_config: DiffusionConfig, device: str
) -> Dict[str, Any]:
    """
    Create fallback input features when none are provided.

    Args:
        partial_coords: Partial coordinates tensor [B, N_atom, 3]
        diffusion_config: DiffusionConfig structured config
        device: Device to run on

    Returns:
        Dict of fallback input features
    """
    N = partial_coords.shape[1]
    # Get the dimension for s_inputs
    c_s_inputs_dim = diffusion_config.diffusion_config.get("c_s_inputs", None)
    if c_s_inputs_dim is None:
        raise ValueError("Missing config value for 'c_s_inputs'")

    # Get the dimension for ref_element
    ref_element_dim = diffusion_config.diffusion_config.get("ref_element_dim", None)
    if ref_element_dim is None:
        raise ValueError("Missing config value for 'ref_element_dim'")

    # Get the dimension for ref_atom_name_chars
    ref_atom_name_chars_dim = diffusion_config.diffusion_config.get("ref_atom_name_chars_dim", None)
    if ref_atom_name_chars_dim is None:
        raise ValueError("Missing config value for 'ref_atom_name_chars_dim'")

    # Get the dimension for restype and profile
    restype_dim = diffusion_config.diffusion_config.get("restype_dim", None)
    if restype_dim is None:
        raise ValueError("Missing config value for 'restype_dim'")

    return {
        "atom_to_token_idx": torch.arange(N, device=device).unsqueeze(0),
        "ref_pos": partial_coords.to(device),
        "ref_space_uid": torch.arange(N, device=device).unsqueeze(0),
        "ref_charge": torch.zeros(1, N, 1, device=device),
        "ref_element": torch.zeros(1, N, ref_element_dim, device=device),
        "ref_atom_name_chars": torch.zeros(1, N, ref_atom_name_chars_dim, device=device),
        "ref_mask": torch.ones(1, N, 1, device=device),
        "restype": torch.zeros(1, N, restype_dim, device=device),
        "profile": torch.zeros(1, N, restype_dim, device=device),
        "deletion_mean": torch.zeros(1, N, 1, device=device),
        "sing": torch.zeros(1, N, c_s_inputs_dim, device=device),
        # Add s_inputs as well to ensure it's available
        "s_inputs": torch.zeros(1, N, c_s_inputs_dim, device=device),
    }


def validate_stageD_config(cfg):
    if not OmegaConf.is_config(cfg):
        raise ValueError(
            "[UNIQUE-ERR-STAGED-CONFIG-TYPE] Config must be a Hydra DictConfig"
        )
    # PATCH: look for stageD under model
    if "model" not in cfg or "stageD" not in cfg.model:
        raise ValueError(
            "[UNIQUE-ERR-STAGED-MISSING] Config missing required 'model.stageD' group"
        )
    # Handle both single and double nesting of stageD
    stageD_cfg = cfg.model.stageD
    if "diffusion" in stageD_cfg:
        stage_cfg = stageD_cfg.diffusion
    elif "stageD" in stageD_cfg and "diffusion" in stageD_cfg.stageD:
        stage_cfg = stageD_cfg.stageD.diffusion
    else:
        raise ValueError(
            "[UNIQUE-ERR-STAGED-DIFFUSION-MISSING] Config missing required 'diffusion' group in model.stageD. Available keys: "
            + str(list(stageD_cfg.keys()))
        )
    # Check model_architecture block
    if "model_architecture" not in stage_cfg:
        raise ValueError(
            "Config missing required 'model_architecture' block in model.stageD.diffusion!"
        )
    # Ensure no duplicated keys at top level
    forbidden_keys = [
        "sigma_data",
        "c_atom",
        "c_atompair",
        "c_token",
        "c_s",
        "c_z",
        "c_s_inputs",
    ]
    for key in forbidden_keys:
        if key in stage_cfg:
            raise ValueError(
                f"Config key '{key}' should only appear in 'model_architecture', not at the top level of model.stageD.diffusion!"
            )


def parse_diffusion_module_args(stage_cfg):
    """
    Extract all required model dimensions and architectural parameters from the config,
    strictly using config values and never hardcoded fallbacks.
    """
    print("[DEBUG][_parse_diffusion_module_args] stage_cfg:", stage_cfg)
    # Always descend into 'diffusion' if present
    if "diffusion" in stage_cfg:
        print(
            "[DEBUG][_parse_diffusion_module_args] Descending into 'diffusion' section of config."
        )
        base_cfg = stage_cfg["diffusion"]
    else:
        base_cfg = stage_cfg
    # Require model_architecture to be present exactly as specified in the config
    model_architecture = base_cfg.get("model_architecture")
    if model_architecture is None:
        raise ValueError(
            "model_architecture section missing in config! (expected at model.stageD.diffusion.model_architecture, no fallback)"
        )
    # Continue as before, using model_architecture directly
    # Filter out parameters that DiffusionModule doesn't accept
    # Only include parameters that are explicitly accepted by DiffusionModule.__init__
    valid_params = [
        "c_token",
        "c_s",
        "c_z",
        "c_s_inputs",
        "c_atom",
        "c_atompair",
        "c_noise_embedding",
        "sigma_data",
    ]
    diffusion_module_args = {
        k: v for k, v in dict(model_architecture).items() if k in valid_params
    }
    # atom_encoder, atom_decoder, transformer
    for subkey in ["atom_encoder", "atom_decoder", "transformer"]:
        subcfg = base_cfg.get(subkey)
        if subcfg is None:
            raise ValueError(
                f"Required config section '{subkey}' missing in config!"
            )
        diffusion_module_args[subkey] = (
            dict(subcfg) if hasattr(subcfg, "items") else dict(subcfg)
        )
    # Optional blocks_per_ckpt, use_fine_grained_checkpoint, initialization
    for key in ["blocks_per_ckpt", "use_fine_grained_checkpoint", "initialization"]:
        val = base_cfg.get(key, None)
        if val is not None:
            diffusion_module_args[key] = val
    # Memory optimization
    mem_cfg = base_cfg.get("memory", {})
    use_ckpt = mem_cfg.get("use_checkpointing", False)
    blocks_per_ckpt = mem_cfg.get("blocks_per_ckpt") if use_ckpt else None
    use_fine_grained = (
        mem_cfg.get("use_fine_grained_checkpoint", False)
        if blocks_per_ckpt
        else False
    )
    diffusion_module_args.update(
        {
            "blocks_per_ckpt": blocks_per_ckpt,
            "use_fine_grained_checkpoint": use_fine_grained,
        }
    )
    print(
        "[DEBUG][_parse_diffusion_module_args] Final module args:",
        diffusion_module_args,
    )
    return diffusion_module_args
