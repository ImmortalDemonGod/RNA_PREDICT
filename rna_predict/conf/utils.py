"""Utilities for Hydra configuration management."""

import os
from pathlib import Path
from typing import Optional, Union, Any, cast

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf, Container

from .config_schema import RNAConfig, validate_config

def get_config(config_path: Optional[str] = None,
               config_name: str = "default",
               overrides: Optional[list] = None) -> RNAConfig:
    """Get configuration using Hydra.

    Args:
        config_path: Path to config directory, relative to project root
        config_name: Name of the config file to use (without .yaml)
        overrides: List of Hydra overrides (e.g. ["training.batch_size=64"])

    Returns:
        Loaded and validated configuration object
    """
    if config_path is None:
        config_path = str(Path(__file__).parent)

    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])

    # Validate the config
    validate_config(cast(Union[dict, RNAConfig], cfg))
    return cast(RNAConfig, cfg)

def save_config(config: Union[RNAConfig, DictConfig],
                save_path: str,
                resolve: bool = True) -> None:
    """Save configuration to disk.

    Args:
        config: Configuration object to save
        save_path: Path to save the config to
        resolve: Whether to resolve interpolations before saving
    """
    # Convert to OmegaConf if needed
    if not OmegaConf.is_config(config):
        config = OmegaConf.structured(config)

    # Resolve interpolations if requested
    if resolve:
        container = OmegaConf.to_container(config, resolve=True)
        config = cast(Union[RNAConfig, DictConfig], OmegaConf.create(container))

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save config
    OmegaConf.save(config=config, f=save_path)

def get_run_dir() -> str:
    """Get the Hydra run directory for the current job.

    Returns:
        Absolute path to the run directory
    """
    if HydraConfig.initialized():
        return HydraConfig.get().run.dir
    return get_original_cwd()

def resolve_path(path: str) -> str:
    """Resolve a path relative to the original working directory.

    Args:
        path: Path to resolve

    Returns:
        Absolute path
    """
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)

def get_overrides() -> list:
    """Get the overrides used for the current run.

    Returns:
        List of override strings
    """
    if not HydraConfig.initialized():
        return []
    return HydraConfig.get().overrides.task

def update_config(config: Union[RNAConfig, DictConfig], updates: dict[str, Any]) -> RNAConfig:
    """Update configuration with new values.

    Args:
        config: Configuration object to update
        updates: Dictionary of updates (nested keys separated by dots)

    Returns:
        Updated configuration

    Example:
        >>> cfg = update_config(cfg, {
        ...     "training.batch_size": 64,
        ...     "model.stageA.dropout": 0.5
        ... })
    """
    try:
        # Convert to OmegaConf if needed
        if not OmegaConf.is_config(config):
            config = OmegaConf.structured(config)

        # Apply updates
        for key, value in updates.items():
            print(f"[DEBUG] Attempting to update key: {key} with value: {value}")
            try:
                # Special handling for StageA dropout validation
                if key == "model.stageA.dropout" and (value < 0.0 or value > 1.0):
                    print(f"[DEBUG] Raising ValueError for dropout: {value}")
                    raise ValueError(f"dropout must be between 0 and 1, got {value}")

                # Special handling for device validation
                if key == "model.stageA.device" and value not in ["cuda", "cpu", "mps"]:
                    print(f"[DEBUG] Raising ValueError for device: {value}")
                    raise ValueError(f"device must be 'cuda', 'cpu', or 'mps', got {value}")

                # Use a try-except block to catch any errors during update
                try:
                    OmegaConf.update(cast(Container, config), key, value)
                    print(f"[DEBUG] Updated {key} to {value}")
                except Exception as e:
                    print(f"[ERROR] Failed to update {key} to {value}: {e}")
                    raise ValueError(f"Failed to update {key} to {value}: {e}") from e
            except Exception as e:
                print(f"[ERROR] Error processing update for {key}: {e}")
                raise

        # Validate updated config with a try-except block
        print(f"[DEBUG] Validating config after updates: {updates}")
        try:
            validate_config(cast(Union[dict, RNAConfig], config))
        except Exception as e:
            print(f"[ERROR] Config validation failed: {e}")
            raise ValueError(f"Config validation failed after updates: {e}") from e

        return cast(RNAConfig, config)
    except Exception as e:
        print(f"[ERROR] Unexpected error in update_config: {e}")
        raise