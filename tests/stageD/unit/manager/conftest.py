"""Test fixtures for Stage D manager tests."""
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import pytest

from rna_predict.conf.config_schema import StageDConfig


@pytest.fixture
def hydra_cfg_factory():
    """Factory fixture for creating Hydra configs with overrides.

    Returns:
        Callable: Factory function that takes override kwargs and returns a DictConfig
    """
    def _factory(**overrides) -> DictConfig:
        # Register base config
        cs = ConfigStore.instance()
        cs.store(name="stageD", node=StageDConfig)

        # Create base config
        base_cfg = OmegaConf.structured(StageDConfig)

        # Process overrides to ensure proper nesting
        processed_overrides = {}

        # Handle diffusion parameters specifically
        if "diffusion" in overrides:
            diffusion_overrides = overrides.pop("diffusion")

            # Check if num_steps is in diffusion overrides and move it to inference.num_steps
            if "num_steps" in diffusion_overrides:
                if "inference" not in processed_overrides:
                    processed_overrides["inference"] = {}
                processed_overrides["inference"]["num_steps"] = diffusion_overrides.pop("num_steps")

            # Check if temperature is in diffusion overrides and move it to inference.temperature
            if "temperature" in diffusion_overrides:
                if "inference" not in processed_overrides:
                    processed_overrides["inference"] = {}
                processed_overrides["inference"]["temperature"] = diffusion_overrides.pop("temperature")

            # Add any remaining diffusion overrides
            for k, v in diffusion_overrides.items():
                processed_overrides[k] = v

        # Add any other overrides
        for k, v in overrides.items():
            if k != "diffusion":
                processed_overrides[k] = v

        # Create override config
        override_cfg = OmegaConf.create(processed_overrides)

        # Merge with base config
        diffusion_cfg = OmegaConf.merge(base_cfg, override_cfg)

        # Set up the full config tree for ProtenixDiffusionManager
        cfg = OmegaConf.create({
            "stageD": {"diffusion": diffusion_cfg},
        })

        # Add required attributes for test expectations at the top level for test compatibility
        cfg.num_inference_steps = diffusion_cfg.diffusion.inference.num_steps if hasattr(diffusion_cfg, "diffusion") and hasattr(diffusion_cfg.diffusion, "inference") and hasattr(diffusion_cfg.diffusion.inference, "num_steps") else 2
        cfg.temperature = diffusion_cfg.diffusion.inference.temperature if hasattr(diffusion_cfg, "diffusion") and hasattr(diffusion_cfg.diffusion, "inference") and hasattr(diffusion_cfg.diffusion.inference, "temperature") else 1.0
        cfg.device = diffusion_cfg.device if hasattr(diffusion_cfg, "device") else "cpu"

        return cfg

    return _factory


@pytest.fixture
def trunk_embeddings_factory():
    """Factory fixture for creating mock trunk embeddings tensors.

    Returns:
        Callable: Factory function that takes batch, length, feat dims and returns embeddings dict
    """
    def _factory(batch: int = 1, length: int = 50, feat: int = 384) -> dict:
        # Create mock trunk embeddings with standard shapes, using correct feature dims for test
        trunk_embeddings = {
            "s_trunk": torch.randn(batch, length, 384),
            "z_trunk": torch.randn(batch, length, length, 128),
            "s_inputs": torch.randn(batch, length, 32),
            "pair": torch.randn(batch, length, length, 128),
            # Add all required meta features with dummy tensors (shape [batch, length] or [batch, length, 1] as needed)
            "ref_space_uid": torch.zeros(batch, length, dtype=torch.long),
            "atom_to_token_idx": torch.zeros(batch, length, dtype=torch.long),
            "ref_pos": torch.zeros(batch, length, 3),
            "ref_charge": torch.zeros(batch, length, 1),
            "ref_element": torch.zeros(batch, length, 128),
            "ref_atom_name_chars": torch.zeros(batch, length, 256),
            "ref_mask": torch.ones(batch, length, 1),
            "restype": torch.zeros(batch, length, 32),
            "profile": torch.zeros(batch, length, 32),
            "deletion_mean": torch.zeros(batch, length, 1),
            "sing": torch.zeros(batch, length, 449),
        }
        return trunk_embeddings

    return _factory
