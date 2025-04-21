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
        
        # Apply any overrides
        override_cfg = OmegaConf.create(overrides)
        diffusion_cfg = OmegaConf.merge(base_cfg, override_cfg)
        # Set up the full config tree for ProtenixDiffusionManager
        cfg = OmegaConf.create({
            "stageD": {"diffusion": diffusion_cfg},
        })
        # Add required attributes for test expectations
        # e.g. num_inference_steps and temperature at the top level for test compatibility
        if "inference" in diffusion_cfg:
            inf = diffusion_cfg["inference"]
            cfg.num_inference_steps = inf.get("num_steps", 2)
            cfg.temperature = inf.get("temperature", 1.0)
        else:
            cfg.num_inference_steps = 2
            cfg.temperature = 1.0
        cfg.device = diffusion_cfg.get("device", "cpu")
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
