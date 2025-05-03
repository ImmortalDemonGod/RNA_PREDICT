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
        cfg = OmegaConf.merge(base_cfg, override_cfg)
        
        return cfg
    
    return _factory


@pytest.fixture
def trunk_embeddings_factory():
    """Factory fixture for creating mock trunk embeddings tensors.
    
    Returns:
        Callable: Factory function that takes batch, length, feat dims and returns embeddings dict
    """
    def _factory(batch: int = 1, length: int = 50, feat: int = 64) -> dict:
        # Create mock trunk embeddings with standard shapes
        trunk_embeddings = {
            "s_trunk": torch.randn(batch, length, feat),
            "z_trunk": torch.randn(batch, length, length, feat // 2),
            "s_inputs": torch.randn(batch, length, feat // 4)
        }
        return trunk_embeddings
    
    return _factory 