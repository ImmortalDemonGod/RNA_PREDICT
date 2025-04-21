import pytest
from omegaconf import OmegaConf
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import DiffusionManagerConfig


def test_diffusion_manager_config_section_vs_full():
    """
    Test that passing only the section (not full config) to DiffusionManagerConfig.from_hydra_cfg raises the expected ValueError.
    This covers the unique error for missing 'stageD' group.
    """
    # Simulate section config (missing top-level 'stageD')
    section_cfg = OmegaConf.create({
        'diffusion': {
            'device': 'cpu',
            'inference': {'num_steps': 2, 'temperature': 1.0},
            'debug_logging': False
        }
    })
    with pytest.raises(ValueError, match="Config missing required 'stageD' group"):
        DiffusionManagerConfig.from_hydra_cfg(section_cfg)

    # Simulate full config (should not raise)
    full_cfg = OmegaConf.create({
        'stageD': {
            'diffusion': {
                'device': 'cpu',
                'inference': {'num_steps': 2, 'temperature': 1.0},
                'debug_logging': False
            }
        }
    })
    # Should not raise
    try:
        DiffusionManagerConfig.from_hydra_cfg(full_cfg)
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError: {e}")
