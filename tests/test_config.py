import pytest
from pathlib import Path
from hydra import initialize, compose

@pytest.fixture
def hydra_config():
    with initialize(version_base=None, config_path="../rna_predict/conf"):
        cfg = compose(config_name="default")
    return cfg

@pytest.mark.skip(reason="Project now uses Hydra for configuration management")
def test_config_file_exists():
    """This test is skipped because the project now uses Hydra for configuration management."""
    config_path = Path("conf/config.yaml")
    assert config_path.exists(), "Configuration file should exist"

@pytest.mark.skip(reason="Project now uses Hydra for configuration management")
def test_config_structure():
    """This test is skipped because the project now uses Hydra for configuration management."""
    pass

@pytest.mark.skip(reason="Project now uses Hydra for configuration management")
def test_model_config():
    """This test is skipped because the project now uses Hydra for configuration management."""
    pass

@pytest.mark.skip(reason="Project now uses Hydra for configuration management")
def test_training_config():
    """This test is skipped because the project now uses Hydra for configuration management."""
    pass

@pytest.mark.skip(reason="Project now uses Hydra for configuration management")
def test_paths_config():
    """This test is skipped because the project now uses Hydra for configuration management."""
    pass

@pytest.mark.skip(reason="Project now uses Hydra for configuration management")
def test_environment_variable_override():
    """This test is skipped because the project now uses Hydra for configuration management."""
    pass

def test_hydra_config_structure(hydra_config):
    """Test that the Hydra configuration has the expected structure."""
    # Test basic structure
    assert "model" in hydra_config, "Config should have model section"
    assert "pipeline" in hydra_config, "Config should have pipeline section"
    assert "device" in hydra_config, "Config should have device field"
    assert "seed" in hydra_config, "Config should have seed field"

def test_hydra_model_config(hydra_config):
    """Test that the model configuration has the expected structure."""
    model = hydra_config.model
    assert "stageA" in model, "Model config should have stageA"
    assert "stageB" in model, "Model config should have stageB"
    assert "stageC" in model, "Model config should have stageC"
    assert "stageD" in model, "Model config should have stageD"

    # Type checks for StageA
    assert 0 <= model.stageA.dropout <= 1, "Dropout should be between 0 and 1"
    assert model.stageA.device in ["cuda", "cpu", "mps"], "Device should be cuda, cpu, or mps"

def test_hydra_environment_variable_override():
    """Test that Hydra configuration values can be overridden."""
    # Initialize Hydra with an override
    with initialize(version_base=None, config_path="../rna_predict/conf"):
        cfg = compose(config_name="default", overrides=["device=cpu"])

    # Check that the override was applied
    assert cfg.device == "cpu", "Override should change device value"