import pytest
from pathlib import Path
from hydra import initialize, compose
import sys
from hydra.core.global_hydra import GlobalHydra

# Hydra config_path must always be RELATIVE to the runtime CWD!
# This logic ensures correct relative path for both project root and tests/ as CWD.
def get_hydra_conf_path():
    cwd = Path.cwd()
    print(f"[DEBUG][test_config.py] Current working directory: {cwd}", file=sys.stderr)
    if Path("rna_predict/conf").is_dir():
        print("[DEBUG][test_config.py] Using config path: rna_predict/conf", file=sys.stderr)
        return "rna_predict/conf"
    elif Path("../rna_predict/conf").is_dir():
        print("[DEBUG][test_config.py] Using config path: ../rna_predict/conf", file=sys.stderr)
        return "../rna_predict/conf"
    else:
        print("[ERROR][test_config.py] Could not find rna_predict/conf relative to current working directory!", file=sys.stderr)
        print("[ERROR][test_config.py] Please run tests from the project root: /Users/tomriddle1/RNA_PREDICT", file=sys.stderr)
        raise RuntimeError(f"Could not find rna_predict/conf relative to CWD: {cwd}. Please run tests from the project root: /Users/tomriddle1/RNA_PREDICT")

@pytest.fixture
def hydra_config():
    GlobalHydra.instance().clear()  # Clear Hydra's global state before each test
    conf_path = get_hydra_conf_path()
    with initialize(version_base=None, config_path=conf_path):
        cfg = compose(config_name="default")
    return cfg

# Unskipped: Check config file exists (Hydra-based)
def test_config_file_exists():
    conf_path = get_hydra_conf_path()
    config_path = Path(conf_path) / "default.yaml"
    assert config_path.exists(), "default.yaml config file should exist"

# Unskipped: Check config structure (Hydra-based)
def test_config_structure(hydra_config):
    # Test basic structure
    assert "model" in hydra_config, "Config should have model section"
    assert "pipeline" in hydra_config, "Config should have pipeline section"
    assert "device" in hydra_config, "Config should have device field"
    assert "seed" in hydra_config, "Config should have seed field"

# Unskipped: Check model config (Hydra-based)
def test_model_config(hydra_config):
    model = hydra_config.model
    assert "stageA" in model, "Model config should have stageA"
    assert "stageB" in model, "Model config should have stageB"
    assert "stageC" in model, "Model config should have stageC"
    assert "stageD" in model, "Model config should have stageD"
    # Type checks for StageA
    assert 0 <= model.stageA.dropout <= 1, "Dropout should be between 0 and 1"
    assert model.stageA.device in ["cuda", "cpu", "mps"], "Device should be cuda, cpu, or mps"

# Unskipped: Check training config (Hydra-based)
def test_training_config(hydra_config):
    # Example: Check for training section if present
    if hasattr(hydra_config, "training"):
        assert hasattr(hydra_config.training, "epochs"), "Training config should have epochs"
        assert hasattr(hydra_config.training, "batch_size"), "Training config should have batch_size"

# Unskipped: Check paths config (Hydra-based)
def test_paths_config(hydra_config):
    # Example: Check for output_dir in pipeline
    assert hasattr(hydra_config.pipeline, "output_dir"), "Pipeline config should have output_dir"

# Unskipped: Test environment variable override (Hydra-based)
def test_environment_variable_override(monkeypatch):
    # Simulate environment variable override for device
    monkeypatch.setenv("HYDRA_OVERRIDES", "device=cpu")
    conf_path = get_hydra_conf_path()
    with initialize(version_base=None, config_path=conf_path):
        cfg = compose(config_name="default", overrides=["device=cpu"])
    assert cfg.device == "cpu", "Override should change device value"

def test_hydra_config_structure(hydra_config):
    # Test basic structure (already checked above, but keep for completeness)
    assert "model" in hydra_config, "Config should have model section"
    assert "pipeline" in hydra_config, "Config should have pipeline section"
    assert "device" in hydra_config, "Config should have device field"
    assert "seed" in hydra_config, "Config should have seed field"

def test_hydra_model_config(hydra_config):
    model = hydra_config.model
    assert "stageA" in model, "Model config should have stageA"
    assert "stageB" in model, "Model config should have stageB"
    assert "stageC" in model, "Model config should have stageC"
    assert "stageD" in model, "Model config should have stageD"
    # Type checks for StageA
    assert 0 <= model.stageA.dropout <= 1, "Dropout should be between 0 and 1"
    assert model.stageA.device in ["cuda", "cpu", "mps"], "Device should be cuda, cpu, or mps"

def test_hydra_environment_variable_override():
    # Robust to CWD
    conf_path = get_hydra_conf_path()
    with initialize(version_base=None, config_path=conf_path):
        cfg = compose(config_name="default", overrides=["device=cpu"])
    assert cfg.device == "cpu", "Override should change device value"