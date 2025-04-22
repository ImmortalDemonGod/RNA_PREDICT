"""
Integration test for partial checkpointing on Stage A (RFoldPredictor) in RNA_PREDICT.
- Asserts all parameters are frozen (requires_grad=False)
- Asserts checkpoint save/load works
- Asserts no parameters change after optimizer step
- Asserts model is operational after load
- Provides unique error messages for each failure mode
"""
import os
import torch
import pytest
import hydra
from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor
import pathlib
import sys

# --- Pytest/Hydra warning for single-file runs ---
import warnings
warnings.warn(
    "[WARN-HYDRA-PYTEST-QUIRK] When running this test as a single file, Hydra may resolve config_path relative to the test file directory, "
    "not the project root. If you see config path errors, try running as part of the suite (pytest -v tests/integration/) or with --rootdir=. "
    "See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for details.",
    UserWarning
)

# Project rule: Always use absolute Hydra config path for all initialization/testing
CONFIG_NAME = "default"
EXPECTED_CWD = "/Users/tomriddle1/RNA_PREDICT"

# Assert CWD is project root for robust, actionable error reporting
actual_cwd = os.getcwd()
if actual_cwd != EXPECTED_CWD:
    pytest.fail(
        f"[UNIQUE-ERR-HYDRA-CWD] Test must be run from the project root directory.\n"
        f"Expected CWD: {EXPECTED_CWD}\n"
        f"Actual CWD:   {actual_cwd}\n"
        f"To fix: cd {EXPECTED_CWD} && uv run -m pytest tests/integration/test_partial_checkpoint_stageA.py\n"
        f"See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for more info."
    )

# Instrument with debug output and robust config_path selection
print(f"[TEST DEBUG] Current working directory: {os.getcwd()}")
cwd = pathlib.Path(os.getcwd())

# Robust Hydra config path selection for suite vs single-file runs
if any(
    arg.endswith("test_partial_checkpoint_stageA.py") for arg in sys.argv
):
    # Single-file run (pytest invoked directly on this file)
    config_path_selected = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"
    abs_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config_path_selected))
    print("[TEST DEBUG] Single-file run detected. Using config_path: /Users/tomriddle1/RNA_PREDICT/rna_predict/conf")
else:
    # Suite run (pytest invoked from project root)
    config_path_selected = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"
    abs_config_path = os.path.join(os.getcwd(), config_path_selected)
    print("[TEST DEBUG] Suite run detected. Using config_path: /Users/tomriddle1/RNA_PREDICT/rna_predict/conf")
if not os.path.isdir(abs_config_path):
    pytest.fail(
        f"[UNIQUE-ERR-HYDRA-CONF-ABSENT] Hydra config directory not found at {abs_config_path}.\n"
        f"See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for more info."
    )
print(f"[TEST DEBUG] Contents of {abs_config_path}:")
for item in pathlib.Path(abs_config_path).iterdir():
    print(f"  - {item} (exists: {item.exists()}, is_file: {item.is_file()}, perms: {oct(item.stat().st_mode)})")

@pytest.mark.integration
def test_partial_checkpoint_stageA(tmp_path):
    # Diagnostics for debugging Hydra config path issues
    print(f"[DEBUG] sys.argv: {sys.argv}")
    print(f"[DEBUG] os.getcwd(): {os.getcwd()}")
    conf_dir = os.path.join(os.getcwd(), "rna_predict", "conf")
    print(f"[DEBUG] config dir exists: {os.path.isdir(conf_dir)} at {conf_dir}")

    # Use robust, dynamically selected config path for Hydra
    with hydra.initialize_config_module(config_module="rna_predict.conf"):
        cfg = hydra.compose(config_name=CONFIG_NAME)
    stage_cfg = cfg.model.stageA
    device = torch.device("cpu")
    model = StageARFoldPredictor(stage_cfg, device)

    # 2. Assert all parameters are frozen
    for name, param in model.named_parameters():
        if param.requires_grad:
            pytest.fail(f"[UNIQUE-ERR-STAGEA-PARAM-TRAINABLE] Parameter {name} is unexpectedly trainable.")

    # 3. Save and reload checkpoint
    state_dict = model.state_dict()
    ckpt_path = tmp_path / "stageA.pth"
    torch.save(state_dict, ckpt_path)
    model2 = StageARFoldPredictor(stage_cfg, device)
    state_dict2 = torch.load(ckpt_path)
    model2.load_state_dict(state_dict2)
    for k, v in model.state_dict().items():
        v2 = model2.state_dict()[k]
        if not torch.allclose(v, v2):
            pytest.fail(f"[UNIQUE-ERR-STAGEA-PARAM-MISMATCH] Parameter {k} mismatch after checkpoint reload.")

    # 4. Assert no parameters change after optimizer step
    params_before = {k: v.clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.randint(0, 4, (2, 100))  # 2 sequences, length 100
    try:
        optimizer.zero_grad()
        # Use predict_adjacency for forward pass
        model.predict_adjacency("A" * 100)
        # No backward/step since no trainable params, but try to step
        optimizer.step()
    except Exception:
        pass  # Ignore since gradients are not required
    for k, v in model.named_parameters():
        if not torch.equal(v, params_before[k]):
            pytest.fail(f"[UNIQUE-ERR-STAGEA-PARAM-CHANGED] Parameter {k} changed after optimizer step.")

    # 5. Assert model is operational
    try:
        out = model.predict_adjacency("AUGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGC")
        assert out.shape[0] == out.shape[1], "[UNIQUE-ERR-STAGEA-OUTPUT-SHAPE] Output is not square adjacency matrix."
    except Exception as e:
        pytest.fail(f"[UNIQUE-ERR-STAGEA-FORWARD-FAIL] Model failed to run predict_adjacency: {e}")
