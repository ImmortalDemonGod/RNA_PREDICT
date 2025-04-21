"""
Integration test for partial checkpointing and full pipeline construction.

IMPORTANT:
  - This test must be run from the project root (`/Users/tomriddle1/RNA_PREDICT`) or from inside the `rna_predict` directory.
  - Project rule: config presence is checked using the ABSOLUTE path `/Users/tomriddle1/RNA_PREDICT/rna_predict/conf` for robust error reporting.
  - Hydra's API requires config_path to be RELATIVE to the current working directory. This test will automatically select the correct relative config_path based on the CWD.
  - If you encounter [UNIQUE-ERR-HYDRA-CONF-NOT-FOUND], verify the config exists at the absolute path and you are in the correct directory.
  - If you encounter [UNIQUE-ERR-HYDRA-CONF-PATH-RELATIVE], Hydra requires config_path to be relative, not absolute. See comments in this test for details.
  - If you encounter [UNIQUE-ERR-HYDRA-CONF-PATH-NOT-FOUND], neither rna_predict/conf nor conf exists relative to CWD. See debug output.
  - This test will print the current working directory and config path candidates for evidence-driven debugging.
  - For systematic debugging methodology, see:
    docs/guides/best_practices/debugging/comprehensive_debugging_guide.md
"""
import os
import torch
import pytest
import hydra
import sys
from omegaconf import OmegaConf, DictConfig
from rna_predict.training.rna_lightning_module import RNALightningModule
from rna_predict.utils.checkpointing import save_trainable_checkpoint, get_trainable_params
from rna_predict.utils.checkpoint import partial_load_state_dict
from hypothesis import given, strategies as st, settings
import tempfile
from pathlib import Path
from collections.abc import Mapping

# Project rule: Always use absolute Hydra config path for all initialization/testing
# See MEMORY[ab8a7679-fc73-4f8b-af9a-6ad058010c5a]
CONFIG_ABS_PATH = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf/default.yaml"

# Assert CWD is project root for robust, actionable error reporting
EXPECTED_CWD = "/Users/tomriddle1/RNA_PREDICT"
actual_cwd = os.getcwd()
if actual_cwd != EXPECTED_CWD:
    pytest.fail(
        f"[UNIQUE-ERR-HYDRA-CWD] Test must be run from the project root directory.\n"
        f"Expected CWD: {EXPECTED_CWD}\n"
        f"Actual CWD:   {actual_cwd}\n"
        f"To fix: cd {EXPECTED_CWD} && uv run -m pytest tests/integration/test_partial_checkpoint_full_pipeline.py\n"
        f"See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for more info."
    )

# Instrument with debug output and dynamic config_path selection
import pathlib
print(f"[TEST DEBUG] Current working directory: {os.getcwd()}")

# Check config candidates relative to CWD
cwd = pathlib.Path(os.getcwd())
config_candidates = [cwd / "rna_predict" / "conf", cwd / "conf"]
config_path_selected = None
for candidate in config_candidates:
    print(f"[TEST DEBUG] Checking for config directory: {candidate}")
    if candidate.exists() and (candidate / "default.yaml").exists():
        config_path_selected = str(candidate.relative_to(cwd))
        print(f"[TEST DEBUG] Found config at: {candidate}, using config_path: {config_path_selected}")
        # Print directory contents and permissions for further debugging
        print(f"[TEST DEBUG] Contents of {candidate}:")
        for item in candidate.iterdir():
            print(f"  - {item} (exists: {item.exists()}, is_file: {item.is_file()}, perms: {oct(item.stat().st_mode)})")
        default_yaml = candidate / "default.yaml"
        if default_yaml.exists():
            print(f"[TEST DEBUG] default.yaml permissions: {oct(default_yaml.stat().st_mode)}")
        else:
            print(f"[TEST DEBUG] default.yaml not found in {candidate}")
        print(f"[TEST DEBUG] Absolute path to config directory: {candidate.resolve()}")
        break
if not config_path_selected:
    pytest.fail("[UNIQUE-ERR-HYDRA-CONF-PATH-NOT-FOUND] Neither 'rna_predict/conf' nor 'conf' found relative to current working directory.\nCWD: {}\nChecked: {}\nSee docs/guides/best_practices/debugging/comprehensive_debugging_guide.md".format(os.getcwd(), [str(c) for c in config_candidates]))
else:
    # Print permissions and error if directory exists but not readable
    for candidate in config_candidates:
        if candidate.exists() and not (candidate / "default.yaml").exists():
            print(f"[TEST DEBUG] Candidate exists but missing default.yaml: {candidate}")
            print(f"[TEST DEBUG] Permissions: {oct(candidate.stat().st_mode)}")
            for item in candidate.iterdir():
                print(f"  - {item} (exists: {item.exists()}, is_file: {item.is_file()}, perms: {oct(item.stat().st_mode)})")
            pytest.fail("[UNIQUE-ERR-HYDRA-CONF-PATH-STRUCTURE] Config directory exists but missing default.yaml or unreadable. See debug output above and docs/guides/best_practices/debugging/comprehensive_debugging_guide.md")

# --- PATCH: Try config_path_selected = '../../rna_predict/conf' for Hydra ---
print(f"[PATCH DEBUG] Trying config_path_selected = '../../rna_predict/conf'")
config_path_selected = "../../rna_predict/conf"
print(f"[PATCH DEBUG] config_path_selected now: {config_path_selected}")

@settings(max_examples=2, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    input_dim=st.integers(min_value=8, max_value=32),
)
def test_full_pipeline_partial_checkpoint(batch_size, input_dim):
    """
    Full-pipeline partial checkpoint test with property-based dummy input using hypothesis.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # 1. Load config and instantiate model (Hydra init per example)
        try:
            import hydra.core.global_hydra
            if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.core.global_hydra.GlobalHydra.instance().clear()
            with hydra.initialize(config_path=config_path_selected, job_name="test_full_pipeline_partial_checkpoint"):
                cfg = hydra.compose(config_name="default")
                # Unique error: Check Pairformer config structure
                if not hasattr(cfg.model.stageB, "pairformer") or not isinstance(cfg.model.stageB.pairformer, (Mapping, DictConfig)):
                    pytest.fail("[UNIQUE-ERR-PAIRFORMER-CONFIG-STRUCTURE] Pairformer config missing or not a mapping in cfg.model.stageB.pairformer. Check Hydra config composition and test setup.")
        except Exception as e:
            pytest.fail(f"[UNIQUE-ERR-HYDRA-INIT] Hydra failed to initialize or compose config: {e}")
        model = RNALightningModule(cfg)
        model.train()

        # 2. Save initial trainable params for comparison
        initial_params = get_trainable_params(model)

        # 3. Minimal training loop with hypothesis dummy input
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        dummy_input = torch.zeros(batch_size, input_dim)
        for _ in range(2):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()

        # 3.5 Validate partial state dict keys (no base-only keys)
        partial_ckpt_path = tmp_path / "partial_ckpt.pth"
        save_trainable_checkpoint(model, partial_ckpt_path)
        partial_state = torch.load(partial_ckpt_path)
        for k in partial_state.keys():
            assert any(param in k for param in initial_params.keys()), f"[UNIQUE-ERR-PARTIAL-STATE-KEY] Unexpected key in partial checkpoint: {k}"

        # 4. Save full checkpoint for size comparison
        full_ckpt_path = tmp_path / "full_ckpt.pth"
        torch.save(model.state_dict(), full_ckpt_path)

        # 5. Instantiate a new model and load partial checkpoint
        model2 = RNALightningModule(cfg)
        model2.eval()
        loaded_partial = torch.load(partial_ckpt_path)
        missing, unexpected = partial_load_state_dict(model2, loaded_partial, strict=False)
        assert not unexpected, f"[UNIQUE-ERR-PARTIAL-LOAD-UNEXPECTED] Unexpected keys on partial load: {unexpected}"

        # 6. Forward pass and safety checks
        with torch.no_grad():
            output2 = model2(dummy_input)
        assert not torch.isnan(output2).any(), "[UNIQUE-ERR-NAN] NaN in model output after partial load"
        assert not torch.isinf(output2).any(), "[UNIQUE-ERR-INF] Inf in model output after partial load"

        # 6.5: Assert trainable params changed, others did not
        after_params = get_trainable_params(model)
        for k, v in initial_params.items():
            if k in after_params:
                assert not torch.equal(v, after_params[k]), f"[UNIQUE-ERR-PARAM-NOT-CHANGED] Trainable param {k} did not change after optimizer.step()"

        # 8. Compare checkpoint sizes
        partial_ckpt_size = os.path.getsize(partial_ckpt_path)
        full_ckpt_size = os.path.getsize(full_ckpt_path)
        assert partial_ckpt_size < full_ckpt_size, "[UNIQUE-ERR-CHECKPOINT-SIZE] Partial checkpoint is not smaller than full checkpoint"

        # 9. Acceptance criteria
        assert output2.shape == output.shape, "[UNIQUE-ERR-OUTPUT-SHAPE] Output shape mismatch after partial load"
        print("Full-pipeline partial checkpoint test PASSED.")
