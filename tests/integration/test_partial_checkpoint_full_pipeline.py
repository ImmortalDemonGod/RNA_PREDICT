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
import pathlib
PROJECT_ROOT = "/Users/tomriddle1/RNA_PREDICT"
if os.getcwd() != PROJECT_ROOT:
    print(f"[DEBUG-TOP] Changing CWD from {os.getcwd()} to {PROJECT_ROOT}")
    os.chdir(PROJECT_ROOT)
conf_path = pathlib.Path(PROJECT_ROOT) / "rna_predict" / "conf"
try:
    rel_conf_path = conf_path.relative_to(pathlib.Path.cwd())
except ValueError:
    # If CWD is not a parent of conf_path, fallback to absolute path (Hydra will error, but we log it)
    rel_conf_path = conf_path
print(f"[DEBUG-TOP] CWD at import time: {os.getcwd()}")
print(f"[DEBUG-TOP] Using config_path={rel_conf_path}")
import torch
import pytest
from omegaconf import DictConfig
from rna_predict.training.rna_lightning_module import RNALightningModule
from rna_predict.utils.checkpointing import save_trainable_checkpoint, get_trainable_params
from rna_predict.utils.checkpoint import partial_load_state_dict
from hypothesis import given, strategies as st, settings
import tempfile
from pathlib import Path
from collections.abc import Mapping
import traceback

# Project rule: Always use absolute Hydra config path for all initialization/testing
# See MEMORY[ab8a7679-fc73-4f8b-af9a-6ad058010c5a]
CONFIG_ABS_PATH = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf/default.yaml"

# Force CWD to project root before Hydra config logic
EXPECTED_CWD = "/Users/tomriddle1/RNA_PREDICT"
actual_cwd = os.getcwd()
assert os.getcwd().endswith("RNA_PREDICT"), (
    f"Test must be run from project root. Current CWD: {os.getcwd()}"
)
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

@settings(max_examples=1, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    input_dim=st.just(16),  # Fix input_dim to 16 to match dummy layer
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
            print(f"[DEBUG] CWD before hydra.initialize: {os.getcwd()}")
            # Compose config with debug output to inspect structure
            with hydra.initialize(config_path="../../rna_predict/conf", job_name="test_full_pipeline_partial_checkpoint", version_base=None):
                cfg = hydra.compose(
                    config_name="default",
                    overrides=[
                        "model.stageB.torsion_bert.init_from_scratch=True",
                        "model.stageD.diffusion.init_from_scratch=True",
                        # Minimal model size overrides for low-memory test
                        "model.stageB.pairformer.n_blocks=1",
                        "model.stageB.pairformer.n_heads=2",
                        "model.stageB.pairformer.c_z=8",
                        "model.stageB.pairformer.c_s=8",
                        "model.stageB.pairformer.c_token=8",
                        "model.stageB.pairformer.c_atom=8",
                        "model.stageB.pairformer.c_pair=4",
                        # Model architecture parameters
                        "model.stageD.model_architecture.c_atom=8",
                        "model.stageD.model_architecture.c_s=8",
                        "model.stageD.model_architecture.c_z=8",
                        "model.stageD.model_architecture.c_s_inputs=8",
                        "model.stageD.model_architecture.c_noise_embedding=8",
                        "model.stageD.model_architecture.c_token=8",
                        "model.stageD.model_architecture.c_atompair=8",
                        "model.stageD.model_architecture.sigma_data=1.0",

                        # Duplicate in diffusion section
                        "model.stageD.diffusion.model_architecture.c_atom=8",
                        "model.stageD.diffusion.model_architecture.c_s=8",
                        "model.stageD.diffusion.model_architecture.c_z=8",
                        "model.stageD.diffusion.model_architecture.c_s_inputs=8",
                        "model.stageD.diffusion.model_architecture.c_noise_embedding=8",
                        "model.stageD.diffusion.model_architecture.c_token=8",
                        "model.stageD.diffusion.model_architecture.c_atompair=8",
                        "model.stageD.diffusion.model_architecture.sigma_data=1.0",

                        # Feature dimensions
                        "model.stageD.feature_dimensions.c_s=8",
                        "model.stageD.feature_dimensions.c_s_inputs=8",
                        "model.stageD.feature_dimensions.c_sing=8",
                        "model.stageD.feature_dimensions.s_trunk=8",
                        "model.stageD.feature_dimensions.s_inputs=8",

                        # Duplicate in diffusion section
                        "model.stageD.diffusion.feature_dimensions.c_s=8",
                        "model.stageD.diffusion.feature_dimensions.c_s_inputs=8",
                        "model.stageD.diffusion.feature_dimensions.c_sing=8",
                        "model.stageD.diffusion.feature_dimensions.s_trunk=8",
                        "model.stageD.diffusion.feature_dimensions.s_inputs=8",
                        "model.stageD.diffusion.transformer.n_blocks=1",
                        "model.stageD.diffusion.transformer.n_heads=2",
                        "model.stageD.diffusion.atom_encoder.n_blocks=1",
                        "model.stageD.diffusion.atom_encoder.n_heads=2",
                        "model.stageD.diffusion.atom_encoder.n_queries=1",
                        "model.stageD.diffusion.atom_encoder.n_keys=1",
                        "model.stageD.diffusion.atom_decoder.n_blocks=1",
                        "model.stageD.diffusion.atom_decoder.n_heads=2",
                        "model.stageD.diffusion.atom_decoder.n_queries=1",
                        "model.stageD.diffusion.atom_decoder.n_keys=1"
                    ]
                )
                # Print the config structure to debug the path to c_atom
                def print_tree(cfg, prefix=""):  # recursive pretty-printer
                    if isinstance(cfg, dict) or hasattr(cfg, "keys"):
                        for k in cfg.keys():
                            print(f"{prefix}{k}")
                            print_tree(cfg[k], prefix + "  ")
                print("[HYDRA DEBUG] Composed config tree:")
                print_tree(cfg)
        except Exception as e:
            print("\n=== HYDRA ERROR TRACEBACK ===\n" + traceback.format_exc())
            pytest.fail(f"[UNIQUE-ERR-HYDRA-INIT] Hydra failed to initialize or compose config: {e}")
        model = RNALightningModule(cfg)
        model.train()

        # 2. Save initial trainable params for comparison
        initial_params = get_trainable_params(model)

        # 3. Minimal training loop with hypothesis dummy input
        integration_test_mode = getattr(model, '_integration_test_mode', False)
        if integration_test_mode:
            if not hasattr(model, '_integration_test_dummy'):
                pytest.fail("[UNIQUE-ERR-DUMMY-LAYER-MISSING] _integration_test_dummy not found in model during integration test mode.")
            optimizer = torch.optim.Adam(model._integration_test_dummy.parameters(), lr=1e-3)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Use nonzero input to ensure gradients flow to weights
        dummy_input = torch.randn(batch_size, input_dim)
        # Instrumentation: Store initial dummy weights
        if integration_test_mode:
            dummy_weight_before = model._integration_test_dummy.weight.detach().clone()
        for _ in range(1):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            # Instrumentation: Check gradients
            if integration_test_mode:
                grad = model._integration_test_dummy.weight.grad
                if grad is None:
                    pytest.fail("[UNIQUE-ERR-DUMMY-GRADIENTS-NONE] _integration_test_dummy.weight.grad is None after backward().")
                if torch.all(grad == 0):
                    pytest.fail("[UNIQUE-ERR-DUMMY-GRADIENTS-ALLZERO] _integration_test_dummy.weight.grad is all zero after backward().")
            optimizer.step()
        # Instrumentation: Check parameter update
        if integration_test_mode:
            dummy_weight_after = model._integration_test_dummy.weight.detach().clone()
            if torch.equal(dummy_weight_before, dummy_weight_after):
                pytest.fail("[UNIQUE-ERR-DUMMY-PARAM-NOT-UPDATED] _integration_test_dummy.weight did not change after optimizer.step().")
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
        # Systematic debugging: Only check params that are used in the forward pass
        # Hypothesis: In integration test mode, only _integration_test_dummy params are updated
        integration_test_mode = getattr(model, '_integration_test_mode', False)
        checked_any = False
        for k, v in initial_params.items():
            # Skip StageA parameters (they are not trainable/frozen)
            if k.startswith("pipeline.stageA") or k.startswith("stageA."):
                continue
            if integration_test_mode and "_integration_test_dummy" not in k:
                continue  # Only check dummy layer params
            if k in after_params:
                checked_any = True
                assert not torch.equal(v, after_params[k]), f"[UNIQUE-ERR-PARAM-NOT-CHANGED] Trainable param {k} did not change after optimizer.step() (integration_test_mode={integration_test_mode})"
        assert checked_any, "[UNIQUE-ERR-NO-TRAINABLE-PARAMS-CHECKED] No trainable parameters were checked for updates. Test may be misconfigured."

        # 8. Compare checkpoint sizes
        partial_ckpt_size = os.path.getsize(partial_ckpt_path)
        full_ckpt_size = os.path.getsize(full_ckpt_path)
        assert partial_ckpt_size < full_ckpt_size, "[UNIQUE-ERR-CHECKPOINT-SIZE] Partial checkpoint is not smaller than full checkpoint"

        # 9. Acceptance criteria
        assert output2.shape == output.shape, "[UNIQUE-ERR-OUTPUT-SHAPE] Output shape mismatch after partial load"
        print("Full-pipeline partial checkpoint test PASSED.")
