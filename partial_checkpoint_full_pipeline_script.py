"""
Integration script for partial checkpointing and full pipeline construction in RNA_PREDICT.

Run from the project root: /Users/tomriddle1/RNA_PREDICT
This script:
    1. Asserts correct CWD and config presence.
    2. Loads config and builds RNALightningModule (currently a placeholder/stub).
    3. Saves a "partial" checkpoint (stub logic).
    4. Loads checkpoint using partial_load_state_dict (stub logic).
    5. Runs a forward pass (stub logic).
    6. Compares checkpoint sizes (stub logic).
    7. Emits unique errors/warnings for all stubbed or not-implemented checks.
    8. Documents all limitations and TODOs for future real model integration.
All errors are unique and actionable. See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for systematic debugging.
"""
import os
import pathlib
import hydra
from omegaconf import DictConfig
import torch
import sys

EXPECTED_CWD = "/Users/tomriddle1/RNA_PREDICT"
CONFIG_ABS_PATH = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf/default.yaml"

# 1. Assert CWD is project root for robust, actionable error reporting
actual_cwd = os.getcwd()
if actual_cwd != EXPECTED_CWD:
    print(
        f"[UNIQUE-ERR-HYDRA-CWD] Script must be run from the project root directory.\n"
        f"Expected CWD: {EXPECTED_CWD}\n"
        f"Actual CWD:   {actual_cwd}\n"
        f"To fix: cd {EXPECTED_CWD} && uv run partial_checkpoint_full_pipeline_script.py\n"
        f"See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for more info."
    )
    sys.exit(1)

# 2. Pre-test: Fail early if config is not accessible (strict absolute path)
if not os.path.exists(CONFIG_ABS_PATH):
    print(f"[UNIQUE-ERR-HYDRA-CONF-NOT-FOUND] {CONFIG_ABS_PATH} not found. Run this script from the project root and ensure config is present at absolute path. See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md")
    sys.exit(1)

# 3. Instrument with debug output and dynamic config_path selection
cwd = pathlib.Path(os.getcwd())
config_candidates = [cwd / "rna_predict" / "conf", cwd / "conf"]
config_path_selected = None
for candidate in config_candidates:
    print(f"[SCRIPT DEBUG] Checking for config directory: {candidate}")
    if candidate.exists() and (candidate / "default.yaml").exists():
        config_path_selected = str(candidate.relative_to(cwd))
        print(f"[SCRIPT DEBUG] Found config at: {candidate}, using config_path: {config_path_selected}")
        # Print directory contents and permissions for further debugging
        print(f"[SCRIPT DEBUG] Contents of {candidate}:")
        for item in candidate.iterdir():
            print(f"  - {item} (exists: {item.exists()}, is_file: {item.is_file()}, perms: {oct(item.stat().st_mode)})")
        default_yaml = candidate / "default.yaml"
        if default_yaml.exists():
            print(f"[SCRIPT DEBUG] default.yaml permissions: {oct(default_yaml.stat().st_mode)}")
        else:
            print(f"[SCRIPT DEBUG] default.yaml not found in {candidate}")
        print(f"[SCRIPT DEBUG] Absolute path to config directory: {candidate.resolve()}")
        break
if not config_path_selected:
    print("[UNIQUE-ERR-HYDRA-CONF-PATH-NOT-FOUND] Neither 'rna_predict/conf' nor 'conf' found relative to current working directory.\nCWD: {}\nChecked: {}\nSee docs/guides/best_practices/debugging/comprehensive_debugging_guide.md".format(os.getcwd(), [str(c) for c in config_candidates]))
    sys.exit(1)

# 4. Load config and build pipeline
try:
    with hydra.initialize(config_path="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf", job_name="partial_checkpoint_full_pipeline_script"):
        cfg = hydra.compose(config_name="default")
    print("[SCRIPT DEBUG] Hydra loaded config successfully:")
    print(cfg)
except Exception as e:
    print(f"[UNIQUE-ERR-HYDRA-INIT] Exception during hydra.initialize: {e}")
    sys.exit(1)

# 5. Instantiate RNALightningModule (placeholder-aware)
try:
    from rna_predict.training.rna_lightning_module import RNALightningModule
    model = RNALightningModule(cfg)
    print("[SCRIPT DEBUG] RNALightningModule instantiated successfully.")
    if isinstance(model.pipeline, torch.nn.Identity):
        print("[UNIQUE-WARN-STUB] RNALightningModule.pipeline is a stub (Identity). No real model logic is present. All further assertions are stubbed.")
except Exception as e:
    print(f"[UNIQUE-ERR-LIGHTNING-INSTANTIATION] Exception during RNALightningModule instantiation: {e}")
    sys.exit(1)

# 6. Save a "partial" checkpoint (stub logic)
import tempfile
partial_ckpt_path = tempfile.mktemp(suffix="_partial.ckpt")
try:
    from rna_predict.utils.checkpointing import save_trainable_checkpoint
    save_trainable_checkpoint(model, partial_ckpt_path)
    print(f"[SCRIPT DEBUG] Partial checkpoint saved at {partial_ckpt_path}")
except Exception as e:
    print(f"[UNIQUE-ERR-CHECKPOINT-SAVE] Exception during save_trainable_checkpoint: {e}")
    sys.exit(1)

# 7. Load checkpoint using partial_load_state_dict (stub logic)
try:
    import torch
    checkpoint = torch.load(partial_ckpt_path)
    # Try to import and use partial_load_state_dict utility
    try:
        from rna_predict.utils.checkpoint import partial_load_state_dict
        missing_keys, unexpected_keys = partial_load_state_dict(model, checkpoint, strict=False)
        if missing_keys or unexpected_keys:
            print(f"[UNIQUE-WARN-STUB] partial_load_state_dict: missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")
        else:
            print(f"[SCRIPT DEBUG] partial_load_state_dict loaded with no key/shape errors.")
    except ImportError:
        print("[UNIQUE-ERR-NOT-IMPLEMENTED] partial_load_state_dict utility not found. Skipping checkpoint load test.")
except Exception as e:
    print(f"[UNIQUE-ERR-CHECKPOINT-LOAD] Exception during checkpoint load: {e}")
    sys.exit(1)

# 8. Run a forward pass (stub logic)
try:
    dummy_input = torch.zeros(4, 16)  # matches dummy dataloader in RNALightningModule
    output = model(dummy_input)
    if isinstance(model.pipeline, torch.nn.Identity):
        if output.shape != dummy_input.shape:
            print(f"[UNIQUE-ERR-STUB-FORWARD] Identity forward pass returned wrong shape: {output.shape}")
        else:
            print(f"[SCRIPT DEBUG] Forward pass (stub) succeeded with output shape {output.shape}")
    else:
        print(f"[UNIQUE-WARN-STUB] Non-stub pipeline detected. Add real output shape/type assertions here.")
except Exception as e:
    print(f"[UNIQUE-ERR-FORWARD] Exception during forward pass: {e}")
    sys.exit(1)

# 9. Compare checkpoint sizes (stub logic)
try:
    import os
    # Save a full checkpoint for comparison
    full_ckpt_path = tempfile.mktemp(suffix="_full.ckpt")
    torch.save(model.state_dict(), full_ckpt_path)
    partial_size = os.path.getsize(partial_ckpt_path)
    full_size = os.path.getsize(full_ckpt_path)
    if partial_size >= full_size:
        print(f"[UNIQUE-WARN-STUB] Partial checkpoint is not smaller than full checkpoint (stub logic). Sizes: partial={partial_size}, full={full_size}")
    else:
        print(f"[SCRIPT DEBUG] Partial checkpoint size: {partial_size}, full checkpoint size: {full_size}")
except Exception as e:
    print(f"[UNIQUE-ERR-CHECKPOINT-SIZE] Exception during checkpoint size comparison: {e}")
    sys.exit(1)

print("[SCRIPT SUCCESS] Partial checkpoint full pipeline script completed successfully (stub logic). See unique warnings for stubbed assertions.")
