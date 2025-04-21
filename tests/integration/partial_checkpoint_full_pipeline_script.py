"""
Integration script for partial checkpointing and full pipeline construction in RNA_PREDICT.

- Run this script from the project root: /Users/tomriddle1/RNA_PREDICT
- This script will:
    1. Assert correct CWD and config presence.
    2. Dynamically select the config path for Hydra.
    3. Load the config and build the pipeline using the config.
    4. (Optionally) Run a dummy forward pass or checkpoint logic.
- All errors are unique and actionable. See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for systematic debugging.
"""
import os
print(f"[SCRIPT DEBUG] os.getcwd() at script start: {os.getcwd()}")
import pathlib
import sys
try:
    import hydra
except ImportError:
    print("[UNIQUE-ERR-HYDRA-NOT-INSTALLED] hydra-core is required but not installed. Please check your network/certificate settings and ensure pip can fetch packages. See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for troubleshooting.")
    sys.exit(1)
from omegaconf import DictConfig
import torch

EXPECTED_CWD = "/Users/tomriddle1/RNA_PREDICT"
CONFIG_ABS_PATH = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf/default.yaml"

# 1. Assert CWD is project root for robust, actionable error reporting
actual_cwd = os.getcwd()
if actual_cwd != EXPECTED_CWD:
    print(
        f"[UNIQUE-ERR-HYDRA-CWD] Script must be run from the project root directory.\n"
        f"Expected CWD: {EXPECTED_CWD}\n"
        f"Actual CWD:   {actual_cwd}\n"
        f"To fix: cd {EXPECTED_CWD} && uv run tests/integration/partial_checkpoint_full_pipeline_script.py\n"
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
    # Hydra requires config_path to be relative to CWD
    ABS_CONFIG_PATH = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"
    if not os.path.isdir(ABS_CONFIG_PATH):
        print(f"[UNIQUE-ERR-HYDRA-ABS-CONF-NOT-FOUND] Absolute config directory '{ABS_CONFIG_PATH}' not found or not a directory.\nSee docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for troubleshooting.")
        sys.exit(1)
    if os.getcwd() != EXPECTED_CWD:
        print(f"[UNIQUE-ERR-HYDRA-CWD] Script must be run from the project root directory.\n"
              f"Expected CWD: {EXPECTED_CWD}\n"
              f"Actual CWD:   {os.getcwd()}\n"
              f"To fix: cd {EXPECTED_CWD} && uv run tests/integration/partial_checkpoint_full_pipeline_script.py\n"
              f"See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for more info.")
        sys.exit(1)
    with hydra.initialize(config_path="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf", job_name="partial_checkpoint_full_pipeline_script"):
        cfg = hydra.compose(config_name="default")
    print("[SCRIPT DEBUG] Hydra loaded config successfully:")
    print(cfg)
except Exception as e:
    print(f"[UNIQUE-ERR-HYDRA-INIT] Exception during hydra.initialize: {e}")
    sys.exit(1)

# 5. Build the pipeline and run a dummy forward pass if possible
try:
    from rna_predict.pipeline.build_pipeline import build_pipeline
    model = build_pipeline(cfg)
    print("[SCRIPT DEBUG] Model instantiated successfully.")
    # Optionally, add a dummy forward pass or checkpoint logic here
    # e.g., x = torch.randn(1, 8); model(x)
except Exception as e:
    print(f"[UNIQUE-ERR-PIPELINE-INSTANTIATION] Exception during model instantiation: {e}")
    sys.exit(1)

print("[SCRIPT SUCCESS] Partial checkpoint full pipeline script completed successfully.")
