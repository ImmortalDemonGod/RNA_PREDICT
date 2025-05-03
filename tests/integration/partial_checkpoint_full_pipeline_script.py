"""
Integration script for partial checkpointing and full pipeline construction in RNA_PREDICT.

- This script will:
    1. Assert correct CWD and config presence.
    2. Dynamically select the config path for Hydra.
    3. Load the config and build the pipeline using the config.
    4. (Optionally) Run a dummy forward pass or checkpoint logic.
- All errors are unique and actionable. See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for systematic debugging.
"""
import pathlib
import sys
try:
    import hydra
except ImportError:
    print("[UNIQUE-ERR-HYDRA-NOT-INSTALLED] hydra-core is required but not installed. Please check your network/certificate settings and ensure pip can fetch packages. See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for troubleshooting.")
    sys.exit(1)

# 1. Define project root using file location (portable)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
print(f"[SCRIPT DEBUG] Using PROJECT_ROOT: {PROJECT_ROOT}")

# 2. Assert CWD is project root for robust, actionable error reporting
if pathlib.Path.cwd() != PROJECT_ROOT:
    sys.exit("[UNIQUE-ERR-CWD] Run from project root")

# 3. Pre-test: Fail early if config is not accessible (using project root)
config_path = PROJECT_ROOT / "rna_predict" / "conf" / "default.yaml"
if not config_path.exists():
    print(f"[UNIQUE-ERR-HYDRA-CONF-NOT-FOUND] {config_path} not found. Run this script from the project root and ensure config is present. See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md")
    sys.exit(1)

# 4. Instrument with debug output and dynamic config_path selection
config_candidates = [PROJECT_ROOT / "rna_predict" / "conf", PROJECT_ROOT / "conf"]
config_path_selected = None
for candidate in config_candidates:
    print(f"[SCRIPT DEBUG] Checking for config directory: {candidate}")
    if candidate.exists() and (candidate / "default.yaml").exists():
        config_path_selected = str(candidate.relative_to(PROJECT_ROOT))
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
    print("[UNIQUE-ERR-HYDRA-CONF-PATH-NOT-FOUND] Neither 'rna_predict/conf' nor 'conf' found relative to project root.\nProject root: {}\nChecked: {}\nSee docs/guides/best_practices/debugging/comprehensive_debugging_guide.md".format(PROJECT_ROOT, [str(c) for c in config_candidates]))
    sys.exit(1)

# 5. Load config and build pipeline
try:
    # Use the config_path_selected we found earlier
    with hydra.initialize(config_path=config_path_selected, job_name="partial_checkpoint_full_pipeline_script", version_base=None):
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
