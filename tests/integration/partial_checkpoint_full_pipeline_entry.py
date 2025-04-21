import os
import pathlib
import hydra
from omegaconf import DictConfig
import pytest

EXPECTED_CWD = "/Users/tomriddle1/RNA_PREDICT"
CONFIG_ABS_PATH = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf/default.yaml"

@hydra.main(config_path="rna_predict/conf", config_name="default")
def main(cfg: DictConfig):
    print(f"[HYDRA MAIN DEBUG] Current working directory: {os.getcwd()}")
    print(f"[HYDRA MAIN DEBUG] Config keys: {list(cfg.keys())}")
    # Add your test logic here, e.g. instantiate model, run checks, etc.
    # For now, just print the config as proof of success
    print("[HYDRA MAIN DEBUG] Loaded config:")
    print(cfg)

if __name__ == "__main__":
    # Assert CWD is project root for robust, actionable error reporting
    actual_cwd = os.getcwd()
    if actual_cwd != EXPECTED_CWD:
        print(f"[UNIQUE-ERR-HYDRA-CWD] Test must be run from the project root directory.\n"
              f"Expected CWD: {EXPECTED_CWD}\n"
              f"Actual CWD:   {actual_cwd}\n"
              f"To fix: cd {EXPECTED_CWD} && uv run partial_checkpoint_full_pipeline_entry.py\n"
              f"See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md for more info.")
        exit(1)
    # Pre-test: Fail early if config is not accessible (strict absolute path)
    if not os.path.exists(CONFIG_ABS_PATH):
        print(f"[UNIQUE-ERR-HYDRA-CONF-NOT-FOUND] {CONFIG_ABS_PATH} not found. Run this test from the project root and ensure config is present at absolute path. See docs/guides/best_practices/debugging/comprehensive_debugging_guide.md")
        exit(1)
    main()
