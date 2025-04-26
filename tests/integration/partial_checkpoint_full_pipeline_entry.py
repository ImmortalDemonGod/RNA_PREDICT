import os
import pathlib
import sys
import hydra
from omegaconf import DictConfig

# Define project root using file location (portable)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

@hydra.main(config_path="../../rna_predict/conf", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(f"[HYDRA MAIN DEBUG] Current working directory: {os.getcwd()}")
    print(f"[HYDRA MAIN DEBUG] Config keys: {list(cfg.keys())}")
    # Add your test logic here, e.g. instantiate model, run checks, etc.
    # For now, just print the config as proof of success
    print("[HYDRA MAIN DEBUG] Loaded config:")
    print(cfg)

if __name__ == "__main__":
    # Assert CWD is project root for robust, actionable error reporting
    if pathlib.Path.cwd() != PROJECT_ROOT:
        sys.exit("[UNIQUE-ERR-CWD] Run from project root")

    # Pre-test: Fail early if config is not accessible (using project root)
    config_path = PROJECT_ROOT / "rna_predict" / "conf" / "default.yaml"
    if not config_path.exists():
        print(f"[UNIQUE-ERR-HYDRA-CONF-NOT-FOUND] {config_path} not found. Run this script from the project root and ensure config is present.")
        sys.exit(1)

    main()
