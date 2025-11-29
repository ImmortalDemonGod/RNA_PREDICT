import torch
from rna_predict.training.rna_lightning_module import RNALightningModule
import hydra
from omegaconf import OmegaConf
import sys
import os

if __name__ == "__main__":
    print("[DEBUG] Script started.")
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/print_lightning_state_dict_keys.py /absolute/path/to/conf")
        sys.exit(1)
    config_path = sys.argv[1]
    print(f"[DEBUG] config_path argument: {config_path}")
    if not os.path.isdir(config_path):
        print(f"[ERROR] Config path {config_path} is not a directory!")
        sys.exit(2)
    print(f"[DEBUG] Contents of config directory {config_path}:")
    for fname in os.listdir(config_path):
        print(" -", fname)
    # Always resolve the config path relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(config_path):
        abs_config_path = config_path
    else:
        abs_config_path = os.path.abspath(os.path.join(script_dir, config_path))
    rel_config_path = os.path.relpath(abs_config_path, script_dir)
    print(f"[DEBUG] Input config_path: {config_path}")
    print(f"[DEBUG] Absolute config_path: {abs_config_path}")
    print(f"[DEBUG] Script directory: {script_dir}")
    print(f"[DEBUG] Using relative config_path for hydra.initialize: {rel_config_path}")
    try:
        with hydra.initialize(config_path=rel_config_path, version_base=None):
            print("[DEBUG] hydra.initialize successful")
            cfg = hydra.compose(config_name="default")
            print("[DEBUG] hydra.compose successful")
            print("[DEBUG] Top-level config keys:", list(cfg.keys()))
            if 'model' in cfg:
                print("[DEBUG] Top-level model keys:", list(cfg.model.keys()))
            else:
                print("[ERROR] No 'model' key in composed config!")
            model = RNALightningModule(cfg)
            print("[DEBUG] RNALightningModule instantiated")
            state_dict = model.state_dict()
            print("Top-level state_dict keys (first 30):")
            for i, k in enumerate(state_dict.keys()):
                print(f"{i}: {k}")
                if i >= 29:
                    break
            print(f"Total keys: {len(state_dict)}")
            print("\nAll top-level module attributes:")
            for attr in dir(model):
                if not attr.startswith('_'):
                    print(attr)
            print("\n[STAGE B/TORSION/BERT] Matching state_dict keys:")
            found = False
            for k in state_dict.keys():
                if any(s in k.lower() for s in ("stageb", "torsion", "bert")):
                    print(k)
                    found = True
            if not found:
                print("[NONE FOUND]")

            print("\n[MODEL.] Matching state_dict keys:")
            found_model = False
            for k in state_dict.keys():
                if "model." in k.lower():
                    print(k)
                    found_model = True
            if not found_model:
                print("[NONE FOUND]")
                print("\n[SAMPLE OF FIRST 100 STATE_DICT KEYS]:")
                for i, k in enumerate(state_dict.keys()):
                    print(f"{i}: {k}")
                    if i >= 99:
                        break

            # Print first 20 unique prefixes before the first dot
            print("\n[UNIQUE PREFIXES OF STATE_DICT KEYS]:")
            prefixes = []
            seen = set()
            for k in state_dict.keys():
                prefix = k.split('.', 1)[0]
                if prefix not in seen:
                    prefixes.append(prefix)
                    seen.add(prefix)
                if len(prefixes) >= 20:
                    break
            for i, prefix in enumerate(prefixes):
                print(f"{i}: {prefix}")

    except Exception as e:
        print("[EXCEPTION]", type(e).__name__, str(e))
        import traceback
        traceback.print_exc()
