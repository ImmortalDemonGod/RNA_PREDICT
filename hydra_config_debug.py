import os
import pathlib
import hydra

print(f"[DEBUG] Current working directory: {os.getcwd()}")
cwd = pathlib.Path(os.getcwd())
config_candidates = [cwd / "rna_predict" / "conf", cwd / "conf"]
config_path_selected = None
for candidate in config_candidates:
    print(f"[DEBUG] Checking for config directory: {candidate}")
    if candidate.exists() and (candidate / "default.yaml").exists():
        config_path_selected = str(candidate.relative_to(cwd))
        print(f"[DEBUG] Found config at: {candidate}, using config_path: {config_path_selected}")
        print(f"[DEBUG] Contents of {candidate}:")
        for item in candidate.iterdir():
            print(f"  - {item} (exists: {item.exists()}, is_file: {item.is_file()}, perms: {oct(item.stat().st_mode)})")
        default_yaml = candidate / "default.yaml"
        if default_yaml.exists():
            print(f"[DEBUG] default.yaml permissions: {oct(default_yaml.stat().st_mode)}")
        else:
            print(f"[DEBUG] default.yaml not found in {candidate}")
        print(f"[DEBUG] Absolute path to config directory: {candidate.resolve()}")
        break
if not config_path_selected:
    raise RuntimeError(f"[UNIQUE-ERR-HYDRA-CONF-PATH-NOT-FOUND] Neither 'rna_predict/conf' nor 'conf' found relative to current working directory.\nCWD: {os.getcwd()}\nChecked: {[str(c) for c in config_candidates]}")

try:
    with hydra.initialize(config_path=config_path_selected, job_name="hydra_config_debug"):
        cfg = hydra.compose(config_name="default")
    print("[DEBUG] Hydra loaded config successfully:")
    print(cfg)
except Exception as e:
    print(f"[UNIQUE-ERR-HYDRA-EXTERNAL-DEBUG] Exception during hydra.initialize: {e}")
    raise
