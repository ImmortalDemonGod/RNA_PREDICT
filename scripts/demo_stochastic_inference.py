"""
Demo script: Stochastic Inference for Unique RNA Structure Predictions
Runs RNAPredictor.predict_submission on a sample sequence with stochastic inference enabled.
Prints the DataFrame and checks that all 5 repeats are unique.
"""
import os
import sys
import torch
import pandas as pd

# --- Ensure script is run from scripts/ directory for Hydra config loading ---
CWD = os.getcwd()
if not os.path.basename(CWD) == 'scripts':
    print("[ERROR] Please run this script from the scripts/ directory (where this file lives).\nCurrent working directory:", CWD)
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rna_predict.predict import RNAPredictor

# --- Hydra config loading with inheritance ---
cfg = None
try:
    import hydra
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    # NOTE: This script must be run from the scripts/ directory (where this file lives)
    hydra_conf_dir = "../rna_predict/conf"
    with initialize(version_base=None, config_path=hydra_conf_dir):
        cfg = compose(config_name="predict.yaml")
except ImportError:
    print("[ERROR] hydra-core is not installed. Please install hydra-core to run this script.")
    sys.exit(1)

if cfg is None or not hasattr(cfg, 'device'):
    print("[ERROR] Config loading failed or device key missing. Check your config files and Hydra setup.")
    sys.exit(1)

# --- Debug logging: print full config and prediction section ---
try:
    from omegaconf import OmegaConf
    print("\n[DEBUG] Full loaded config:\n" + OmegaConf.to_yaml(cfg))
    if hasattr(cfg, 'prediction'):
        print("[DEBUG] Prediction config section:\n" + OmegaConf.to_yaml(cfg.prediction))
    else:
        print("[DEBUG] No 'prediction' section in config!")
except Exception as e:
    print(f"[DEBUG] Could not print config: {e}")

# Instantiate predictor
predictor = RNAPredictor(cfg)

# Use a short, plausible RNA sequence for demo
sequence = "AUGCUAGCUAGCUA"

print("[INFO] Running stochastic inference for sequence:", sequence)
df = predictor.predict_submission(sequence)

print("\n[INFO] Prediction DataFrame:")
print(df.head())

# Check uniqueness of all repeats (x_1, x_2, ..., x_5)
coords_cols = [f"x_{i+1}" for i in range(5)] + [f"y_{i+1}" for i in range(5)] + [f"z_{i+1}" for i in range(5)]
unique_structs = set()
for i in range(5):
    coords = tuple(df[[f"x_{i+1}", f"y_{i+1}", f"z_{i+1}"]].to_numpy().flatten())
    unique_structs.add(coords)

if len(unique_structs) == 5:
    print("\n[SUCCESS] All 5 predicted structures are unique!")
else:
    print(f"\n[WARNING] Only {len(unique_structs)} unique structures found. Check stochastic inference setup.")
    for i, coords in enumerate(unique_structs):
        print(f"Structure {i+1}: {coords[:6]} ... (truncated)")

print("\n[INFO] Done.")
