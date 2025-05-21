# %%
# Cell : clean auto-generated requirements file  (run FIRST!)
# -----------------------------------------------------------
import os
import sys
import pathlib
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from transformers import *
# from omegaconf import OmegaConf # Removed: Unused
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from rna_predict.kaggle.kaggle_env import (
    setup_kaggle_environment,
    print_kaggle_input_tree,
    print_system_info,
    set_offline_env_vars,
    symlink_torsionbert_checkpoint,
    symlink_dnabert_checkpoint,
    patch_transformers_for_local
)
from rna_predict.kaggle.data_utils import load_kaggle_data, collapse_to_one_row_per_residue, process_test_sequences
from rna_predict.kaggle.submission_validator import run_sanity_checks

setup_kaggle_environment()


# (Requirements cleaning is now handled by setup_kaggle_environment())

# %%
# Cell: show what’s inside every mounted Kaggle dataset  🔍 (Python version)
# --------------------------------------------------------


print_kaggle_input_tree()

# %%
# ---
# NOTE: Wheel installation is now handled by setup_kaggle_environment() in Python.
# This bash block is retained for manual/fallback use or inspection, but is not required for normal operation.
# ---


# %%
# ---
# Cell: ALL-IN-ONE Environment Setup  (no uninstalls, no online pip)
# ---



# Run all Kaggle/offline environment setup (includes wheels, symlinks, offline vars, etc.)
setup_kaggle_environment()

# Print system diagnostics (Python, OS, CPU, memory, disk)
print_system_info()

# TODO: All hardcoded paths and version strings below should be moved to config for Hydra integration.
# (Keep marking with # TODO as you modularize further.)

# %%
# -*- coding: utf-8 -*-
"""
Cell 1: ENVIRONMENT SETUP & LOGGING
-----------------------------------
"""

# Machine Learning Libraries


# TODO: XGBoost import removed in cleanup pass 1
# from xgboost import XGBRegressor


# =======================
# Imports (Standard Library)
# =======================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.info("Cell 1 complete: Libraries imported and logging initialized.")


# Call the data loader at the notebook's data import step
(
    train_sequences,
    train_labels,
    validation_sequences,
    validation_labels,
    test_sequences,
    sample_submission,
) = load_kaggle_data()

logging.info("Cell 2 complete: Data loaded and assigned.")



# Set up HuggingFace offline environment and symlink checkpoints before anything else
set_offline_env_vars()
symlink_torsionbert_checkpoint()
symlink_dnabert_checkpoint()
patch_transformers_for_local()

# Cell: RNA Prediction with TorsionBERT  (offline-ready)
# ------------------------------------------------------


# ╔══════════════════════════════════════════════════════════════════════╗
# 3) LOGGING & tiny shell helper
# ╚══════════════════════════════════════════════════════════════════════╝
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

# %%
# ╔══════════════════════════════════════════════════════════════════════╗
# 5) RNAPredictor CONFIG (Hydra best practices, stochastic inference)
# ╚══════════════════════════════════════════════════════════════════════╝

# Import OmegaConf and torch if not already imported in the cell


 # Ensure logging is imported if you use logger.info

TEST_SEQS  = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
SAMPLE_SUB = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
OUTPUT_CSV = "submission.csv"

# Import create_predictor from the new config module
from rna_predict.kaggle.predictor_config import create_predictor

# Usage example:
predictor = create_predictor()

# %%
# ────────────────────────────────────────────────────────────────────────────
# 6) PREDICTION UTILITIES  ● de-duplication / aggregation safeguard  ✅
# -----------------------------------------------------------------------
# NOTE: This cell REPLACES the previous buggy version.
# Fix: drop existing "ID" column before inserting the new one.
# -----------------------------------------------------------------------



# %%
# ╔══════════════════════════════════════════════════════════════════════╗
# 7) TOY SANITY-CHECK – demonstrates collapse function
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n=== Toy sanity-check ===")
toy_raw  = create_predictor().predict_submission("ACGUACGU", prediction_repeats=2)
toy_comp = collapse_to_one_row_per_residue(toy_raw, "TOY")
print(toy_comp.head())

# ╔══════════════════════════════════════════════════════════════════════╗
# 8) FULL TEST SET  (comment out to iterate faster during dev)
# ╚══════════════════════════════════════════════════════════════════════╝
if os.path.exists(TEST_SEQS) and os.path.exists(SAMPLE_SUB):
    process_test_sequences(TEST_SEQS, SAMPLE_SUB, OUTPUT_CSV, batch=1)
    # Call the new sanity checker
    if pathlib.Path(OUTPUT_CSV).exists(): # Ensure submission file was created
        run_sanity_checks(TEST_SEQS, OUTPUT_CSV)
    else:
        logging.error(f"Submission file {OUTPUT_CSV} not found after processing. Skipping sanity checks.")
else:
    logging.warning("Test CSVs missing – adjust paths or upload files. Skipping processing and sanity checks.")



# %%
# Cell 12: CONCLUSIONS & NEXT STEPS
# ---------------------------------
'''
We've done:
- Group-based imputation
- Preserved resname
- Hyperparameter tuning via RandomizedSearchCV
- Final training on full combined data
- Test predictions with the same coordinate repeated across 5 structures

Suggestions for further improvement:
- Fine-tune hyperparameters with a broader search or Bayesian optimization
- Explore more advanced RNA 3D features
- Generate truly distinct 5 structures instead of repeating the same coordinates
'''
logging.info("Notebook complete. Good luck on the leaderboard!")
print("All done! Submit 'submission.csv' to the competition.")

# 

import pathlib
import sys
import os
print("\n📂  Listing the first two levels of /kaggle/working …\n")
working_root = pathlib.Path("/kaggle/working")
if working_root.exists():
    for item in sorted(working_root.iterdir()):
        print(f"  {item}")
        if item.is_dir():
            for sub in sorted(item.iterdir()):
                print(f"    {sub}")
print("\n✅  Done.\n")
