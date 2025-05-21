# %%
# Cell : clean auto-generated requirements file  (run FIRST!)
# -----------------------------------------------------------
import os
import sys
import pathlib
import itertools
import textwrap
import numpy as np
import pandas as pd
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
else:
    logging.warning("Test CSVs missing – adjust paths or upload files.")



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

# 
# Cell : sanity-check submission.csv against test_sequences.csv  ✅
# ----------------------------------------------------------------

import pandas as pd
import pathlib
import textwrap
import sys
TEST_CSV = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
SUB_CSV  = "submission.csv"
TOL      = 1.0  # Å – treat coords within ±1 Å as identical

# ── 0)  helpers ─────────────────────────────────────────────────────────
def auto_col(df, pref):
    for c in pref:
        if c in df.columns:
            return c
    return df.columns[0]

def preview(s, n=5):
    lst = list(s)
    return ", ".join(lst[:n]) + (" …" if len(lst) > n else "")

# ── 1)  load / basic info ───────────────────────────────────────────────
for f in (TEST_CSV, SUB_CSV):
    if not pathlib.Path(f).is_file():
        sys.exit(f"[ERROR] {f} not found!")

test_sequences = pd.read_csv(TEST_CSV)
submission     = pd.read_csv(SUB_CSV)

id_col_test = auto_col(test_sequences, ["ID", "id", "seq_id", "sequence_id"])
id_col_sub  = auto_col(submission,     ["ID", "id", "seq_id", "sequence_id"])

# ── 2)  expected vs actual rows ─────────────────────────────────────────
expected_rows = test_sequences["sequence"].str.len().sum()
print("\n━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Expected rows        : {expected_rows:,}")
print(f"submission.csv rows  : {len(submission):,}")
dupes = submission[id_col_sub].duplicated().sum()
print(f"Duplicate {id_col_sub!r} rows : {dupes:,}")

# ── 3)  build the *full* ID set   "<sequenceID>_<resIdx>"  ─────────────
full_id_set = {
    f"{sid}_{idx}"
    for sid, seq in zip(test_sequences[id_col_test], test_sequences["sequence"])
    for idx in range(1, len(seq) + 1)
}
sub_id_set = set(submission[id_col_sub].astype(str))

missing = full_id_set - sub_id_set
extra   = sub_id_set  - full_id_set

print("\n━━ ID reconciliation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"IDs missing from submission : {len(missing):,}")
print(f"Unexpected extra IDs        : {len(extra):,}")
if missing: print("  → first few missing :", preview(missing))
if extra:   print("  → first few extras  :", preview(extra))

# ── 4)  per-sequence coverage (how many residues per sequence?) ────────
seq_len = test_sequences.set_index(id_col_test)["sequence"].str.len()

# **FIXED LINE BELOW** – use expand=True to ensure a 1-D Series (avoids ndarray shape (n, 3))
prefixes = (
    submission[id_col_sub]
    .astype(str)
    .str.rsplit("_", n=1, expand=True)[0]   # returns a Series, not a nested ndarray
)

coverage = prefixes.value_counts().reindex(seq_len.index).fillna(0).astype(int)
bad_cov  = coverage[coverage != seq_len]

print("\n━━ Per-sequence coverage ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Sequences with wrong #rows : {len(bad_cov):,}")
if len(bad_cov):
    print("  id  | expected | got")
    for sid, got in itertools.islice(bad_cov.items(), 5):
        print(f" {sid:<6}| {seq_len[sid]:>8} | {got}")

# ── 5)  column sanity ───────────────────────────────────────────────────
REQ_COLS = ["ID", "resname", "resid"] + [f"{ax}_{i}" for i in range(1, 6) for ax in "xyz"]
missing_cols = [c for c in REQ_COLS if c not in submission.columns]

print("\n━━ Column sanity ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Missing required columns   : {len(missing_cols)}")
if missing_cols:
    print(textwrap.fill(", ".join(missing_cols), width=88))

# ── 6)  structure-repeat uniqueness ────────────────────────────────────
trip_cols = np.array([[f"{ax}_{i}" for ax in "xyz"] for i in range(1, 6)])
coords = submission[trip_cols.flatten()].values.reshape(len(submission), 5, 3)

def unique_triplet_count(row):
    """Return #unique (x,y,z) triplets in a 5×3 slice."""
    uniq = []
    for v in row:
        if not any(np.allclose(v, u, atol=TOL) for u in uniq):
            uniq.append(v)
    return len(uniq)

# 👉 replace apply_along_axis with a 1-liner list-comprehension  ✅
uniq_counts = np.array([unique_triplet_count(row) for row in coords])

all_identical = (uniq_counts == 1).sum()
truly_unique  = (uniq_counts > 1).sum()

print("\n━━ Structure-repeat uniqueness ━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Rows where 5 structures are identical : {all_identical:,}")
print(f"Rows with ≥2 distinct triplets         : {truly_unique:,}")

# Per-sequence share of unique repeats
sub_seq_id = prefixes.to_numpy()   # 1-D array of sequence IDs
per_seq_unique = (
    pd.Series(uniq_counts > 1, index=sub_seq_id)
      .groupby(level=0).mean()
      .sort_values(ascending=False)
)

print("\nTop 5 sequences with most unique repeats:")
for sid, frac in per_seq_unique.head(5).items():
    print(f"  {sid:<6}: {frac:6.1%} rows diversified")

print("\n✅  Sanity check finished.")
