# %%
# Cell : clean auto-generated requirements file  (run FIRST!)
# -----------------------------------------------------------
from rna_predict.kaggle.kaggle_env import setup_kaggle_environment

setup_kaggle_environment()

import pathlib, sys, os  # Only keep what's actually used below
# (Requirements cleaning is now handled by setup_kaggle_environment())

# %%
# Cell: show what’s inside every mounted Kaggle dataset  🔍 (Python version)
# --------------------------------------------------------
from rna_predict.kaggle.kaggle_env import print_kaggle_input_tree

print_kaggle_input_tree()

# %%
# ---
# NOTE: Wheel installation is now handled by setup_kaggle_environment() in Python.
# This bash block is retained for manual/fallback use or inspection, but is not required for normal operation.
# ---
# %%bash
# # Cell : offline installs that match the *current* wheel set (lean version)
# # -----------------------------------------------------------------------
# set -euo pipefail
#
# # ── let pip look inside EVERY sub-folder of /kaggle/input ───────────────
# WHEEL_ROOT="/kaggle/input"
# FIND_LINKS_ARGS=""
# for d in "$WHEEL_ROOT" "$WHEEL_ROOT"/*; do
#   FIND_LINKS_ARGS+=" --find-links $d"
# done
#
# p () {                 # quiet install; warn (don’t die) if something fails
#   # shellcheck disable=SC2086
#   pip install --no-index $FIND_LINKS_ARGS --quiet "$@" \
#   || echo "[WARN] install failed → skipped: $*"
# }
#
# # ────────────────────────────────────────────────────────────────────────
# # 1) Core scientific stack
# # ────────────────────────────────────────────────────────────────────────
# p numpy==1.24.3
# p pandas==2.2.3
# p scipy==1.10.1
# p tqdm==4.67.1
# p seaborn==0.12.2
# p biopython==1.85
# p torch               # pre-installed in the Kaggle image
#
# # ────────────────────────────────────────────────────────────────────────
# # 2)  ML / NLP stack
# # ────────────────────────────────────────────────────────────────────────
# p huggingface_hub==0.31.1      # needs hf-xet (you already uploaded)
# p transformers==4.51.3
# p pytorch_lightning==2.5.0.post0   # gives us Lightning-core features
#
# # ────────────────────────────────────────────────────────────────────────
# # 3)  Extra deps *rna_predict* really imports
# # ────────────────────────────────────────────────────────────────────────
# p lightning-utilities==0.11.2  # comes with PL wheel but list explicitly
# p datasets==3.6.0
# p einops==0.8.1
# p hypothesis==6.131.15
# p black==25.1.0                # needs pathspec 0.12.1 → you uploaded both
# p pathspec==0.12.1
# p isort==6.0.1
# p ruff==0.11.9
# p mss==10.0.0
# p mdanalysis==2.9.0
# p mmtf-python==1.1.3
# p GridDataFormats==1.0.2
# p mrcfile==1.5.4
# p lxml==5.4.0
# p dearpygui==2.0.0
# p py-cpuinfo==9.0.0
# p Pillow                        # pillow-11-2-1 wheel present
# p exit-codes==1.3.0             # small helper used by HF-Hub 0.31+
#
# # ────────────────────────────────────────────────────────────────────────
# # 4)  Config utilities
# # ────────────────────────────────────────────────────────────────────────
# p hydra-core==1.3.2
# p omegaconf==2.3.0
# p ml_collections==1.1.0         # required by Protenix
#
# # ────────────────────────────────────────────────────────────────────────
# # 5)  rna-predict itself  (no-deps so nothing reaches PyPI)
# # ────────────────────────────────────────────────────────────────────────
# pip install --no-index --no-deps --quiet \
#   /kaggle/input/rna-structure-predict/rna_predict-2.0.3-py3-none-any.whl
#
# # ────────────────────────────────────────────────────────────────────────
# # 6)  Protenix 0.4.6  (wheel, but ignore its heavy deps like RDKit)
# # ────────────────────────────────────────────────────────────────────────
# pip install --no-index --no-deps --quiet \
#   /kaggle/input/protenix-0-4-6/protenix-0.4.6-py3-none-any.whl \
#   || echo "[WARN] Protenix wheel install failed."
#
# # ────────────────────────────────────────────────────────────────────────
# # 7)  Runtime shim: make “import lightning” point to pytorch_lightning
# # ────────────────────────────────────────────────────────────────────────
# python - <<'PY'
# import sys, importlib, types
# try:
#     import pytorch_lightning as pl
#     sys.modules.setdefault("lightning", pl)
# except ImportError:
#     print("[WARN] pytorch_lightning missing – shim not created")
# PY
#
# echo "✅  Offline wheel install phase complete."

# %%
# ---
# Cell: ALL-IN-ONE Environment Setup  (no uninstalls, no online pip)
# ---

from rna_predict.kaggle.kaggle_env import setup_kaggle_environment, print_system_info

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
import os
import sys
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
# TODO: XGBoost import removed in cleanup pass 1
# from xgboost import XGBRegressor


# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.info("Cell 1 complete: Libraries imported and logging initialized.")



from rna_predict.kaggle.data_utils import load_kaggle_data

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

# %%
"""
Cell 3: COMBINE TRAIN + VALIDATION & BASIC EDA
----------------------------------------------
We concatenate the train and validation sets to maximize data. 
Then we do a quick EDA check on shapes, missingness, etc.
"""

# Combine sequences and labels
trainval_sequences = pd.concat([train_sequences, validation_sequences], ignore_index=True)
trainval_labels = pd.concat([train_labels, validation_labels], ignore_index=True)

logging.info(f"Combined train+validation sequences: {trainval_sequences.shape}, labels: {trainval_labels.shape}")

# Quick check for missing
logging.info("Missing in combined sequences:\n" + str(trainval_sequences.isnull().sum()))
logging.info("Missing in combined labels:\n" + str(trainval_labels.isnull().sum()))

# Example EDA: sequence length distribution
trainval_sequences['sequence_length'] = trainval_sequences['sequence'].str.len()

plt.figure(figsize=(10,4))
sns.boxplot(x=trainval_sequences['sequence_length'], color='skyblue')
plt.title("Boxplot of Sequence Length (Train + Validation)")
plt.xlabel("Sequence Length")
plt.show()

logging.info("Cell 3 complete: Basic EDA finished.")

# %%
"""
Cell 4: HANDLE MISSING COORDINATES & MERGE
------------------------------------------
We replace '-1e18' with np.nan, then merge sequences with labels on target_id.
"""

# Replace -1e18 with np.nan in the labels
for col in ['x_1','y_1','z_1']:
    trainval_labels[col] = trainval_labels[col].replace(-1e18, np.nan)

logging.info("Replaced -1e18 with NaN in trainval_labels for x_1, y_1, z_1.")

# Extract pdb_id, chain_id from ID
trainval_labels['pdb_id']   = trainval_labels['ID'].apply(lambda x: x.split('_')[0])
trainval_labels['chain_id'] = trainval_labels['ID'].apply(lambda x: x.split('_')[1])
trainval_labels['target_id'] = trainval_labels['pdb_id'] + "_" + trainval_labels['chain_id']

# Merge
train_data = pd.merge(trainval_labels, trainval_sequences, on='target_id', how='left')
logging.info(f"Merged train_data shape: {train_data.shape}")

# Quick check
logging.info(f"Missing in x_1: {train_data['x_1'].isnull().sum()}, "
             f"y_1: {train_data['y_1'].isnull().sum()}, "
             f"z_1: {train_data['z_1'].isnull().sum()}")

logging.info("Cell 4 complete: Merged train_data, ready for group-based imputation.")

# %%
"""
Cell 5: FEATURE ENGINEERING
---------------------------
Create numerical/categorical features from the 'sequence'.
We'll keep 'resname' from the labels as a valuable feature.
"""

def engineer_features(df):
    """
    Create numerical & (some) categorical features from raw RNA sequence data.
    """
    df = df.copy()
    # Sequence-based
    df['seq_length'] = df['sequence'].str.len()
    df['A_cnt'] = df['sequence'].str.count('A')
    df['C_cnt'] = df['sequence'].str.count('C')
    df['G_cnt'] = df['sequence'].str.count('G')
    df['U_cnt'] = df['sequence'].str.count('U')
    df['begin_seq'] = df['sequence'].str[0]
    df['end_seq']   = df['sequence'].str[-1]
    
    # Di-nucleotide counts (example set)
    for pair in ['AC','AG','AU','CA','CG','CU','GA','GC','GU','UA','UC','UG',
                 'AA','CC','GG','UU']:
        df[f'{pair}_cnt'] = df['sequence'].str.count(pair)

    return df

# Apply feature engineering
train_data = engineer_features(train_data)

logging.info("Feature engineering applied to merged train_data.")

# We'll show an example of newly added columns
example_cols = ['seq_length','A_cnt','C_cnt','G_cnt','U_cnt','begin_seq','end_seq','AC_cnt','AA_cnt']
logging.info(f"Columns after FE sample:\n{train_data[example_cols].head(3)}")

logging.info("Cell 5 complete: Feature engineering done.")

# %%
"""
Cell 6: GROUP-BASED IMPUTATION
------------------------------
We impute missing x_1, y_1, z_1 within each (target_id, resname) group.
Finally, if any NAs remain, we fill them with a global median or drop them.
"""

# Perform group-based fill for x_1, y_1, z_1
train_data[['x_1','y_1','z_1']] = (
    train_data
    .groupby(['target_id','resname'])[['x_1','y_1','z_1']]
    .apply(lambda grp: grp.fillna(grp.mean()))
    .reset_index(level=['target_id','resname'], drop=True)
)

# In case any remain after group-based mean fill (e.g. group is all NaN), do a global fill
num_cols = ['x_1','y_1','z_1']
global_imputer = SimpleImputer(strategy='median')
train_data[num_cols] = global_imputer.fit_transform(train_data[num_cols])

# If you'd prefer to drop any leftover NAs instead:
# train_data.dropna(subset=['x_1','y_1','z_1'], inplace=True)

logging.info("Group-based imputation + global median fallback complete.")

# Confirm missing values
logging.info(f"Remaining missing x_1: {train_data['x_1'].isna().sum()}, "
             f"y_1: {train_data['y_1'].isna().sum()}, z_1: {train_data['z_1'].isna().sum()}")

logging.info("Cell 6 complete: Group-based imputation finished.")

# %%
"""
Cell 7: PREPARE DATA FOR MODELING
---------------------------------
We'll define the columns we won't use, set up X and y for x_1, y_1, z_1, 
and one-hot encode any relevant categorical columns (including resname).
"""

# Unused columns
unused_cols = [
    'ID','pdb_id','chain_id','resid',
    'x_1','y_1','z_1',
    'sequence','description','temporal_cutoff','all_sequences',
    'target_id'  # key used for merges
]

# We'll keep resname, begin_seq, end_seq as features this time
feature_cols = [col for col in train_data.columns if col not in unused_cols]

# Make a copy
train_df = train_data.copy()

# Convert to categories
for cat_col in ['resname','begin_seq','end_seq']:
    if cat_col in feature_cols:
        train_df[cat_col] = train_df[cat_col].astype('category')

# One-hot encode
train_df = pd.get_dummies(train_df, columns=['resname','begin_seq','end_seq'], drop_first=True)

# Our final set of features
X_cols = [col for col in train_df.columns if col not in unused_cols]

X_full = train_df[X_cols]
y_x_full = train_df['x_1']
y_y_full = train_df['y_1']
y_z_full = train_df['z_1']

logging.info(f"Feature matrix shape: {X_full.shape}")
logging.info("Cell 7 complete: Prepared data for modeling.")

# %%
"""
Cell 8: KFold CV for X, Y, Z & Hyperparam Search
------------------------------------------------
We'll do a simplified KFold cross-validation for each coordinate 
to get a sense of good hyperparams, then train final models.
"""

from sklearn.model_selection import KFold, RandomizedSearchCV
import numpy as np

# Example hyperparameter grid (you can expand as needed)
# TODO: param_dist removed in cleanup pass 1

# TODO: run_random_search removed in cleanup pass 1


# TODO: This cell previously performed XGBoost hyperparam search (RandomizedSearchCV),
# but all related code has now been removed. Candidate for full removal or repurposing.

# %%
"""
Cell 9: FINAL TRAINING ON FULL DATA
-----------------------------------
Use the best hyperparams for each coordinate found in CV. 
Retrain each coordinate model on all data (X_full, y_*_full).
"""

# TODO: get_best_xgb removed in cleanup pass 1


logging.info("Retraining final model for X coordinate...")
#model_x = get_best_xgb(best_params_x)
#model_x.fit(X_full, y_x_full)

logging.info("Retraining final model for Y coordinate...")
#model_y = get_best_xgb(best_params_y)
#model_y.fit(X_full, y_y_full)

logging.info("Retraining final model for Z coordinate...")
#model_z = get_best_xgb(best_params_z)
#model_z.fit(X_full, y_z_full)

logging.info("Cell 9 complete: Final models trained.")

# %%
"""
Cell 10: PREPARE & ENGINEER TEST DATA
-------------------------------------
• Expand test_sequences into (ID, resname, resid)
• Merge residue‑level grid with per‑sequence engineered features
• Align with training feature matrix X_full, fill missing values
"""

# ---------- 1. Expand residue grid ----------
test_expanded = [
    [row["target_id"], nt, i]
    for _, row in test_sequences.iterrows()
    for i, nt in enumerate(row["sequence"], start=1)
]
test_clean_df = pd.DataFrame(test_expanded, columns=["ID", "resname", "resid"])
logging.info(f"test_clean_df shape: {test_clean_df.shape} (expanded test sequences)")

# ---------- 2. Per‑sequence engineered features ----------
test_feats = engineer_features(test_sequences)

# Merge – one row per residue, sequence‑level features broadcast to each residue
test_merged = pd.merge(
    test_clean_df,
    test_feats.drop(columns=["seq_length"]),   # drop if not needed
    left_on="ID",
    right_on="target_id",
    how="left"
)
logging.info(f"test_merged shape after merging: {test_merged.shape}")

# ---------- 3. Clean up ----------
# Replace sentinel values
for col in ["x_1", "y_1", "z_1"]:
    if col in test_merged.columns:
        test_merged[col] = test_merged[col].replace(-1e18, np.nan)

# Drop columns not used by the model
drop_cols = ["sequence", "description", "temporal_cutoff", "all_sequences", "target_id"]
test_merged.drop(columns=[c for c in drop_cols if c in test_merged.columns], inplace=True, errors="ignore")

# ---------- 4. Categorical handling ----------
cat_cols = {"resname", "begin_seq", "end_seq"} & set(test_merged.columns)
for col in cat_cols:
    test_merged[col] = test_merged[col].astype("category")
test_merged = pd.get_dummies(test_merged, columns=list(cat_cols), drop_first=True)

# ---------- 5. Column alignment ----------
# Single vectorised reindex instead of per‑column insertion → no fragmentation warning
test_merged = test_merged.reindex(columns=X_full.columns, fill_value=0)

# ---------- 6. Missing‑value imputation ----------
# Fit a NEW median imputer on the training feature matrix (numeric cols only)
numeric_cols = X_full.select_dtypes(include=np.number).columns
feature_imputer = SimpleImputer(strategy="median")
feature_imputer.fit(X_full[numeric_cols])

test_merged[numeric_cols] = feature_imputer.transform(test_merged[numeric_cols])

# ---------- 7. All done ----------
test_merged_imputed = test_merged.copy()
logging.info("Cell 10 complete: Test data prepared, aligned, and imputed.")

# %%
!HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/kaggle/working TRANSFORMERS_CACHE=/kaggle/working ln -sf /kaggle/input/rna-torsion-bert-checkpoint-base/kaggle/working/rna_torsionBERT /kaggle/working/rna_torsionBERT

# %%
# Cell: RNA Prediction with TorsionBERT  (offline-ready)
# ------------------------------------------------------
import pandas as pd, torch, os, logging, sys, transformers
from omegaconf import OmegaConf
from functools import partial

# ╔══════════════════════════════════════════════════════════════════════╗
# 1) LINK LOCAL CHECKPOINTS ▸ rna_torsionBERT  &  DNA_Bert_3
# ╚══════════════════════════════════════════════════════════════════════╝
if not os.path.exists("/kaggle/working/rna_torsionBERT"):
    os.symlink(
        "/kaggle/input/rna-torsion-bert-checkpoint-base/kaggle/working/rna_torsionBERT",
        "/kaggle/working/rna_torsionBERT"
    )

DNA_BERT_SRC = "/kaggle/input/dna-bert-rna/DNA_bert_3"
DNA_BERT_DST = "/kaggle/working/zhihan1996/DNA_bert_3"   # path hard-coded in torsionBERT
if not os.path.exists(DNA_BERT_DST):
    os.makedirs("/kaggle/working/zhihan1996", exist_ok=True)
    os.symlink(DNA_BERT_SRC, DNA_BERT_DST)

# ╔══════════════════════════════════════════════════════════════════════╗
# 2) FORCE OFFLINE MODE  +  MAP zhihan1996/* IDs → local folders
# ╚══════════════════════════════════════════════════════════════════════╝
os.environ.update({
    "HF_HUB_OFFLINE":      "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE":"1",
    "HF_HOME":             "/kaggle/working",
    "TRANSFORMERS_CACHE":  "/kaggle/working"
})
def _localize(repo,*a,**kw):
    """
    Redirect zhihan1996/DNA_bert_* to local paths and
    force local_files_only for every HF load.
    """
    if isinstance(repo,str) and repo.startswith("zhihan1996/DNA_bert_"):
        repo = "/kaggle/working/" + repo
    kw["local_files_only"] = True
    return repo,a,kw
# robust monkey-patch (handles partials, repeated patching, etc.)
for _cls in ("AutoConfig","AutoTokenizer","AutoModel"):
    obj      = getattr(transformers, _cls)
    base_cls = obj.func if isinstance(obj, partial) else obj
    if not hasattr(base_cls, "from_pretrained"):
        continue
    _orig = base_cls.from_pretrained
    def _wrap(repo,*a,__o=_orig,**kw):
        repo,a,kw = _localize(repo,*a,**kw)
        return __o(repo,*a,**kw)
    base_cls.from_pretrained = _wrap

# accept DNA-Bert-3 custom config
try:
    from importlib import import_module
    custom_conf = import_module(
        "transformers_modules.DNA_bert_3.configuration_bert"
    ).BertConfig
    transformers.models.bert.modeling_bert.BertModel.config_class = custom_conf
except Exception as e:
    logging.warning(f"[WARN] DNA_Bert_3 config patch skipped: {e}")

# ╔══════════════════════════════════════════════════════════════════════╗
# 3) LOGGING & tiny shell helper
# ╚══════════════════════════════════════════════════════════════════════╝
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
def run_and_print(cmd):
    import subprocess, shlex, textwrap
    res = subprocess.run(cmd if isinstance(cmd,list) else shlex.split(cmd),
                         capture_output=True, text=True)
    if res.stdout: print(res.stdout, end="")
    if res.stderr: print("STDERR:", textwrap.shorten(res.stderr,400), end="")
    return res

# ╔══════════════════════════════════════════════════════════════════════╗
# 4) ENSURE hydra-core (local wheel) – omegaconf already present
# ╚══════════════════════════════════════════════════════════════════════╝
run_and_print([
    "pip","install","--no-index","--no-deps","--force-reinstall",
    "/kaggle/input/hydra-core-132whl/hydra_core-1.3.2-py3-none-any.whl"
])



# %%
# ╔══════════════════════════════════════════════════════════════════════╗
# 5) RNAPredictor CONFIG (Hydra best practices, stochastic inference)
# ╚══════════════════════════════════════════════════════════════════════╝
from rna_predict.interface import RNAPredictor
# Import OmegaConf and torch if not already imported in the cell
from omegaconf import OmegaConf
import torch
import logging # Ensure logging is imported if you use logger.info

TEST_SEQS  = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
SAMPLE_SUB = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
OUTPUT_CSV = "submission.csv"

def create_predictor():
    """Instantiate RNAPredictor with local checkpoints & GPU/CPU autodetect, matching Hydra config structure."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}") # Assuming logging is configured
    cfg = OmegaConf.create({
        # Top-level keys consistent with a full Hydra config (e.g., default.yaml)
        "device": device,
        "seed": 42, # Good for reproducibility if used by models
        "atoms_per_residue": 44, # Standard value
        "extraction_backend": "dssr", # Or "mdanalysis" as needed

        "pipeline": { # General pipeline settings
            "verbose": True,
            "save_intermediates": True,
            # output_dir is usually set by Hydra's run directory or overridden
        },

        "prediction": { # Prediction-specific settings
            "repeats": 5,
            "residue_atom_choice": 0,
            "enable_stochastic_inference_for_submission": True, # CRITICAL: Ensures unique predictions
            # "submission_seeds": [42, 101, 2024, 7, 1991],  # Optional: for reproducible stochastic runs
        },

        "model": {
            # ── Stage B: torsion-angle prediction ──────────────────────────
            "stageB": {
                "torsion_bert": {
                    "model_name_or_path": "/kaggle/working/rna_torsionBERT", # Path to local TorsionBERT model
                    "device": device,
                    "angle_mode": "degrees",   # CHANGED: Set to "degrees" for consistency with StageC
                                               # This ensures StageBTorsionBertPredictor outputs angles in degrees.
                    "num_angles": 7,
                    "max_length": 512,
                    "checkpoint_path": None,   # Can be overridden if a specific checkpoint is needed
                    "debug_logging": True,     # Set to False if logs are too verbose
                    "init_from_scratch": False, # Assumes using pretrained TorsionBERT
                    "lora": {                  # LoRA config (currently disabled)
                        "enabled": False,
                        "r": 8,
                        "alpha": 16,
                        "dropout": 0.1,
                        "target_modules": ["query", "value"],
                    },
                }
                # Pairformer config would go here if used: "pairformer": { ... }
            },
            # ── Stage C: 3D reconstruction (MP-NeRF) ──────────────────────
            "stageC": {
                "enabled": True,
                "method": "mp_nerf",
                "do_ring_closure": False,       # Consistent with default.yaml; notebook log showed True, adjust if needed.
                "place_bases": True,
                "sugar_pucker": "C3'-endo",
                "device": device,
                "debug_logging": True,          # Set to False if logs are too verbose
                "angle_representation": "degrees", # StageC expects angles in degrees from StageB
                "use_metadata": False,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,          # Consistent with default.yaml; notebook log showed True.
                "chunk_size": None,
            },
            # ── Stage D: Diffusion refinement (minimal placeholder) ────────
            # Add full StageD config if it's actively used in this notebook
            "stageD": {
                "enabled": False, # Set to True if StageD is part of this specific notebook's pipeline
                "mode": "inference",
                "device": device,
                "debug_logging": True,
                # Placeholder for other essential StageD keys if enabled:
                # "ref_element_size": 128,
                # "ref_atom_name_chars_size": 256,
                # "profile_size": 32,
                # "model_architecture": { ... },
                # "diffusion": { ... }
            },
        }
    })
    return RNAPredictor(cfg)

# Usage example:
predictor = create_predictor()

# %%
# ────────────────────────────────────────────────────────────────────────────
# PATCH ▸ guarantee predict_submission returns ONE ROW per residue  ✅
#         • works both when Stage C gives [L, atoms, 3]  OR  [N_atoms, 3]
#         • keeps all original columns created by coords_to_df
# ────────────────────────────────────────────────────────────────────────────
import logging, torch, pandas as pd
from rna_predict.interface import RNAPredictor
from rna_predict.utils.submission import coords_to_df, extract_atom, reshape_coords

log = logging.getLogger("rna_predict.patch.flat2res")

def _predict_submission_patched(
    self,
    sequence: str,
    prediction_repeats: int | None = None,
    residue_atom_choice: int | None = None,
):
    """
    Collapses per-atom coordinates → one canonical atom per residue.
    • Prefers phosphate (“P”); falls back to first atom per residue.
    • Always returns exactly len(sequence) rows, preserving coords_to_df schema.
    """
    # ── original prologue ────────────────────────────────────────────────
    result      = self.predict_3d_structure(sequence)
    coords_flat = result["coords"]                       # 2-D [N_atoms, 3]

    # 🔧 NEW: make it a plain tensor so .numpy() is allowed
    if coords_flat.requires_grad:                        # ← the bug-fix
        coords_flat = coords_flat.detach()

    metadata        = result.get("atom_metadata", {})
    atom_names      = metadata.get("atom_names", [])
    residue_indices = metadata.get("residue_indices", [])

    repeats  = prediction_repeats if prediction_repeats is not None else self.default_repeats
    atom_idx = residue_atom_choice if residue_atom_choice is not None else self.default_atom_choice

    # ────────────────────────────────────────────────────────────────────
    # ❶  FLAT-COORDS PATH   (Stage C returned [N_atoms, 3])
    # ────────────────────────────────────────────────────────────────────
    if coords_flat.dim() == 2 and coords_flat.shape[0] != len(sequence):
        if not atom_names or not residue_indices:
            log.error("[flat-coords] missing atom metadata → falling back to legacy per-atom output")
            base = {
                "ID":      range(1, len(coords_flat) + 1),
                "resname": ["X"] * len(coords_flat),
                "resid":   range(1, len(coords_flat) + 1),
            }
            df = pd.DataFrame(base)
            for i in range(1, repeats + 1):
                df[[f"{ax}_{i}" for ax in "xyz"]] = coords_flat.cpu().numpy()
            return df

        tmp = pd.DataFrame({
            "atom_name": atom_names,
            "res0":      residue_indices,       # 0-based residue index
            "x": coords_flat[:, 0].cpu().numpy(),
            "y": coords_flat[:, 1].cpu().numpy(),
            "z": coords_flat[:, 2].cpu().numpy(),
        })

        # pick one atom per residue (prefer P)
        picked = (tmp[tmp.atom_name == "P"]
                  .drop_duplicates("res0", keep="first")
                  .sort_values("res0"))
        if len(picked) != len(sequence):        # fallback if some P’s missing
            log.warning("[flat-coords] P-selection gave %d/%d rows – using first atom fallback",
                        len(picked), len(sequence))
            picked = (tmp.groupby("res0", as_index=False)
                         .first()
                         .sort_values("res0"))

        per_res_coords = torch.tensor(
            picked[["x", "y", "z"]].values,
            dtype=coords_flat.dtype,
            device=coords_flat.device,
        )

        return coords_to_df(sequence, per_res_coords, repeats)

    # ────────────────────────────────────────────────────────────────────
    # ❷  ORIGINAL “reshaped” PATH  (Stage C returned [L, atoms, 3])
    # ────────────────────────────────────────────────────────────────────
    coords = reshape_coords(coords_flat, len(sequence))
    if coords.dim() == 2 and coords.shape[0] != len(sequence):
        # reshape failed → treat as flat once more
        log.warning("[reshape_coords] produced flat coords – rerouting through flat-coords logic.")
        result["coords"] = coords
        return _predict_submission_patched(self, sequence, prediction_repeats, residue_atom_choice)

    atom_coords = extract_atom(coords, atom_idx)
    return coords_to_df(sequence, atom_coords, repeats)

# install the patch (simple attribute assignment is enough)
RNAPredictor.predict_submission = _predict_submission_patched
log.info("✓ RNAPredictor.predict_submission patched (flat-coords fix)")

# %%
toy = create_predictor().predict_submission("ACGUACGU", prediction_repeats=1)
assert len(toy) == 8            # ✅ one row per residue
print(toy.head())

# %%
# ────────────────────────────────────────────────────────────────────────
# 6) PREDICTION UTILITIES  ● de-duplication / aggregation safeguard  ✅
# -----------------------------------------------------------------------
# NOTE: This cell REPLACES the previous buggy version.
# Fix: drop existing "ID" column before inserting the new one.
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd
import logging


def _auto_column(df: pd.DataFrame, pref: list[str]) -> str:
    """Return first column present in *pref* (fallback → df.columns[0])."""
    for c in pref:
        if c in df.columns:
            return c
    return df.columns[0]


def _collapse_to_one_row_per_residue(df_raw: pd.DataFrame, seq_id: str) -> pd.DataFrame:
    """Reduce predictor output to one row per residue with clean IDs."""

    df = df_raw.copy()

    # 1️⃣  First repeat/angle only
    if "repeat_idx" in df.columns:
        df = df[df["repeat_idx"] == 0]
    if "angle_idx" in df.columns:
        df = df[df["angle_idx"] == 0]

    # 2️⃣  Canonical atom (phosphate) if available
    if "atom_name" in df.columns:
        canon = df[df["atom_name"] == "P"].copy()
        if not canon.empty and canon["resid"].nunique() == canon.shape[0]:
            df = canon

    # 3️⃣  Average duplicates if still present
    if not (df["resid"].is_unique and len(df) == df["resid"].nunique()):
        coord_cols = [c for c in df.columns if c[:2] in ("x_", "y_", "z_")]
        key_cols   = ["resid", "resname"]
        df = (
            df.groupby(key_cols, as_index=False)[coord_cols].mean()
              .sort_values("resid")
              .reset_index(drop=True)
        )

    # 4️⃣  Re‑index residues and rebuild ID
    df = df.sort_values("resid").reset_index(drop=True)
    df["resid"] = np.arange(1, len(df) + 1)
    if "ID" in df.columns:
        df = df.drop(columns="ID")
    df.insert(0, "ID", [f"{seq_id}_{r}" for r in df["resid"]])

    # 5️⃣  Return only required columns
    coord_cols = [c for c in df.columns if c[:2] in ("x_", "y_", "z_")]
    return df[["ID", "resname", "resid"] + coord_cols]


def process_test_sequences(test_csv: str, sample_csv: str, out_csv: str, *, batch: int = 1):
    """Generate submission file after collapsing predictions."""

    df_test = pd.read_csv(test_csv)
    logging.info("Loaded %d sequences", len(df_test))

    predictor = create_predictor()

    id_col  = _auto_column(df_test, ["id", "ID", "seq_id", "sequence_id"])
    seq_col = _auto_column(df_test, ["sequence", "Sequence", "seq", "SEQ"])

    frames: list[pd.DataFrame] = []
    for start in range(0, len(df_test), batch):
        end = min(start + batch, len(df_test))
        logging.info("Batch %d–%d", start + 1, end)
        for i in range(start, end):
            sid, seq = df_test.at[i, id_col], df_test.at[i, seq_col]
            try:
                raw  = predictor.predict_submission(seq, prediction_repeats=5)
                tidy = _collapse_to_one_row_per_residue(raw, sid)
                frames.append(tidy)
            except Exception as err:
                logging.error("%s failed: %s", sid, err)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(out_csv, index=False)
    logging.info("Saved → %s  (#rows = %s)", out_csv, f"{len(results):,}")
    return results


# %%
# ╔══════════════════════════════════════════════════════════════════════╗
# 7) TOY SANITY-CHECK – demonstrates collapse function
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n=== Toy sanity-check ===")
toy_raw  = create_predictor().predict_submission("ACGUACGU", prediction_repeats=2)
toy_comp = _collapse_to_one_row_per_residue(toy_raw, "TOY")
print(toy_comp.head())

# ╔══════════════════════════════════════════════════════════════════════╗
# 8) FULL TEST SET  (comment out to iterate faster during dev)
# ╚══════════════════════════════════════════════════════════════════════╝
if os.path.exists(TEST_SEQS) and os.path.exists(SAMPLE_SUB):
    process_test_sequences(TEST_SEQS, SAMPLE_SUB, OUTPUT_CSV, batch=1)
else:
    logging.warning("Test CSVs missing – adjust paths or upload files.")

# %%
"""
Cell 11: GENERATE PREDICTIONS & BUILD SUBMISSION
------------------------------------------------
We'll predict (x_1, y_1, z_1) for each residue, 
then replicate those coordinates for structures x_2..z_5.
Finally, we'll align with sample_submission and save submission.csv.
"""

# Predict x_1, y_1, z_1
#test_pred_x = model_x.predict(test_merged_imputed)
#test_pred_y = model_y.predict(test_merged_imputed)
#test_pred_z = model_z.predict(test_merged_imputed)

# Build submission from test_clean_df
#submission = test_clean_df.copy()

# Add predicted coords for structure 1
#submission['x_1'] = test_pred_x
#submission['y_1'] = test_pred_y
#submission['z_1'] = test_pred_z

# For simplicity, replicate for structures 2..5
#for i in [2,3,4,5]:
#    submission[f'x_{i}'] = test_pred_x
#    submission[f'y_{i}'] = test_pred_y
#    submission[f'z_{i}'] = test_pred_z

# Adjust ID format: ID + "_" + resid
#submission['ID'] = submission['ID'] + "_"  + submission['resid'].astype(str)

# Reorder columns to match sample_submission
#final_cols = list(sample_submission.columns)  # ID, resname, resid, x_1..z_5
#submission = submission[['ID','resname','resid',
#                         'x_1','y_1','z_1',
#                         'x_2','y_2','z_2',
#                         'x_3','y_3','z_3',
#                         'x_4','y_4','z_4',
#                         'x_5','y_5','z_5']]

# Merge with sample_submission to match row order
#sample_submission['sort_order'] = range(len(sample_submission))
#submission_merged = pd.merge(
#    submission,
#    sample_submission[['ID','sort_order']],
#    on='ID',
#    how='left'
#).sort_values('sort_order').drop(columns='sort_order')

# This is our final submission dataframe
#submission_df = submission_merged.copy()

# Save to CSV
#submission_df.to_csv("submission.csv", index=False)
#logging.info("submission.csv created successfully.")

#print("Cell 11 complete: Submission file saved. Ready to submit!")

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

# %%
%%bash
# Cell: show what’s inside every mounted Kaggle dataset  🔍
# --------------------------------------------------------
echo -e "\n📂  Listing the first two levels of /kaggle/working …\n"

# Change depth (-maxdepth) if you want more or fewer levels
find /kaggle/working -maxdepth 2 -mindepth 1 -print | sed 's|^|  |'

echo -e "\n✅  Done.\n"

# %%
# Cell : sanity-check submission.csv against test_sequences.csv  ✅
# ----------------------------------------------------------------
import pandas as pd, pathlib, textwrap, sys, itertools, numpy as np

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


