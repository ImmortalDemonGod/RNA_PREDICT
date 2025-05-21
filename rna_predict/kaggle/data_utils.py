import sys
import logging
import pandas as pd
import numpy as np
from rna_predict.kaggle.predictor_config import create_predictor

def load_kaggle_data():
    """
    Loads train, validation, test, and sample submission CSVs from Kaggle paths.
    Returns:
        train_sequences, train_labels, validation_sequences, validation_labels, test_sequences, sample_submission
    Raises SystemExit on failure.
    """
    TRAIN_SEQUENCES_PATH = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
    TRAIN_LABELS_PATH    = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
    VALID_SEQUENCES_PATH = "/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv"
    VALID_LABELS_PATH    = "/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
    TEST_SEQUENCES_PATH  = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
    SAMPLE_SUB_PATH      = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
    try:
        train_sequences = pd.read_csv(TRAIN_SEQUENCES_PATH)
        train_labels = pd.read_csv(TRAIN_LABELS_PATH)
        validation_sequences = pd.read_csv(VALID_SEQUENCES_PATH)
        validation_labels = pd.read_csv(VALID_LABELS_PATH)
        test_sequences = pd.read_csv(TEST_SEQUENCES_PATH)
        sample_submission = pd.read_csv(SAMPLE_SUB_PATH)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)
    logging.info(f"train_sequences: {train_sequences.shape}, train_labels: {train_labels.shape}")
    logging.info(f"validation_sequences: {validation_sequences.shape}, validation_labels: {validation_labels.shape}")
    logging.info(f"test_sequences: {test_sequences.shape}, sample_submission: {sample_submission.shape}")
    return train_sequences, train_labels, validation_sequences, validation_labels, test_sequences, sample_submission


def auto_column(df: pd.DataFrame, pref: list[str]) -> str:
    """Return first column present in *pref* (fallback → df.columns[0])."""
    for c in pref:
        if c in df.columns:
            return c
    return df.columns[0]


def collapse_to_one_row_per_residue(df_raw: pd.DataFrame, seq_id: str) -> pd.DataFrame:
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

    id_col  = auto_column(df_test, ["id", "ID", "seq_id", "sequence_id"])
    seq_col = auto_column(df_test, ["sequence", "Sequence", "seq", "SEQ"])

    frames: list[pd.DataFrame] = []
    for start in range(0, len(df_test), batch):
        end = min(start + batch, len(df_test))
        logging.info("Batch %d–%d", start + 1, end)
        for i in range(start, end):
            sid, seq = df_test.at[i, id_col], df_test.at[i, seq_col]
            try:
                raw  = predictor.predict_submission(seq, prediction_repeats=5)
                tidy = collapse_to_one_row_per_residue(raw, sid)
                frames.append(tidy)
            except Exception as err:
                logging.error("%s failed: %s", sid, err)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(out_csv, index=False)
    logging.info("Saved → %s  (#rows = %s)", out_csv, f"{len(results):,}")
    return results
