import logging
import pandas as pd
import numpy as np
import pathlib
from rna_predict.kaggle.kaggle_env import is_kaggle

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def load_kaggle_data(target_sequences_path=None, sample_submission_path=None):
    """
    Loads the Kaggle competition data (train, test, sample submission).

    Args:
        target_sequences_path (str, optional): Path to the test sequences CSV.
            If None, defaults to the standard Kaggle path.
        sample_submission_path (str, optional): Path to the sample submission CSV.
            If None, defaults to the standard Kaggle path.

    Returns:
        tuple: (train_df, test_df, sample_submission_df)
    """
    try:
        # Determine execution environment
        IS_KAGGLE_ENVIRONMENT = is_kaggle()

        # Define base paths based on environment
        if IS_KAGGLE_ENVIRONMENT:
            BASE_INPUT_ROOT = pathlib.Path("/kaggle/input")
            # Specific dataset path for Kaggle
            BASE_INPUT_PATH = BASE_INPUT_ROOT / "stanford-rna-3d-folding"
            logger.info("Kaggle environment detected for data loading.")
        else:
            # Local environment: CSV Data from external drive
            BASE_INPUT_ROOT_EXTERNAL_DRIVE = pathlib.Path("/Volumes/Totallynotaharddrive/RNA_structure_PREDICT/kaggle/")
            BASE_INPUT_PATH = BASE_INPUT_ROOT_EXTERNAL_DRIVE / "stanford-rna-3d-folding"
            
            # The directory is expected to exist on the external drive
            logger.info(f"Local environment detected. CSV Data input path: {BASE_INPUT_PATH}")

        # Define default paths using the determined base path
        default_train_path = BASE_INPUT_PATH / "train_sequences.csv"
        default_test_path = BASE_INPUT_PATH / "test_sequences.csv"
        default_sample_sub_path = BASE_INPUT_PATH / "sample_submission.csv"

        # Use provided paths or defaults
        train_path = default_train_path # train_sequences.csv is always loaded from default
        test_path = pathlib.Path(target_sequences_path) if target_sequences_path else default_test_path
        sample_sub_path = pathlib.Path(sample_submission_path) if sample_submission_path else default_sample_sub_path

        logger.info(f"Loading train data from: {train_path}")
        train_df = pd.read_csv(train_path)
        
        logger.info(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loading sample submission from: {sample_sub_path}")
        sample_submission_df = pd.read_csv(sample_sub_path)
        
        logger.info("Data loaded successfully.")
        return train_df, test_df, sample_submission_df
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        # Re-raise the exception so the calling script can handle it if needed
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        raise


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


def process_test_sequences(predictor, test_csv: str, sample_csv: str, out_csv: str, *, batch: int = 1):
    """
    Process test sequences and generate a submission file.

    Args:
        predictor: An initialized RNAPredictor instance.
        test_csv (str): Path to the test sequences CSV file.
        sample_csv (str): Path to the sample submission CSV file.
        out_csv (str): Path where the output submission CSV will be saved.
        batch (int, optional): Batch size for processing sequences. Defaults to 1.

    Returns:
        pd.DataFrame: The resulting DataFrame that was saved to CSV.
    """
    df_test = pd.read_csv(test_csv)
    df_sample = pd.read_csv(sample_csv)
    logging.info("Loaded %d sequences", len(df_test))

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
