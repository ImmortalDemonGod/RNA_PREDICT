import sys
import logging
import pandas as pd

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
