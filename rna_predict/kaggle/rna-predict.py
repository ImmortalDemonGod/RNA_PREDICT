# rna-predict.py

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import pathlib
import logging

import hydra
from omegaconf import DictConfig

from rna_predict.kaggle.kaggle_env import (
    setup_kaggle_environment,
    print_kaggle_input_tree,
    is_kaggle,
    print_system_info,
)
from rna_predict.kaggle.data_utils import (
    load_kaggle_data, 
    collapse_to_one_row_per_residue,
    process_test_sequences
)
from rna_predict.kaggle.submission_validator import run_sanity_checks
from rna_predict.kaggle.predictor_config import create_predictor

try:
    from rna_predict.training.train import execute_training_run
    TRAINING_MODULE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import training module/function 'rna_predict.training.train.execute_training_run': {e}. Training mode will not be available.")
    TRAINING_MODULE_AVAILABLE = False

# ==============================================================================
# GLOBAL CONSTANTS & PATH SETUP
# ==============================================================================

# Determine execution environment
IS_KAGGLE_ENVIRONMENT = is_kaggle()

# Define base paths based on environment
if IS_KAGGLE_ENVIRONMENT:
    BASE_INPUT_ROOT = pathlib.Path("/kaggle/input")
    BASE_WORKING_ROOT = pathlib.Path("/kaggle/working")
    # Specific dataset path for Kaggle
    BASE_INPUT_PATH = BASE_INPUT_ROOT / "stanford-rna-3d-folding"
else:
    # Local environment:
    # CSV Data from external drive
    BASE_INPUT_ROOT_EXTERNAL_DRIVE = pathlib.Path("/Volumes/Totallynotaharddrive/RNA_structure_PREDICT/kaggle/")
    BASE_INPUT_PATH = BASE_INPUT_ROOT_EXTERNAL_DRIVE / "stanford-rna-3d-folding"
    
    # Working directory remains local to the project
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    BASE_WORKING_ROOT = PROJECT_ROOT / "outputs" / "kaggle_working"

    # No need to create BASE_INPUT_PATH as it's on the external drive and expected to exist
    BASE_WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    logging.info(f"Local environment detected. CSV Data input path: {BASE_INPUT_PATH}")
    logging.info(f"Local environment detected. Working path: {BASE_WORKING_ROOT}")

TEST_SEQS_PATH = BASE_INPUT_PATH / "test_sequences.csv"
SAMPLE_SUB_PATH = BASE_INPUT_PATH / "sample_submission.csv"
OUTPUT_CSV_PATH = BASE_WORKING_ROOT / "submission.csv"

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
def setup_logging(cfg: DictConfig = None):
    """Configures basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging initialized.")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def perform_environment_setup_and_diagnostics():
    """Handles Kaggle environment setup and prints system diagnostics."""
    logging.info("Starting environment setup and diagnostics...")
    setup_kaggle_environment()
    
    logging.info("Displaying Kaggle input file tree...")
    # print_kaggle_input_tree() will try /kaggle/input by default.
    # For local, it will warn that /kaggle/input doesn't exist, which is acceptable for now.
    # If we want it to list BASE_INPUT_ROOT locally, kaggle_env.py would need changes.
    print_kaggle_input_tree()
    
    logging.info("Displaying system information...")
    print_system_info()
    logging.info("Environment setup and diagnostics complete.")

def verify_data_availability(cfg: DictConfig):
    """Loads Kaggle data to ensure it's accessible, though not all parts are used directly."""
    logging.info("Verifying data availability by attempting to load Kaggle data...")
    try:
        load_kaggle_data() 
        logging.info("Data sources seem available (load_kaggle_data call successful).")
    except Exception as e:
        logging.error(f"Error during load_kaggle_data: {e}. Data might not be available as expected.")
        # Depending on severity, might want to sys.exit()

def initialize_rna_predictor(cfg: DictConfig):
    """Initializes and returns the RNA predictor instance."""
    logging.info("Initializing RNA predictor...")
    try:
        predictor = create_predictor(cfg)
        logging.info("RNA predictor initialized successfully.")
        return predictor
    except Exception as e:
        logging.error(f"Failed to initialize RNA predictor: {e}")
        sys.exit("Exiting due to predictor initialization failure.") 

def run_toy_sanity_check_with_predictor(predictor, cfg: DictConfig):
    """Runs a small sanity check using the provided predictor."""
    if not predictor:
        logging.warning("Predictor not available, skipping toy sanity check.")
        return
        
    logging.info("Running toy sanity-check with the predictor...")
    try:
        toy_sequence = "ACGUACGU"
        toy_raw_output = predictor.predict_submission(toy_sequence, prediction_repeats=2)
        toy_collapsed_output = collapse_to_one_row_per_residue(toy_raw_output, "TOY_SANITY_CHECK")
        logging.info(f"Toy sanity-check for sequence '{toy_sequence}' (first 5 rows of collapsed output):")
        print(toy_collapsed_output.head().to_string()) 
    except Exception as e:
        logging.error(f"Error during toy sanity check: {e}")
    logging.info("Toy sanity-check complete.")

def process_full_test_set_and_validate(predictor, cfg: DictConfig):
    """
    Processes the full test set to generate submission.csv using the predictor,
    and then runs sanity checks on the generated file.
    """
    if not predictor:
        logging.warning("Predictor not available, skipping full test set processing.")
        return

    logging.info("Starting full test set processing...")
    if os.path.exists(TEST_SEQS_PATH) and os.path.exists(SAMPLE_SUB_PATH):
        logging.info(f"Using test sequences from: {TEST_SEQS_PATH}")
        logging.info(f"Using sample submission as template from: {SAMPLE_SUB_PATH}")
        logging.info(f"Will write output to: {OUTPUT_CSV_PATH}")

        try:
            process_test_sequences(
                predictor,               # Pass the initialized predictor
                TEST_SEQS_PATH,          # Correct: positional argument for test_csv
                SAMPLE_SUB_PATH,       # Correct: positional argument for sample_csv
                OUTPUT_CSV_PATH,       # Correct: positional argument for out_csv
                batch=1                # Correct: keyword argument 'batch'
            )

            if pathlib.Path(OUTPUT_CSV_PATH).exists():
                logging.info(f"Submission file '{OUTPUT_CSV_PATH}' created successfully.")
                logging.info("Running sanity checks on the submission file...")
                run_sanity_checks(TEST_SEQS_PATH, OUTPUT_CSV_PATH)
            else:
                logging.error(f"Submission file '{OUTPUT_CSV_PATH}' was NOT found after processing. Sanity checks skipped.")
        except Exception as e:
            logging.error(f"Error during full test set processing or validation: {e}")
            
    else:
        missing_files_msg = "Missing required files: "
        if not os.path.exists(TEST_SEQS_PATH):
            missing_files_msg += f"{TEST_SEQS_PATH} "
        if not os.path.exists(SAMPLE_SUB_PATH):
            missing_files_msg += f"{SAMPLE_SUB_PATH}"
        logging.warning(f"{missing_files_msg.strip()}. Full test set processing and sanity checks skipped.")
    logging.info("Full test set processing stage finished.")

def display_conclusions(cfg: DictConfig = None):
    """Displays concluding messages and original notebook's suggestions."""
    logging.info("Displaying script conclusions...")
    conclusion_text = """
    Script Execution Summary:
    - Environment setup and system diagnostics performed.
    - Data availability verified.
    - RNA predictor initialized.
    - Toy sanity check executed.
    - Full test set processed to generate submission file (if data available).
    - Sanity checks run on the submission file (if generated).

    Original Notebook Suggestions for Further Improvement:
    - Fine-tune hyperparameters with a broader search or Bayesian optimization.
    - Explore more advanced RNA 3D features.
    - Generate truly distinct 5 structures instead of repeating the same coordinates for submission.
    """
    print(conclusion_text)
    logging.info("Script execution complete. Review logs for details.")
    if pathlib.Path(OUTPUT_CSV_PATH).exists():
        print(f"Submission file '{OUTPUT_CSV_PATH}' is ready. Good luck on the leaderboard!")
    else:
        print(f"Submission file '{OUTPUT_CSV_PATH}' was not generated. Check logs for errors.")

def list_working_directory_contents_final(cfg: DictConfig = None):
    """Lists the contents of the working directory at the end of the script."""
    logging.info(f"Listing final contents of {BASE_WORKING_ROOT} directory...")
    # working_root = pathlib.Path("/kaggle/working") # Old line
    print(f"\n📂 Final listing of {BASE_WORKING_ROOT} (up to two levels deep):")
    if BASE_WORKING_ROOT.exists() and BASE_WORKING_ROOT.is_dir():
        for item_count, item in enumerate(sorted(BASE_WORKING_ROOT.iterdir())):
            print(f"  └── {item.name}")
            if item.is_dir():
                for sub_item_count, sub_item in enumerate(sorted(item.iterdir())):
                    print(f"      └── {sub_item.name}")
                    if sub_item_count >= 10: 
                        print("      └── ... (more items)")
                        break
            if item_count >= 20: 
                print("  └── ... (more items)")
                break
        # Check if item_count was initialized (i.e., loop ran at least once)
        if 'item_count' in locals() and item_count == 0 and not any(BASE_WORKING_ROOT.iterdir()):
             print("  └── (empty)")
        elif 'item_count' not in locals() and not any(BASE_WORKING_ROOT.iterdir()): # handles empty dir before loop
             print("  └── (empty)")
    else:
        logging.warning(f"Directory {BASE_WORKING_ROOT} does not exist or is not accessible.")
    print("\n✅ File listing complete.\n")

def run_training_pipeline(cfg: DictConfig):
    """Runs the training pipeline using the provided Hydra configuration."""
    if not TRAINING_MODULE_AVAILABLE:
        logging.error("Training module is not available. Cannot proceed with training. Please check import errors.")
        sys.exit(1)

    logging.info("Starting training pipeline...")
    try:
        # Ensure data paths in cfg are appropriate for Kaggle training if running on Kaggle
        # This might involve modifying cfg or having specific Kaggle configs for training data.
        if is_kaggle():
            logging.info("Kaggle environment detected for training. Ensure 'data' config points to /kaggle/input/ for training data.")
            # Example: cfg.data.train_csv = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
            # Example: cfg.data.val_csv = "/kaggle/input/stanford-rna-3d-folding/val_sequences.csv" # if you have one
            # Ensure hydra.run.dir or equivalent output path is /kaggle/working/
            if not cfg.hydra.run.dir.startswith("/kaggle/working"):
                logging.warning(f"Hydra run dir {cfg.hydra.run.dir} is not in /kaggle/working/. Model outputs might not be saved correctly on Kaggle.")
                # Consider forcing it: cfg.hydra.run.dir = f"/kaggle/working/{cfg.hydra.job.name}/{cfg.hydra.job.id}" or similar
        
        # Call the main training function from your training module
        execute_training_run(cfg) # Pass the full config
        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error during training pipeline: {e}", exc_info=True)
        sys.exit("Exiting due to training pipeline failure.")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
@hydra.main(config_path="../conf", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """Main function to orchestrate the script's execution flow."""
    setup_logging(cfg)
    
    perform_environment_setup_and_diagnostics()
    
    # For now, data verification might be specific to predict mode
    # We'll need to adjust this based on cfg.mode later
    if cfg.get("mode", "predict") == "predict":
        verify_data_availability(cfg) 
    
    predictor = initialize_rna_predictor(cfg)
    
    # This part is prediction-specific, will be conditional later
    if cfg.get("mode", "predict") == "predict":
        run_toy_sanity_check_with_predictor(predictor, cfg)
        process_full_test_set_and_validate(predictor, cfg)
        display_conclusions(cfg)
        list_working_directory_contents_final(cfg)
    elif cfg.mode == "train":
        logging.info("Training mode selected.")
        run_training_pipeline(cfg)
    else:
        logging.error(f"Unknown mode: {cfg.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#/Users/tomriddle1/.local/bin/uv run rna_predict/kaggle/rna-predict.py --config-path /Users/tomriddle1/RNA_PREDICT/rna_predict/conf --config-name default mode=train > /Users/tomriddle1/RNA_PREDICT/outputs/kaggle_run_training_mode.log 2>&1
#