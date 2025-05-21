# rna-predict.py

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import pathlib 
import logging

from rna_predict.kaggle.kaggle_env import (
    setup_kaggle_environment,
    print_kaggle_input_tree,
    print_system_info
)
from rna_predict.kaggle.data_utils import (
    load_kaggle_data, 
    collapse_to_one_row_per_residue,
    process_test_sequences
)
from rna_predict.kaggle.submission_validator import run_sanity_checks
from rna_predict.kaggle.predictor_config import create_predictor

# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================
TEST_SEQS_PATH = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
SAMPLE_SUB_PATH = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
OUTPUT_CSV_PATH = "submission.csv"

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
def setup_logging():
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
    print_kaggle_input_tree()
    
    logging.info("Displaying system information...")
    print_system_info()
    logging.info("Environment setup and diagnostics complete.")

def verify_data_availability():
    """Loads Kaggle data to ensure it's accessible, though not all parts are used directly."""
    logging.info("Verifying data availability by attempting to load Kaggle data...")
    try:
        load_kaggle_data() 
        logging.info("Data sources seem available (load_kaggle_data call successful).")
    except Exception as e:
        logging.error(f"Error during load_kaggle_data: {e}. Data might not be available as expected.")
        # Depending on severity, might want to sys.exit()

def initialize_rna_predictor():
    """Initializes and returns the RNA predictor instance."""
    logging.info("Initializing RNA predictor...")
    try:
        predictor = create_predictor()
        logging.info("RNA predictor initialized successfully.")
        return predictor
    except Exception as e:
        logging.error(f"Failed to initialize RNA predictor: {e}")
        sys.exit("Exiting due to predictor initialization failure.") 

def run_toy_sanity_check_with_predictor(predictor):
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

def process_full_test_set_and_validate(predictor):
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
                test_sequences_path=TEST_SEQS_PATH,
                sample_submission_path=SAMPLE_SUB_PATH,
                output_csv_path=OUTPUT_CSV_PATH,
                predictor_instance=predictor,
                batch_size_override=1 
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

def display_conclusions():
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


def list_working_directory_contents_final():
    """Lists the contents of the /kaggle/working directory at the end of the script."""
    logging.info("Listing final contents of /kaggle/working directory...")
    working_root = pathlib.Path("/kaggle/working")
    print("\n📂 Final listing of /kaggle/working (up to two levels deep):")
    if working_root.exists() and working_root.is_dir():
        for item_count, item in enumerate(sorted(working_root.iterdir())):
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
        if item_count == 0 and not any(working_root.iterdir()):
             print("  └── (empty)")
    else:
        logging.warning(f"Directory {working_root} does not exist or is not accessible.")
    print("\n✅ File listing complete.\n")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Main function to orchestrate the script's execution flow."""
    setup_logging()
    
    perform_environment_setup_and_diagnostics()
    
    verify_data_availability() 
    
    predictor = initialize_rna_predictor()
    
    run_toy_sanity_check_with_predictor(predictor)
    
    process_full_test_set_and_validate(predictor)
    
    display_conclusions()
    
    list_working_directory_contents_final()

if __name__ == "__main__":
    main()
