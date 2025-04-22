import subprocess
from datetime import datetime
import os

# Determine the project root directory (parent of rna_predict/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_python_file(file_path, output_file, extra_args=None):
    """Run a Python file and capture its output.

    Args:
        file_path: Path to the Python file to run
        output_file: Path to the output file to write to
        extra_args: Optional list of extra arguments to pass to the Python file
    """
    rel_path = os.path.relpath(file_path, PROJECT_ROOT) # Get relative path for cleaner output
    try:
        # Write the file header to the output file
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Output from: {rel_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")

        # Build the command
        cmd = ["uv", "run", file_path]
        if extra_args:
            cmd.extend(extra_args)

        # Run the Python file and capture output
        result = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              check=False, # Set check=False to handle non-zero exit codes manually
                              cwd=PROJECT_ROOT) # <--- Set Current Working Directory

        # Append the output to the file
        with open(output_file, 'a') as f:
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(result.stdout)
            if result.stderr:
                # Add a check to ensure stderr is not empty before writing header
                if result.stderr.strip():
                    f.write("STDERR:\n")
                    f.write(result.stderr)
            f.write("\n")

        # Check the return code to determine success
        if result.returncode != 0:
            # Optionally log the specific error code
            with open(output_file, 'a') as f:
                f.write(f"--- PROCESS EXITED WITH ERROR CODE: {result.returncode} ---\n\n")
            return False # Indicate failure
        else:
            return True # Indicate success

    except FileNotFoundError:
         with open(output_file, 'a') as f:
            f.write(f"Error running {rel_path}: File not found.\n\n")
         return False
    except Exception as e:
        with open(output_file, 'a') as f:
            f.write(f"Error running {rel_path}: {str(e)}\n\n")
        return False

def main():
    # Define standardized test sequence to use across all pipeline stages
    # This should match the sequence in test_data.yaml
    test_sequence = "ACGUACGU"

    # List of Python files to run with their extra arguments
    # Use paths relative to PROJECT_ROOT for robustness
    pipeline_stages = [
        (os.path.join(PROJECT_ROOT, "rna_predict/pipeline/stageA/run_stageA.py"),
         [f"test_data.sequence={test_sequence}"]),
        (os.path.join(PROJECT_ROOT, "rna_predict/pipeline/stageB/main.py"),
         [f"test_data.sequence={test_sequence}"]),
        (os.path.join(PROJECT_ROOT, "rna_predict/pipeline/stageC/stage_c_reconstruction.py"),
         [f"test_data.sequence={test_sequence}"]),
        (os.path.join(PROJECT_ROOT, "rna_predict/pipeline/stageD/run_stageD.py"),
         [f"test_data.sequence={test_sequence}"]),
    ]

    # Additional files to run without test sequence override
    additional_files = [
        os.path.join(PROJECT_ROOT, "rna_predict/interface.py"),
        os.path.join(PROJECT_ROOT, "rna_predict/main.py"),
        os.path.join(PROJECT_ROOT, "rna_predict/print_rna_pipeline_output.py"),
        # os.path.join(PROJECT_ROOT, "rna_predict/run_full_pipeline.py") # Avoid running itself
    ]

    # Output file path
    output_file = "combined_pipeline_output.txt"

    # Clear the output file if it exists
    with open(output_file, 'w') as f:
        f.write("RNA Pipeline Combined Output\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Using standardized test sequence: {test_sequence}\n")
        f.write(f"{'='*80}\n\n")

    all_success = True # Track overall success

    # Run each pipeline stage with the standardized test sequence
    for file_path, extra_args in pipeline_stages:
        # Check if file exists before trying to run
        if not os.path.exists(file_path):
            print(f"Skipping {os.path.relpath(file_path, PROJECT_ROOT)}: File not found.")
            with open(output_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Skipped: {os.path.relpath(file_path, PROJECT_ROOT)} (File not found)\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
            all_success = False # Mark overall run as failed if any file is skipped
            continue # Move to the next file

        print(f"Running {os.path.relpath(file_path, PROJECT_ROOT)} with args: {extra_args}...") # Use relative path for cleaner console output
        success = run_python_file(file_path, output_file, extra_args)
        if success:
            print(f"Successfully ran {os.path.relpath(file_path, PROJECT_ROOT)}")
        else:
            print(f"Failed to run {os.path.relpath(file_path, PROJECT_ROOT)}")
            all_success = False # Mark overall run as failed
            # Optional: Stop execution on first failure
            # print("Stopping pipeline due to failure.")
            # break

    # Run additional files without test sequence override
    for file_path in additional_files:
        # Check if file exists before trying to run
        if not os.path.exists(file_path):
            print(f"Skipping {os.path.relpath(file_path, PROJECT_ROOT)}: File not found.")
            with open(output_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Skipped: {os.path.relpath(file_path, PROJECT_ROOT)} (File not found)\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
            all_success = False # Mark overall run as failed if any file is skipped
            continue # Move to the next file

        print(f"Running {os.path.relpath(file_path, PROJECT_ROOT)}...") # Use relative path for cleaner console output
        success = run_python_file(file_path, output_file)
        if success:
            print(f"Successfully ran {os.path.relpath(file_path, PROJECT_ROOT)}")
        else:
            print(f"Failed to run {os.path.relpath(file_path, PROJECT_ROOT)}")
            all_success = False # Mark overall run as failed

    print(f"\n{'='*80}")
    if all_success:
        print("Pipeline completed successfully.")
    else:
        print("Pipeline completed with one or more failures.")
    print(f"All outputs have been combined into: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()