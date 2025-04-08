import subprocess
import sys
from datetime import datetime
import os

def run_python_file(file_path, output_file):
    """Run a Python file and capture its output."""
    rel_path = os.path.relpath(file_path) # Get relative path early for error message
    try:
        # Write the file header to the output file
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Output from: {rel_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
        
        # Run the Python file and capture output
        result = subprocess.run([sys.executable, file_path], 
                              capture_output=True, 
                              text=True,
                              check=False) # Set check=False to handle non-zero exit codes manually
        
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
    # List of Python files to run
    python_files = [
        "/Users/tomriddle1/RNA_PREDICT/rna_predict/pipeline/stageA/run_stageA.py",
        "/Users/tomriddle1/RNA_PREDICT/rna_predict/pipeline/stageB/main.py",
        "/Users/tomriddle1/RNA_PREDICT/rna_predict/pipeline/stageC/stage_c_reconstruction.py",
        "/Users/tomriddle1/RNA_PREDICT/rna_predict/pipeline/stageD/run_stageD.py",
        "/Users/tomriddle1/RNA_PREDICT/rna_predict/interface.py",
        "/Users/tomriddle1/RNA_PREDICT/rna_predict/main.py",
        "/Users/tomriddle1/RNA_PREDICT/rna_predict/print_rna_pipeline_output.py",
        "/Users/tomriddle1/RNA_PREDICT/rna_predict/run_full_pipeline.py" # Note: Original comment about recursion was incorrect.
    ]
    
    # Output file path
    output_file = "combined_pipeline_output.txt"
    
    # Clear the output file if it exists
    with open(output_file, 'w') as f:
        f.write("RNA Pipeline Combined Output\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    all_success = True # Track overall success
    # Run each Python file
    for file_path in python_files:
        # Check if file exists before trying to run
        if not os.path.exists(file_path):
            print(f"Skipping {os.path.relpath(file_path)}: File not found.")
            with open(output_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Skipped: {os.path.relpath(file_path)} (File not found)\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
            all_success = False # Mark overall run as failed if any file is skipped
            continue # Move to the next file
            
        print(f"Running {os.path.relpath(file_path)}...") # Use relative path for cleaner console output
        success = run_python_file(file_path, output_file)
        if success:
            print(f"Successfully ran {os.path.relpath(file_path)}")
        else:
            print(f"Failed to run {os.path.relpath(file_path)}")
            all_success = False # Mark overall run as failed
            # Optional: Stop execution on first failure
            # print("Stopping pipeline due to failure.")
            # break 
    
    print(f"\n{'='*80}")
    if all_success:
        print("Pipeline completed successfully.")
    else:
        print("Pipeline completed with one or more failures.")
    print(f"All outputs have been combined into: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()