import subprocess
import sys
from datetime import datetime
import os

def run_python_file(file_path, output_file):
    """Run a Python file and capture its output."""
    try:
        # Get the relative path for cleaner output
        rel_path = os.path.relpath(file_path)
        
        # Write the file header to the output file
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Output from: {rel_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
        
        # Run the Python file and capture output
        result = subprocess.run([sys.executable, file_path], 
                              capture_output=True, 
                              text=True)
        
        # Append the output to the file
        with open(output_file, 'a') as f:
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(result.stdout)
            if result.stderr:
                f.write("STDERR:\n")
                f.write(result.stderr)
            f.write("\n")
            
        return True
    except Exception as e:
        with open(output_file, 'a') as f:
            f.write(f"Error running {file_path}: {str(e)}\n\n")
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
        "/Users/tomriddle1/RNA_PREDICT/rna_predict/run_full_pipeline.py"
    ]
    
    # Output file path
    output_file = "combined_pipeline_output.txt"
    
    # Clear the output file if it exists
    with open(output_file, 'w') as f:
        f.write("RNA Pipeline Combined Output\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    # Run each Python file
    for file_path in python_files:
        print(f"Running {file_path}...")
        success = run_python_file(file_path, output_file)
        if success:
            print(f"Successfully ran {file_path}")
        else:
            print(f"Failed to run {file_path}")
    
    print(f"\nAll outputs have been combined into: {output_file}")

if __name__ == "__main__":
    main() 