#!/usr/bin/env python
"""
Simple script to test if X3DNA-DSSR is installed and working correctly.
"""

import subprocess
import sys

def test_dssr_installation(dssr_path="x3dna-dssr"):
    """
    Checks whether the X3DNA-DSSR executable is available and functioning.
    
    Attempts to run the specified DSSR executable with the '--version' flag to verify installation and basic operability.
    
    Args:
        dssr_path: Path to the X3DNA-DSSR executable. Defaults to 'x3dna-dssr'.
    
    Returns:
        True if the executable is found and runs successfully; False otherwise.
    """
    try:
        # Try to run DSSR with the --version flag
        result = subprocess.run(
            [dssr_path, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print the version information
        print("X3DNA-DSSR is installed and working correctly!")
        print(f"Version information: {result.stdout.strip()}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Could not find X3DNA-DSSR at '{dssr_path}'")
        print("Make sure X3DNA-DSSR is installed and the path is correct.")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"Error running X3DNA-DSSR: {e}")
        print(f"STDERR: {e.stderr}")
        return False

if __name__ == "__main__":
    # Check if a custom path was provided
    dssr_path = sys.argv[1] if len(sys.argv) > 1 else "x3dna-dssr"
    
    print(f"Testing X3DNA-DSSR installation at: {dssr_path}")
    success = test_dssr_installation(dssr_path)
    
    if not success:
        print("\nTroubleshooting tips:")
        print("1. Make sure you've obtained a license and downloaded X3DNA-DSSR")
        print("2. Check that the executable has the correct permissions (chmod +x)")
        print("3. Verify that the executable is in your PATH or provide the full path")
        print("4. Refer to the setup guide at docs/guides/x3dna_dssr_setup.md")
        sys.exit(1)
