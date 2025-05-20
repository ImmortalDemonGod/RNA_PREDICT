"""
Simple test script for download_file function
"""

import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import urllib.error

from rna_predict.pipeline.stageA.run_stageA import download_file

def test_download_succeeds_after_retries():
    """
    Tests that download_file retries after failures and succeeds on a subsequent attempt.
    
    Simulates network errors on the first two download attempts and verifies that
    download_file retries the correct number of times before completing successfully.
    """
    # Create a temporary directory and file path
    test_dir = tempfile.mkdtemp(prefix="download_test_")
    test_file_path = os.path.join(test_dir, "test_file.bin")
    
    try:
        # Ensure the file doesn't exist
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        
        # Create mocks
        with patch("rna_predict.pipeline.stageA.run_stageA.urllib.request.urlopen") as mock_urlopen, \
             patch("rna_predict.pipeline.stageA.run_stageA.time.sleep") as mock_sleep, \
             patch("rna_predict.pipeline.stageA.run_stageA.shutil.copyfileobj") as mock_copyfileobj:
            
            # Create a mock response for the successful attempt
            mock_response = MagicMock()
            mock_response.__enter__.return_value = mock_response
            
            # Set up the side effect sequence
            mock_urlopen.side_effect = [
                urllib.error.URLError("Connection refused"),
                urllib.error.URLError("Timeout"),
                mock_response
            ]
            
            # Call the function
            download_file("http://example.com/retry-test", test_file_path, debug_logging=True)
            
            # Verify urlopen was called 3 times
            assert mock_urlopen.call_count == 3, f"Expected 3 calls to urlopen, got {mock_urlopen.call_count}"
            
            # Verify sleep was called 2 times
            assert mock_sleep.call_count == 2, f"Expected 2 calls to sleep, got {mock_sleep.call_count}"
            
            # Verify copyfileobj was called once
            assert mock_copyfileobj.call_count == 1, f"Expected 1 call to copyfileobj, got {mock_copyfileobj.call_count}"
            
            print("Test passed!")
    finally:
        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    test_download_succeeds_after_retries()
