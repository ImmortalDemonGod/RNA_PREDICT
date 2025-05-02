"""
Simple test script for download_file function
"""

import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

from rna_predict.pipeline.stageA.run_stageA import download_file

class SimpleTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="simple_test_")
        self.download_path = os.path.join(self.test_dir, "test_file.bin")
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    @patch("shutil.copyfileobj")
    def test_download_succeeds_after_retries(self, mock_copyfileobj, mock_sleep, mock_urlopen):
        # Create a mock response for the successful attempt
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        
        # Set up the side effect sequence
        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection refused"),
            urllib.error.URLError("Timeout"),
            mock_response
        ]
        
        # Ensure the file doesn't exist
        if os.path.exists(self.download_path):
            os.remove(self.download_path)
        
        # Call the function
        download_file("http://example.com/retry-test", self.download_path, debug_logging=True)
        
        # Verify urlopen was called 3 times
        self.assertEqual(mock_urlopen.call_count, 3)
        
        # Verify sleep was called 2 times
        self.assertEqual(mock_sleep.call_count, 2)
        
        # Verify copyfileobj was called once (on the successful attempt)
        mock_copyfileobj.assert_called_once()

if __name__ == "__main__":
    unittest.main()
