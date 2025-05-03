"""
Unit tests for the download_file function in run_stageA.py
"""

import os
import shutil
import tempfile
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from rna_predict.pipeline.stageA.run_stageA import download_file


class TestDownloadFile(unittest.TestCase):
    """Tests for download_file function."""
    
    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp(prefix="download_file_tests_")
        self.download_path = os.path.join(self.test_dir, "test_download.bin")
    
    def tearDown(self):
        """Clean up temporary directory after tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_download_new_file(self):
        """Test downloading a new file."""
        with patch("rna_predict.pipeline.stageA.run_stageA.urllib.request.urlopen") as mock_urlopen, \
             patch("rna_predict.pipeline.stageA.run_stageA.shutil.copyfileobj") as mock_copyfileobj:
            
            # Set up mock response
            mock_response = MagicMock()
            mock_response.__enter__.return_value = mock_response
            mock_urlopen.return_value = mock_response
            
            # Call function
            download_file("http://example.com/test", self.download_path)
            
            # Verify urlopen was called with correct URL and timeout
            mock_urlopen.assert_called_once_with("http://example.com/test", timeout=30)
            
            # Verify copyfileobj was called
            mock_copyfileobj.assert_called_once()
    
    def test_download_succeeds_after_retries(self):
        """Test that download retries on failure and eventually succeeds."""
        with patch("rna_predict.pipeline.stageA.run_stageA.urllib.request.urlopen") as mock_urlopen, \
             patch("rna_predict.pipeline.stageA.run_stageA.time.sleep") as mock_sleep, \
             patch("rna_predict.pipeline.stageA.run_stageA.shutil.copyfileobj") as mock_copyfileobj:
            
            # Set up mock response for successful attempt
            mock_response = MagicMock()
            mock_response.__enter__.return_value = mock_response
            
            # Set up side effect sequence: two failures, then success
            mock_urlopen.side_effect = [
                urllib.error.URLError("Connection refused"),
                urllib.error.URLError("Timeout"),
                mock_response
            ]
            
            # Call function
            download_file("http://example.com/retry-test", self.download_path)
            
            # Verify urlopen was called 3 times
            self.assertEqual(mock_urlopen.call_count, 3)
            
            # Verify sleep was called 2 times (max_retries - 1)
            self.assertEqual(mock_sleep.call_count, 2)
            
            # Verify copyfileobj was called once (on successful attempt)
            mock_copyfileobj.assert_called_once()
    
    def test_download_fails_after_retries(self):
        """Test that download raises RuntimeError after all retries fail."""
        with patch("rna_predict.pipeline.stageA.run_stageA.urllib.request.urlopen",
                  side_effect=urllib.error.URLError("No route")) as mock_urlopen, \
             patch("rna_predict.pipeline.stageA.run_stageA.time.sleep") as mock_sleep:
            
            # Call function and expect RuntimeError
            with self.assertRaises(RuntimeError) as cm:
                download_file("http://bad-url.com", self.download_path)
            
            # Verify error message includes retry count
            self.assertIn("after 3 attempts", str(cm.exception))
            
            # Verify urlopen was called 3 times (default max_retries)
            self.assertEqual(mock_urlopen.call_count, 3)
            
            # Verify sleep was called 2 times (max_retries - 1)
            self.assertEqual(mock_sleep.call_count, 2)
    
    def test_existing_file_skips_download(self):
        """Test that download is skipped if file already exists."""
        # Create existing file
        with open(self.download_path, "wb") as f:
            f.write(b"Existing data")
        
        with patch("rna_predict.pipeline.stageA.run_stageA.urllib.request.urlopen") as mock_urlopen:
            # Call function
            download_file("http://example.com/skip-test", self.download_path)
            
            # Verify urlopen was not called
            mock_urlopen.assert_not_called()
            
            # Verify file content was not changed
            with open(self.download_path, "rb") as f:
                self.assertEqual(f.read(), b"Existing data")


if __name__ == "__main__":
    unittest.main()
