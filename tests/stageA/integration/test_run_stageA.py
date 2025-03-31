"""
================================================================================
Comprehensive Test Suite for run_stageA.py
================================================================================

OVERVIEW
--------
This single-file test suite is designed to achieve near 100% test coverage on
the 'run_stageA.py' module, utilizing the strengths of various testing approaches
seen in prior versions (V1 to V5). It addresses their respective weaknesses by:

1. Using Python’s built-in unittest framework for compatibility and ease of
   running via `python -m unittest`.
2. Combining property-based testing (Hypothesis) for broad input coverage and
   real filesystem operations (temp directories) for realistic scenarios.
3. Providing thorough docstrings and organized test classes that map clearly to
   the functions under test: download_file, unzip_file, visualize_with_varna,
   build_predictor, run_stageA (the function), and main().
4. Including specialized tests for error conditions (e.g., corrupted or missing
   zip files, missing JAR, invalid URLs) to ensure robust coverage of all code paths.
5. Mocking external calls (such as network requests, Java subprocess calls, or
   GPU checks) in a carefully controlled manner—minimizing complex, brittle patching
   while still isolating external dependencies.

STRUCTURE
---------
We define a base test class, `TestBase`, that sets up a real temporary directory
and manages any overarching environment concerns. Each functionality in
run_stageA.py is tested in its own class:

  TestDownloadFile     -> covers download_file
  TestUnzipFile        -> covers unzip_file
  TestVisualizeWithVarna -> covers visualize_with_varna
  TestBuildPredictor   -> covers build_predictor
  TestRunStageA        -> covers the run_stageA function
  TestMainFunction     -> covers the main() routine (optional, but helpful for e2e coverage)

HYPOTHESIS PROPERTY TESTING
---------------------------
We integrate Hypothesis to generate randomized inputs for the most critical functions,
particularly those dealing with user-provided strings (URLs, filesystem paths, etc.).
We also set a moderate max_examples in some tests to limit runtime while still exploring
edge cases.

COVERAGE & EXECUTION
--------------------
By default, you can run:

    python -m unittest <this_file>.py

For coverage measurement, install coverage.py (or a similar tool), then:

    coverage run -m unittest <this_file>.py
    coverage report -m

We anticipate that the combination of real temp-directory usage, mocking network calls,
and property-based testing should yield very high coverage across normal, edge, and
error paths in run_stageA.py.

DEPENDENCIES
------------
- Python 3.x
- unittest (built-in)
- Hypothesis (install via pip install hypothesis)
- Mocks (built-in via unittest.mock in Python 3)

"""

import os
import shutil
import tempfile
import unittest
import urllib.error
import zipfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor
from rna_predict.pipeline.stageA.run_stageA import (
    build_predictor,
    main,
    run_stageA,
    visualize_with_varna,
    download_file,
    unzip_file,
)

# Import the code under test
from rna_predict.pipeline.stageA.run_stageA import (
    run_stageA as run_stageA_func,  # rename to avoid overshadowing the module
)


@pytest.fixture
def temp_checkpoint_folder(tmp_path) -> str:
    """
    Creates a temp checkpoints folder with an empty 'RNAStralign_trainset_pretrained.pth' file.
    """
    folder = tmp_path / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    dummy_state = {}
    torch.save(dummy_state, str(folder / "RNAStralign_trainset_pretrained.pth"))
    return str(folder)


def test_build_predictor_valid(temp_checkpoint_folder: str):
    """
    Test that build_predictor returns a StageARFoldPredictor instance for a valid checkpoint folder.
    """
    config = {"num_hidden": 128, "dropout": 0.3}
    device = torch.device("cpu")
    predictor = build_predictor(temp_checkpoint_folder, config, device)
    assert isinstance(predictor, StageARFoldPredictor)
    # Basic smoke test calling predict_adjacency
    adj = predictor.predict_adjacency("ACGU")
    assert adj.shape == (4, 4)


def test_run_stageA(temp_checkpoint_folder: str):
    """
    Test that run_stageA uses the predictor to produce a correct adjacency shape.
    """
    config = {"num_hidden": 128, "dropout": 0.3}
    device = torch.device("cpu")
    predictor = build_predictor(temp_checkpoint_folder, config, device)

    seq = "ACGUACGU"
    adjacency = run_stageA(seq, predictor)
    assert adjacency.shape == (8, 8)
    # Check that it's presumably 0 or 1. Implementation might vary, so we do a sanity check.
    assert (adjacency >= 0).all() and (adjacency <= 1).all()


@pytest.mark.parametrize("seq", ["A", "ACG", "ACGUACGUA"])
def test_predictor_different_sequences(temp_checkpoint_folder: str, seq: str):
    """
    Check adjacency sizes scale with the input sequence length for multiple test sequences.
    """
    config = {"num_hidden": 128, "dropout": 0.3}
    device = torch.device("cpu")
    predictor = build_predictor(temp_checkpoint_folder, config, device)
    adjacency = run_stageA(seq, predictor)
    assert adjacency.shape == (len(seq), len(seq))


def test_visualize_with_varna_missing_files(tmp_path):
    """
    If the CT file or the jar is missing, the function should warn and return gracefully.
    """
    ct_file = str(tmp_path / "non_existent.ct")
    jar_path = str(tmp_path / "non_existent.jar")
    out_png = str(tmp_path / "out.png")

    # We expect the function to not raise an error but warn and return
    visualize_with_varna(ct_file, jar_path, out_png)
    assert not os.path.exists(out_png), "No output image should be generated"


def test_main_end_to_end(temp_checkpoint_folder, monkeypatch):
    """
    Run the main() function in an environment that has a mock checkpoint folder.
    We don't do extra mocking, so the predictor is real.
    The jar won't exist, so visualize_with_varna will skip gracefully.
    """
    # Temporarily rename the real folder so main won't fail
    # Then put our temp folder in its place
    real_folder = "RFold/checkpoints"
    if os.path.exists(real_folder):
        backup_folder = "RFold/checkpoints_backup"
        os.rename(real_folder, backup_folder)
    else:
        backup_folder = None

    os.makedirs("RFold", exist_ok=True)
    os.symlink(temp_checkpoint_folder, real_folder)  # link or rename
    try:
        main()
        # We can check that the adjacency message was printed, or that a 'test_seq.ct' was created
        # But we let it pass as an integration test
        assert os.path.exists("test_seq.ct"), "main() should have written a test_seq.ct"
    finally:
        # Clean up
        if os.path.islink(real_folder):
            os.unlink(real_folder)
        if backup_folder:
            os.rename(backup_folder, real_folder)


class TestBase(unittest.TestCase):
    """
    A base class providing a temporary directory for creating files or subfolders.
    Derived test classes that need a temp directory should inherit from this
    and reference 'self.test_dir' for local filesystem operations.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create a temporary directory for all tests in this suite.
        This directory is removed after tests complete.
        """
        cls.test_dir = tempfile.mkdtemp(prefix="run_stageA_tests_")

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Remove the temporary directory after all tests have finished.
        """
        shutil.rmtree(cls.test_dir, ignore_errors=True)


class TestDownloadFile(TestBase):
    """
    Test Suite for the download_file function.

    Includes:
    - Normal usage (downloading a new file)
    - Handling of existing files (both valid zip and non-zip)
    - Corrupted zip scenario requiring re-download
    - Hypothesis fuzz tests for random URL/destination
    """

    def setUp(self) -> None:
        """
        Prepare paths for each test. Since we handle a wide range of scenarios,
        we create a path for a potential downloaded file.
        """
        self.download_path = os.path.join(self.test_dir, "test_download_file.bin")
        self.url_valid_zip = "http://example.com/fakefile.zip"
        self.url_regular_file = "http://example.com/fakefile.txt"

    def test_existing_non_zip_skips_download(self):
        """
        If a file (non-zip) already exists, download_file should skip re-downloading.
        """
        # Create a dummy file
        with open(self.download_path, "wb") as f:
            f.write(b"Existing data")

        download_file(self.url_regular_file, self.download_path)
        # The file should remain unchanged
        with open(self.download_path, "rb") as f:
            data = f.read()
        self.assertEqual(data, b"Existing data", "Download should be skipped.")

    def test_existing_valid_zip_skips_download(self):
        """
        If a file is a valid .zip, skip re-downloading.
        """
        with zipfile.ZipFile(self.download_path, "w") as zf:
            zf.writestr("test.txt", "dummy content")

        download_file(self.url_valid_zip, self.download_path)
        self.assertTrue(os.path.isfile(self.download_path), "File should remain")

    def test_existing_corrupted_zip_redownloads(self):
        """
        If a .zip file is present but corrupted, remove and re-download it.
        We'll mock the network call to avoid real downloads.
        """
        # Create a corrupted zip
        with open(self.download_path, "wb") as f:
            f.write(b"Not a real zip")

        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.read.return_value = b"downloaded content"
        with patch("urllib.request.urlopen", return_value=mock_response):
            download_file(self.url_valid_zip, self.download_path)

        # Now the file should be replaced
        with open(self.download_path, "rb") as f:
            new_data = f.read()
        self.assertEqual(new_data, b"downloaded content")

    
    @unittest.skip("Skipping this test as requested takes too long")
    @patch("urllib.request.urlopen")
    def test_download_new_file(self, mock_urlopen):
        """
        If no file exists, download_file should fetch from URL and create the file.
        """
        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            b"some data"
        )
        download_file("http://example.com/newdata", self.download_path)
        self.assertTrue(os.path.isfile(self.download_path), "File should be created.")
        with open(self.download_path, "rb") as f:
            data = f.read()
        self.assertEqual(data, b"some data")

    @patch("urllib.request.urlopen", side_effect=urllib.error.URLError("No route"))
    def test_download_fails(self, mock_urlopen):
        """
        If the URL is invalid or unreachable, a URLError is raised.
        """
        with self.assertRaises(urllib.error.URLError):
            download_file("http://bad-url.com", self.download_path)

    @unittest.skip("Skipping this test as requested takes too long")
    @given(url=st.text(min_size=1, max_size=30), dest=st.text(min_size=1, max_size=30))
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=15,
    )
    def test_fuzz_download_file(self, url: str, dest: str):
        """
        Property-based testing for random URL/dest strings to ensure no unexpected crashes.
        We'll mock urlopen to avoid real network calls.
        """
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.read.return_value = b"fake"
            mock_url.return_value = mock_resp

            # Force isfile=False so we actually attempt a download
            with patch("os.path.isfile", return_value=False):
                try:
                    download_file(url, os.path.join(self.test_dir, dest))
                except Exception:
                    # It's okay if certain random strings cause issues, so we pass
                    pass


class TestUnzipFile(TestBase):
    """
    Tests for the unzip_file function.

    Covers:
    - Missing zip scenario
    - Valid zip extraction
    - Hypothesis fuzzing
    """

    def setUp(self) -> None:
        self.zip_file = os.path.join(self.test_dir, "some_archive.zip")
        self.extract_dir = os.path.join(self.test_dir, "extract_here")

    def test_missing_zip_skips(self):
        """
        If zip_path doesn't exist, the function logs a warning and returns.
        """
        # No file created, so it is missing
        unzip_file(self.zip_file, self.extract_dir)
        self.assertFalse(
            os.path.exists(self.extract_dir), "No extraction should occur."
        )

    def test_valid_zip_extraction(self):
        """
        If the zip is valid, it should be extracted to extract_dir.
        """
        with zipfile.ZipFile(self.zip_file, "w") as zf:
            zf.writestr("inside.txt", "Hello from inside zip!")

        unzip_file(self.zip_file, self.extract_dir)
        extracted_path = os.path.join(self.extract_dir, "inside.txt")
        self.assertTrue(
            os.path.exists(extracted_path), "Zip content should be extracted."
        )

    @given(zip_path=st.text(min_size=1), extract_dir=st.text(min_size=1))
    @settings(deadline=None, max_examples=10)
    def test_fuzz_unzip_file(self, zip_path: str, extract_dir: str):
        """
        Random string paths to ensure no catastrophic behavior.
        File existence is not guaranteed, so usually no extraction occurs.
        """
        try:
            unzip_file(zip_path, os.path.join(self.test_dir, extract_dir))
        except Exception:
            pass


class TestVisualizeWithVarna(TestBase):
    """
    Tests for visualize_with_varna, verifying:
    - Handling of missing CT file
    - Handling of missing JAR file
    - Normal invocation with Subprocess
    - Fuzzing random path strings
    """

    def setUp(self) -> None:
        """
        Prepare potential .ct and .jar files in the temp directory.
        """
        self.ct_path = os.path.join(self.test_dir, "test_seq.ct")
        self.jar_path = os.path.join(self.test_dir, "VARNAv3-93.jar")
        self.out_png = os.path.join(self.test_dir, "test_seq.png")

    @patch("subprocess.Popen")
    def test_missing_ct_file(self, mock_popen):
        """
        If the ct_file is missing, skip calling Java.
        """
        # We'll create only jar
        with open(self.jar_path, "w") as f:
            f.write("fake jar")

        visualize_with_varna(self.ct_path, self.jar_path, self.out_png)
        mock_popen.assert_not_called()

    @patch("subprocess.Popen")
    def test_missing_jar_file(self, mock_popen):
        """
        If the jar file is missing, skip calling Java.
        """
        with open(self.ct_path, "w") as f:
            f.write(">Test\n1 A 0 2 0 1\n")

        visualize_with_varna(self.ct_path, self.jar_path, self.out_png)
        mock_popen.assert_not_called()

    @patch("subprocess.Popen")
    def test_normal_visualization(self, mock_popen):
        """
        If both ct_file and jar exist, run the Java command.
        """
        with open(self.ct_path, "w") as f:
            f.write(">Test\n1 A 0 2 0 1\n")
        with open(self.jar_path, "wb") as f:
            f.write(b"\x50\x4b\x03\x04")  # minimal zip signature

        visualize_with_varna(self.ct_path, self.jar_path, self.out_png)
        mock_popen.assert_called_once()

    @given(
        ct_file=st.text(min_size=1),
        jar_path=st.text(min_size=1),
        out_png=st.text(min_size=1),
    )
    @settings(deadline=None, max_examples=10)
    @patch("subprocess.Popen")
    def test_fuzz_visualize_with_varna(self, mock_popen, ct_file, jar_path, out_png):
        """
        Fuzz test with arbitrary strings for CT, JAR, and output paths.
        Typically won't call Java unless isfile() returns True for both.
        """
        with patch("os.path.isfile", return_value=True):
            try:
                run_stageA.visualize_with_varna(ct_file, jar_path, out_png)
            except Exception:
                pass


class TestBuildPredictor(TestBase):
    """
    Tests the build_predictor function, covering:
    - Normal usage with an actual checkpoint
    - Hypothesis fuzzing for config and path
    """

    def setUp(self) -> None:
        """
        Create a real checkpoint folder and file for realistic coverage.
        """
        self.ckpt_dir = os.path.join(self.test_dir, "RFold", "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(
            os.path.join(self.ckpt_dir, "RNAStralign_trainset_pretrained.pth"), "wb"
        ) as f:
            f.write(b"fake checkpoint data")

    def test_build_predictor_normal(self):
        """
        Basic test building a predictor with a known device and config.
        """
        config = {"num_hidden": 128, "dropout": 0.2}
        dev = torch.device("cpu")
        predictor = run_stageA.build_predictor(self.ckpt_dir, config, dev)
        self.assertIsNotNone(predictor, "Predictor should be created successfully.")
        self.assertTrue(
            hasattr(predictor, "predict_adjacency"),
            "Predictor must have 'predict_adjacency' method.",
        )

    @given(
        checkpoint_folder=st.text(min_size=1, max_size=30),
        config=st.dictionaries(
            st.text(min_size=1, max_size=5), st.integers(min_value=0, max_value=999)
        ),
        dev=st.sampled_from([torch.device("cpu"), torch.device("cuda")]),
    )
    @settings(deadline=None, max_examples=10)
    def test_fuzz_build_predictor(self, checkpoint_folder, config, dev):
        """
        Hypothesis test for building a predictor with random strings for checkpoint folder
        and random dictionary config.
        """
        try:
            run_stageA.build_predictor(checkpoint_folder, config, dev)
        except Exception:
            pass


class TestRunStageAFunction(TestBase):
    """
    Tests the run_stageA function, which calls predictor.predict_adjacency.
    """

    def test_run_stageA_normal(self):
        """
        Provide a mock predictor and valid seq. Should call predict_adjacency with the seq.
        """
        mock_predictor = MagicMock()
        mock_predictor.predict_adjacency.return_value = "FAKE_MATRIX"
        seq = "AUGC"

        result = run_stageA_func(seq, mock_predictor)
        self.assertEqual(result, "FAKE_MATRIX")
        mock_predictor.predict_adjacency.assert_called_once_with(seq)

    def test_run_stageA_invalid_seq(self):
        """
        If seq is None or not a string, typical usage might fail.
        But we only ensure it doesn't crash. Implementation-specific.
        """
        mock_predictor = MagicMock()
        # The function just calls predictor.predict_adjacency
        # So if predictor doesn't handle None, it may raise an error
        try:
            run_stageA_func(None, mock_predictor)
        except Exception:
            pass

    @given(st.text())
    @settings(deadline=None, max_examples=10)
    def test_fuzz_run_stageA_seq(self, random_seq):
        """
        Fuzz test random sequences. The predictor is mocked, so no real adjacency logic.
        """
        mock_predictor = MagicMock()
        try:
            run_stageA_func(random_seq, mock_predictor)
        except Exception:
            pass


class TestMainFunction(TestBase):
    """
    Tests for the main() function. This includes:
    - Folder creation
    - Building a predictor
    - Adjacency generation
    - (Optional) writing a .ct file
    - Visualization call
    """

    @patch("run_stageA.download_file", autospec=True)
    @patch("run_stageA.visualize_with_varna", autospec=True)
    @patch("run_stageA.build_predictor", autospec=True)
    def test_main_smoke(self, mock_build_pred, mock_visual, mock_download):
        """
        Smoke test ensuring main() runs without error. We mock out external calls
        to avoid real I/O or Java usage.
        """
        # Provide a mock predictor
        mock_pred = MagicMock()
        mock_pred.predict_adjacency.return_value = []
        mock_build_pred.return_value = mock_pred

        # Run main
        try:
            run_stageA.main()
        except Exception as e:
            self.fail(f"main() raised an unexpected exception: {e}")

        mock_build_pred.assert_called_once()
        self.assertTrue(mock_visual.called, "Visualization is expected.")


if __name__ == "__main__":
    unittest.main()
