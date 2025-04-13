"""
Comprehensive tests for print_rna_pipeline_output.py.

This module provides thorough testing for the print_rna_pipeline_output.py module,
which handles output formatting and printing for the RNA prediction pipeline.
"""

import io
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from hypothesis import given, settings, strategies as st

from rna_predict.print_rna_pipeline_output import (
    main,
    print_tensor_example,
    setup_pipeline,
)


class TestPrintTensorExample(unittest.TestCase):
    """Tests for the print_tensor_example function."""

    def setUp(self):
        """
        Set up test fixtures by capturing printed output.
        
        This method saves the current sys.stdout in self.stdout_backup and replaces
        sys.stdout with an io.StringIO instance (self.captured_output) to enable
        inspection of output during tests.
        """
        # Redirect stdout to capture print output
        self.stdout_backup = sys.stdout
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output

    def tearDown(self):
        """
        Restores the original standard output stream.
        
        This method resets sys.stdout to the value stored in self.stdout_backup,
        ensuring that modifications made during testing do not affect subsequent tests.
        """
        # Restore stdout
        sys.stdout = self.stdout_backup

    def test_print_tensor_example_none(self):
        """Test print_tensor_example with None tensor."""
        print_tensor_example("test_none", None)
        output = self.captured_output.getvalue()
        self.assertIn("test_none: None", output)

    def test_print_tensor_example_1d_numpy(self):
        """Test print_tensor_example with 1D numpy array."""
        tensor_1d = np.array([1, 2, 3, 4, 5, 6])
        print_tensor_example("test_1d", tensor_1d)
        output = self.captured_output.getvalue()
        self.assertIn("test_1d: shape=(6,)", output)
        self.assertIn("Example values:", output)
        self.assertIn("[1 2 3 4 5", output)  # First 5 values

    def test_print_tensor_example_1d_torch(self):
        """Test print_tensor_example with 1D torch tensor."""
        tensor_1d = torch.tensor([1, 2, 3, 4, 5, 6])
        print_tensor_example("test_1d_torch", tensor_1d)
        output = self.captured_output.getvalue()
        self.assertIn("test_1d_torch: shape=(6,)", output)
        self.assertIn("Example values:", output)
        self.assertIn("[1 2 3 4 5", output)  # First 5 values

    def test_print_tensor_example_1d_short(self):
        """Test print_tensor_example with short 1D tensor (less than max_items)."""
        tensor_1d_short = np.array([1, 2, 3])
        print_tensor_example("test_1d_short", tensor_1d_short)
        output = self.captured_output.getvalue()
        self.assertIn("test_1d_short: shape=(3,)", output)
        self.assertIn("Example values:", output)
        self.assertIn("[1 2 3]", output)  # All values

    def test_print_tensor_example_2d_numpy(self):
        """Test print_tensor_example with 2D numpy array."""
        tensor_2d = np.array([[1, 2, 3, 4, 6], [7, 8, 9, 10, 11]])
        print_tensor_example("test_2d", tensor_2d)
        output = self.captured_output.getvalue()
        self.assertIn("test_2d: shape=(2, 5)", output)
        self.assertIn("Example values:", output)
        self.assertIn("[1 2 3 4 6]", output)  # First row
        self.assertIn("7  8  9 10 11", output)  # Second row

    def test_print_tensor_example_2d_torch(self):
        """Test print_tensor_example with 2D torch tensor."""
        tensor_2d = torch.tensor([[1, 2, 3, 4, 6], [7, 8, 9, 10, 11]])
        print_tensor_example("test_2d_torch", tensor_2d)
        output = self.captured_output.getvalue()
        self.assertIn("test_2d_torch: shape=(2, 5)", output)
        self.assertIn("Example values:", output)
        self.assertIn("[1 2 3 4 6]", output)  # First row
        self.assertIn("7  8  9 10 11", output)  # Second row

    def test_print_tensor_example_2d_wide(self):
        """
        Test print_tensor_example with a wide 2D tensor.
        
        This test uses a 2D NumPy array with more columns than the display limit. It asserts
        that the printed output shows the correct tensor shape and truncates each row's example
        values to the first five elements followed by an ellipsis.
        """
        tensor_2d_wide = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
        print_tensor_example("test_2d_wide", tensor_2d_wide)
        output = self.captured_output.getvalue()
        self.assertIn("test_2d_wide: shape=(2, 8)", output)
        self.assertIn("Example values:", output)
        self.assertIn("[1 2 3 4 5, ...]", output)  # First 5 columns of first row with truncation
        self.assertIn("[ 9 10 11 12 13, ...]", output)  # First 5 columns of second row with truncation

    def test_print_tensor_example_2d_tall(self):
        """Test print_tensor_example with tall 2D tensor (more rows than max_items)."""
        tensor_2d_tall = np.array([[i, i+1, i+2] for i in range(10)])
        print_tensor_example("test_2d_tall", tensor_2d_tall)
        output = self.captured_output.getvalue()
        self.assertIn("test_2d_tall: shape=(10, 3)", output)
        self.assertIn("Example values:", output)
        self.assertIn("[0 1 2]", output)  # First row
        self.assertIn("[4 5 6]", output)  # Fifth row
        self.assertIn(" ...]", output)  # Truncation indicator

    def test_print_tensor_example_3d_small(self):
        """
        Test that print_tensor_example prints the correct output for a small 3D tensor.
        
        A 3D tensor with shape (2, 3, 4) is used to verify that when the number of items in the
        second dimension is below the truncation threshold, the printed output includes the
        tensor's name and shape, as well as the expected "Example values:" and "Data:" sections.
        """
        tensor_3d_small = np.zeros((2, 3, 4))
        print_tensor_example("test_3d_small", tensor_3d_small)
        output = self.captured_output.getvalue()
        self.assertIn("test_3d_small: shape=(2, 3, 4)", output)
        self.assertIn("Example values:", output)
        self.assertIn("Data:", output)

    def test_print_tensor_example_3d_large(self):
        """Test print_tensor_example with large 3D tensor (more items than max_items in dim 1)."""
        tensor_3d_large = np.zeros((2, 10, 4))
        print_tensor_example("test_3d_large", tensor_3d_large)
        output = self.captured_output.getvalue()
        self.assertIn("test_3d_large: shape=(2, 10, 4)", output)
        self.assertIn("Example values:", output)
        self.assertIn("First slice:", output)

    @given(
        tensor=st.one_of(
            # 1D numpy arrays
            st.builds(
                np.array,
                st.lists(
                    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    min_size=1,
                    max_size=10,
                ),
            ),
            # 1D torch tensors
            st.builds(
                torch.tensor,
                st.lists(
                    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    min_size=1,
                    max_size=10,
                ),
            ),
        )
    )
    @settings(deadline=None)
    def test_print_tensor_example_hypothesis(self, tensor):
        """Test print_tensor_example with random tensors using Hypothesis."""
        print_tensor_example("test_hypothesis", tensor)
        output = self.captured_output.getvalue()
        self.assertIn("test_hypothesis: shape=", output)
        self.assertIn("Example values:", output)


class TestStageAPredictor(unittest.TestCase):
    """Tests for the DummyStageAPredictor class defined in setup_pipeline."""

    def setUp(self):
        """
        Initializes the stage A predictor for test cases.
        
        Retrieves the pipeline configuration using setup_pipeline() and assigns the
        stageA_predictor from the configuration to an instance attribute for use in
        subsequent tests.
        """
        # Get the DummyStageAPredictor from setup_pipeline
        config, _ = setup_pipeline()
        self.predictor = config["stageA_predictor"]

    def test_predict_adjacency_empty_seq(self):
        """
        Verifies that an empty input sequence produces an empty adjacency matrix.
        
        Ensures that calling predict_adjacency with an empty string returns a matrix with shape (0, 0).
        """
        adj = self.predictor.predict_adjacency("")
        self.assertEqual(adj.shape, (0, 0))

    def test_predict_adjacency_single_char(self):
        """Test predict_adjacency with single character sequence."""
        adj = self.predictor.predict_adjacency("A")
        self.assertEqual(adj.shape, (1, 1))
        self.assertAlmostEqual(adj[0, 0], 1.0, places=5)  # Diagonal should be 1.0

    def test_predict_adjacency_short_seq(self):
        """
        Verifies that predict_adjacency returns the correct matrix for a short RNA sequence.
        
        For the sequence "AU", the expected 2x2 matrix has 1.0 on the diagonals and 0.8 
        on the off-diagonals.
        """
        adj = self.predictor.predict_adjacency("AU")
        self.assertEqual(adj.shape, (2, 2))
        self.assertAlmostEqual(adj[0, 0], 1.0, places=5)  # Diagonal should be 1.0
        self.assertAlmostEqual(adj[1, 1], 1.0, places=5)  # Diagonal should be 1.0
        self.assertAlmostEqual(adj[0, 1], 0.8, places=5)  # Adjacent positions should be 0.8
        self.assertAlmostEqual(adj[1, 0], 0.8, places=5)  # Adjacent positions should be 0.8

    def test_predict_adjacency_long_seq(self):
        """Test predict_adjacency with long sequence."""
        seq = "AUGCAUGC"
        adj = self.predictor.predict_adjacency(seq)
        self.assertEqual(adj.shape, (len(seq), len(seq)))

        # Check diagonal
        for i in range(len(seq)):
            self.assertAlmostEqual(adj[i, i], 1.0, places=5)

        # Check adjacent positions
        for i in range(len(seq) - 1):
            self.assertAlmostEqual(adj[i, i+1], 0.8, places=5)
            self.assertAlmostEqual(adj[i+1, i], 0.8, places=5)

        # Check non-local interactions
        self.assertAlmostEqual(adj[0, len(seq)-1], 0.5, places=5)
        self.assertAlmostEqual(adj[len(seq)-1, 0], 0.5, places=5)

    @given(seq=st.text(alphabet="AUGC", min_size=0, max_size=20))
    @settings(deadline=None)
    def test_predict_adjacency_hypothesis(self, seq):
        """
        Tests that the predict_adjacency method returns a valid adjacency matrix.
        
        This test uses random sequence inputs generated by Hypothesis to verify that:
        - The output matrix is square with dimensions equal to the sequence length.
        - Diagonal elements are approximately 1.0.
        - Immediate neighbors are approximately 0.8 when the sequence has at least two elements.
        - For sequences longer than four characters, the first and last matrix entries are approximately 0.5.
        """
        adj = self.predictor.predict_adjacency(seq)

        # Check shape
        self.assertEqual(adj.shape, (len(seq), len(seq)))

        if len(seq) > 0:
            # Check diagonal
            for i in range(len(seq)):
                self.assertAlmostEqual(adj[i, i], 1.0, places=5)

            # Check adjacent positions
            if len(seq) > 1:
                for i in range(len(seq) - 1):
                    self.assertAlmostEqual(adj[i, i+1], 0.8, places=5)
                    self.assertAlmostEqual(adj[i+1, i], 0.8, places=5)

                # Check non-local interactions
                if len(seq) > 4:
                    self.assertAlmostEqual(adj[0, len(seq)-1], 0.5, places=5)
                    self.assertAlmostEqual(adj[len(seq)-1, 0], 0.5, places=5)


class TestSetupPipeline(unittest.TestCase):
    """Tests for the setup_pipeline function."""

    def test_setup_pipeline_basic(self):
        """Test setup_pipeline with basic configuration."""
        config, device = setup_pipeline()

        # Check device
        self.assertEqual(device, "cpu")

        # Check required keys in config
        required_keys = [
            "stageA_predictor",
            "torsion_bert_model",
            "pairformer_model",
            "merger",
            "enable_stageC",
            "merge_latent",
            "init_z_from_adjacency",
        ]
        for key in required_keys:
            self.assertIn(key, config)

        # Check values
        self.assertIsInstance(config["stageA_predictor"], object)  # We can't check the exact class type since it's defined inside setup_pipeline
        self.assertTrue(config["enable_stageC"])
        self.assertTrue(config["merge_latent"])
        self.assertTrue(config["init_z_from_adjacency"])

    @patch("rna_predict.print_rna_pipeline_output.STAGE_D_AVAILABLE", True)
    @patch("rna_predict.print_rna_pipeline_output.ProtenixDiffusionManager")
    def test_setup_pipeline_with_stageD(self, mock_diffusion_manager):
        """
        Verify that setup_pipeline configures Stage D when available.
        
        This test mocks the diffusion manager to simulate Stage D support and confirms that the
        pipeline configuration contains the expected keys ("diffusion_manager", "stageD_config",
        "run_stageD") with correct values. It also ensures that the diffusion manager is invoked once
        with the device set to "cpu" and that the configuration dictionary is correctly passed as an argument.
        """
        # Mock the ProtenixDiffusionManager
        mock_diffusion_manager.return_value = "mock_diffusion_manager"

        config, _ = setup_pipeline()

        # Check Stage D related keys in config
        stageD_keys = [
            "diffusion_manager",
            "stageD_config",
            "run_stageD",
        ]
        for key in stageD_keys:
            self.assertIn(key, config)

        # Check values
        self.assertEqual(config["diffusion_manager"], "mock_diffusion_manager")
        self.assertTrue(config["run_stageD"])

        # Check that ProtenixDiffusionManager was called with the correct arguments
        mock_diffusion_manager.assert_called_once()
        args, kwargs = mock_diffusion_manager.call_args
        self.assertEqual(kwargs["device"], "cpu")
        self.assertIsInstance(args[0], dict)  # First arg should be config dict

    def test_setup_pipeline_with_torsion_model_exception(self):
        """
        Tests that setup_pipeline gracefully handles torsion model init failure.
        
        Simulates a torsion model initialization exception and verifies that
        setup_pipeline prints a warning and recovers by creating a fallback dummy
        torsion model with expected output dimensions and prediction behavior.
        """
        # We'll use a context manager to patch the StageBTorsionBertPredictor
        with patch("rna_predict.print_rna_pipeline_output.StageBTorsionBertPredictor") as mock_torsion_model:
            # First call raises an exception, second call returns a mock object
            mock_instance = MagicMock()
            mock_instance.output_dim = 14
            mock_instance.predict.return_value = torch.randn((4, 14))  # For a sequence of length 4

            # Configure the side effect to raise an exception on first call, then return the mock instance
            mock_torsion_model.side_effect = [Exception("Test exception"), mock_instance]

            # Redirect stdout to capture print output
            stdout_backup = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output

            try:
                config, _ = setup_pipeline()

                # Check that warning was printed
                output = captured_output.getvalue()
                self.assertIn("[Warning]", output)
                self.assertIn("Could not load", output)

                # Check that a dummy model was created
                self.assertEqual(config["torsion_bert_model"].output_dim, 14)

                # Test the mock predict method
                seq = "AUGC"
                result = config["torsion_bert_model"].predict(seq)
                self.assertEqual(result.shape, (len(seq), 14))
            finally:
                # Restore stdout
                sys.stdout = stdout_backup


class TestMain(unittest.TestCase):
    """Tests for the main function."""

    @patch("rna_predict.print_rna_pipeline_output.setup_pipeline")
    @patch("rna_predict.print_rna_pipeline_output.run_full_pipeline")
    @patch("rna_predict.print_rna_pipeline_output.print_tensor_example")
    def test_main_success(self, mock_print_tensor, mock_run_pipeline, mock_setup):
        """Test main function with successful pipeline run."""
        # Mock setup_pipeline
        mock_setup.return_value = ({"mock_config": True}, "cpu")

        # Mock run_full_pipeline
        mock_results = {
            "key1": "value1",
            "key2": "value2",
        }
        mock_run_pipeline.return_value = mock_results

        # Redirect stdout to capture print output
        stdout_backup = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            main()

            # Check that setup_pipeline was called
            mock_setup.assert_called_once()

            # Check that run_full_pipeline was called with the correct arguments
            mock_run_pipeline.assert_called_once_with("AUGCAUGG", {"mock_config": True}, device="cpu")

            # Check that print_tensor_example was called for each result
            self.assertEqual(mock_print_tensor.call_count, 2)

            # Check output
            output = captured_output.getvalue()
            self.assertIn("Running RNA prediction pipeline", output)
            self.assertIn("Stage D available:", output)
            self.assertIn("Pipeline Output with Examples", output)
            self.assertIn("Done.", output)
        finally:
            # Restore stdout
            sys.stdout = stdout_backup

    @patch("rna_predict.print_rna_pipeline_output.setup_pipeline")
    @patch("rna_predict.print_rna_pipeline_output.run_full_pipeline")
    def test_main_exception(self, mock_run_pipeline, mock_setup):
        """Test main function when pipeline run raises an exception."""
        # Mock setup_pipeline
        mock_setup.return_value = ({"mock_config": True}, "cpu")

        # Mock run_full_pipeline to raise an exception
        mock_run_pipeline.side_effect = Exception("Test exception")

        # Redirect stdout to capture print output
        stdout_backup = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            main()

            # Check output
            output = captured_output.getvalue()
            self.assertIn("Error running pipeline:", output)
            self.assertIn("Test exception", output)
            self.assertIn("Done.", output)
        finally:
            # Restore stdout
            sys.stdout = stdout_backup


if __name__ == "__main__":
    unittest.main()
