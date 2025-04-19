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
from omegaconf import DictConfig

from rna_predict.print_rna_pipeline_output import (
    print_tensor_example,
    setup_pipeline,
    _main_impl,  # Import _main_impl at the module level
)


class TestPrintTensorExample(unittest.TestCase):
    """Tests for the print_tensor_example function."""

    def setUp(self):
        """Set up test fixtures."""
        # Redirect stdout to capture print output
        self.stdout_backup = sys.stdout
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output

    def tearDown(self):
        """Clean up after tests."""
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
        """Test print_tensor_example with wide 2D tensor (more columns than max_items)."""
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
        """Test print_tensor_example with small 3D tensor (fewer items than max_items in dim 1)."""
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
        """Set up test fixtures."""
        # Get the DummyStageAPredictor from setup_pipeline
        from omegaconf import DictConfig
        dummy_cfg = DictConfig({"device": "cpu"})
        config, _ = setup_pipeline(dummy_cfg)
        self.predictor = config["stageA_predictor"]

    def test_predict_adjacency_empty_seq(self):
        """Test predict_adjacency with empty sequence."""
        adj = self.predictor.predict_adjacency("")
        self.assertEqual(adj.shape, (0, 0))

    def test_predict_adjacency_single_char(self):
        """Test predict_adjacency with single character sequence."""
        adj = self.predictor.predict_adjacency("A")
        self.assertEqual(adj.shape, (1, 1))
        self.assertAlmostEqual(adj[0, 0], 1.0, places=5)  # Diagonal should be 1.0

    def test_predict_adjacency_short_seq(self):
        """Test predict_adjacency with short sequence."""
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

    def _check_diagonal(self, adj, seq):
        """Check that the diagonal elements are 1.0."""
        for i in range(len(seq)):
            self.assertAlmostEqual(adj[i, i], 1.0, places=5)

    def _check_adjacency(self, adj, seq):
        """Check that adjacent positions are 0.8."""
        for i in range(len(seq) - 1):
            self.assertAlmostEqual(adj[i, i+1], 0.8, places=5)
            self.assertAlmostEqual(adj[i+1, i], 0.8, places=5)

    def _check_nonlocal(self, adj, seq):
        """Check non-local (end-to-end) interactions are 0.5."""
        self.assertAlmostEqual(adj[0, len(seq)-1], 0.5, places=5)
        self.assertAlmostEqual(adj[len(seq)-1, 0], 0.5, places=5)

    @given(seq=st.text(alphabet="AUGC", min_size=0, max_size=20))
    @settings(deadline=None)
    def test_predict_adjacency_hypothesis(self, seq):
        """Test predict_adjacency with random sequences using Hypothesis."""
        adj = self.predictor.predict_adjacency(seq)
        self.assertEqual(adj.shape, (len(seq), len(seq)))
        if len(seq) > 0:
            self._check_diagonal(adj, seq)
            if len(seq) > 1:
                self._check_adjacency(adj, seq)
                if len(seq) > 4:
                    self._check_nonlocal(adj, seq)


class TestSetupPipeline(unittest.TestCase):
    """Tests for the setup_pipeline function."""

    def test_setup_pipeline_basic(self):
        """Test setup_pipeline with basic configuration."""
        config, device = setup_pipeline(DictConfig({"device": "cpu"}))

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
    @patch("rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.ProtenixDiffusionManager")
    @patch("rna_predict.pipeline.stageB.torsion.torsion_bert_predictor.StageBTorsionBertPredictor")
    @patch("rna_predict.pipeline.stageB.pairwise.pairformer_wrapper.PairformerWrapper")
    def test_setup_pipeline_with_stageD(self, mock_pairformer, mock_torsionbert, mock_diffusion_manager):
        """Test setup_pipeline with Stage D available."""
        # Mock the ProtenixDiffusionManager
        mock_diffusion_manager.return_value = "mock_diffusion_manager"
        mock_torsionbert.return_value = object()  # Dummy torsion model
        mock_pairformer.return_value = object()   # Dummy pairformer model

        # Use updated Hydra config structure for Stage D and required dependencies
        config_dict = {
            "device": "cpu",
            "stageD": {},  # Top-level key to trigger Stage D logic if needed
            "model": {
                "stageB": {
                    "torsion_bert": {
                        "dummy": True,
                        "angle_mode": "degrees",
                        "num_angles": 7,
                        "max_length": 32,
                        "model_name_or_path": "dummy"
                    }
                },
                "pairformer": {
                    "dummy": True,
                    "c_s": 64,
                    "c_z": 32,
                    "max_length": 32
                },
                "stageD": {
                    "enabled": True,
                    "some_required_param": 1
                },
                "stageD_diffusion": {
                    "enabled": True,
                    "some_required_param": 1
                },
            },
        }
        config, _ = setup_pipeline(DictConfig(config_dict))

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

        # In the updated implementation, we don't call ProtenixDiffusionManager directly
        # Instead, we just set the diffusion_manager key in the config
        # So we don't need to check the call arguments

    def test_setup_pipeline_with_torsion_model_exception(self):
        """Test setup_pipeline when torsion model initialization raises an exception."""
        # We'll use a context manager to patch the StageBTorsionBertPredictor
        with patch("rna_predict.pipeline.stageB.torsion.torsion_bert_predictor.StageBTorsionBertPredictor") as mock_torsion_model:
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
                config, _ = setup_pipeline(DictConfig({"device": "cpu"}))

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

    @patch("rna_predict.print_rna_pipeline_output.run_full_pipeline")
    @patch("rna_predict.print_rna_pipeline_output.print_tensor_example")
    def test_main_success(self, mock_print_tensor, mock_run_pipeline):
        """Test main function with successful pipeline run.
        Note: setup_pipeline is not called in this code path, so we do not assert it here.
        """
        mock_results = {
            "key1": "value1",
            "key2": "value2",
        }
        mock_run_pipeline.return_value = mock_results
        stdout_backup = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            from hypothesis import given, strategies as st
            required_keys = {"device": "cpu", "model": {"stageD": {"diffusion": {}}}}
            extra_keys_strategy = st.dictionaries(
                st.text(min_size=1, max_size=10).filter(lambda k: k not in required_keys),
                st.integers() | st.text() | st.booleans(),
                min_size=0, max_size=2
            )
            @given(st.builds(lambda extras: {**required_keys, **extras}, extra_keys_strategy))
            def inner_test(cfg_dict):
                cfg = DictConfig(cfg_dict)
                _main_impl(cfg)
                last_call = mock_run_pipeline.call_args
                kwargs = last_call.kwargs
                self.assertEqual(kwargs["sequence"], "AUGCAUGG", "run_full_pipeline should be called with the correct sequence (unique error: CASCADE-SP-005)")
                self.assertEqual(kwargs["device"], "cpu", "run_full_pipeline should be called with the correct device (unique error: CASCADE-SP-006)")
                self.assertIn("device", kwargs["cfg"], "config should contain 'device' key (unique error: CASCADE-SP-007)")
                self.assertEqual(kwargs["cfg"]["device"], "cpu", "config 'device' key should be 'cpu' (unique error: CASCADE-SP-008)")
                self.assertIn("model", kwargs["cfg"], "config should contain 'model' key (unique error: CASCADE-SP-009)")
                self.assertEqual(mock_print_tensor.call_count, 2)
                output = captured_output.getvalue()
                self.assertIn("Running RNA prediction pipeline", output)
                self.assertIn("Stage D available:", output)
                self.assertIn("Pipeline Output with Examples", output)
                self.assertIn("Done.", output)
                mock_run_pipeline.reset_mock()
                mock_print_tensor.reset_mock()
            inner_test()
        finally:
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
            # Create a mock DictConfig for testing
            cfg = DictConfig({"model": {"stageD": {"diffusion": {}}}})

            # Call the main implementation function directly with the mock config
            # This bypasses the Hydra decorator
            _main_impl(cfg)

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
