import unittest
from unittest.mock import patch, MagicMock
import torch
from torch import Tensor
from hypothesis import given, strategies as st, settings, example
from hypothesis import HealthCheck
from hypothesis.strategies import dictionaries, text

# We import the function under test.
# Adjust this import as needed if your module/package layout differs.
from rna_predict.pipeline.stageD.run_stageD import run_stageD_diffusion

class TestRunStageDDiffusion(unittest.TestCase):
    """
    Tests for the run_stageD_diffusion function, which orchestrates
    final diffusion-based refinement in Stage D.
    """

    def setUp(self) -> None:
        """
        Common setup for all tests, creating default test inputs:
         - partial_coords: a small tensor of shape (batch=1, atoms=5, 3D coords)
         - trunk_embeddings: dictionary with typical embeddings
         - diffusion_config: minimal valid config
         - device: typically 'cpu'
        """
        self.partial_coords = torch.zeros(1, 5, 3)
        self.trunk_embeddings = {
            "s_trunk": torch.zeros(1, 5, 384),  # shape [B, N_token, 384]
            "pair": torch.zeros(1, 5, 5, 32),   # shape [B, N_token, N_token, c_z]
        }
        self.diffusion_config = {
            "c_atom": 128,
            "c_s": 384,
            "c_z": 32,
            "c_token": 832,
            "transformer": {"n_blocks": 4, "n_heads": 16},
        }
        self.device = "cpu"

    @patch("run_stageD.load_rna_data_and_features")
    def test_inference_mode(self, mock_load_rna):
        """
        Test that run_stageD_diffusion works in inference mode and
        returns a tensor of expected shape, given typical embeddings.
        We mock load_rna_data_and_features to avoid file I/O.
        """
        # Mock the return to emulate typical shapes from real data
        # atom_feature_dict, token_feature_dict
        mock_load_rna.return_value = (
            {"atom_to_token_idx": torch.zeros(1, 5, dtype=torch.long)},
            {
                "restype": torch.zeros(1, 5, dtype=torch.long),
                "profile": torch.zeros(1, 5, 10),
            },
        )

        coords_out = run_stageD_diffusion(
            partial_coords=self.partial_coords,
            trunk_embeddings=self.trunk_embeddings,
            diffusion_config=self.diffusion_config,
            mode="inference",
            device=self.device,
        )

        self.assertIsInstance(coords_out, torch.Tensor)
        # For N_sample=1 inside the code, we expect shape [B, N_sample, N_atom, 3].
        # B=1, N_atom=5 => shape should be (1, 1, 5, 3) in many configurations,
        # but may vary based on the code's expansions.
        self.assertTrue(coords_out.dim() >= 3, "Output must have at least 3 dims.")
        # Basic sanity checks: coordinate dimension should be last
        self.assertEqual(coords_out.shape[-1], 3, "Last dimension must be 3 for coordinates.")

    @patch("run_stageD.load_rna_data_and_features")
    def test_train_mode(self, mock_load_rna):
        """
        Test that run_stageD_diffusion works in train mode and returns
        (x_denoised, loss, sigma) with correct shapes/types.
        """
        # Mock typical shapes from load_rna_data_and_features
        mock_load_rna.return_value = (
            {"atom_to_token_idx": torch.zeros(1, 5, dtype=torch.long)},
            {
                "restype": torch.zeros(1, 5, dtype=torch.long),
                "profile": torch.zeros(1, 5, 10),
            },
        )

        result = run_stageD_diffusion(
            partial_coords=self.partial_coords,
            trunk_embeddings=self.trunk_embeddings,
            diffusion_config=self.diffusion_config,
            mode="train",
            device=self.device,
        )
        # Train mode returns a tuple: (x_denoised, loss, sigma).
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        x_denoised, loss, sigma = result
        self.assertIsInstance(x_denoised, torch.Tensor)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(sigma, torch.Tensor)

        # Basic shape checks
        self.assertTrue(x_denoised.dim() >= 3, "x_denoised must have at least 3 dims.")
        self.assertEqual(x_denoised.shape[-1], 3, "Last dim must be 3 for coordinates.")
        self.assertTrue(loss.dim() == 0, "Loss should be a scalar tensor.")
        self.assertTrue(sigma.dim() == 0, "Sigma should be a scalar tensor.")

    def test_invalid_mode_raises(self):
        """
        Test that running with an unsupported mode raises ValueError.
        """
        with self.assertRaises(ValueError):
            run_stageD_diffusion(
                partial_coords=self.partial_coords,
                trunk_embeddings=self.trunk_embeddings,
                diffusion_config=self.diffusion_config,
                mode="unsupported_mode",
                device=self.device,
            )

    @patch("run_stageD.load_rna_data_and_features")
    def test_deletion_mean_handling(self, mock_load_rna):
        """
        Test that code path adjusting deletion_mean shape is covered.
        This ensures lines that handle 'deletion_mean' dimension mismatch are tested.
        """
        # The presence of 'deletion_mean' with shape mismatch triggers the logic.
        mock_load_rna.return_value = (
            {"atom_to_token_idx": torch.zeros(1, 5, dtype=torch.long)},
            {
                "restype": torch.zeros(1, 5, dtype=torch.long),
                "profile": torch.zeros(1, 5, 10),
                "deletion_mean": torch.zeros(1, 5)  # shape [1, 5], will become [1,5,1]
            },
        )

        coords_out = run_stageD_diffusion(
            partial_coords=self.partial_coords,
            trunk_embeddings=self.trunk_embeddings,
            diffusion_config=self.diffusion_config,
            mode="inference",
            device=self.device,
        )
        self.assertIsInstance(coords_out, torch.Tensor)
        # We only check the function didn't crash and returned a tensor
        self.assertTrue(coords_out.shape[-1] == 3)

    @patch("run_stageD.load_rna_data_and_features")
    def test_round_trip_diffusion(self, mock_load_rna):
        """
        Example 'round-trip'-style test: call inference mode twice,
        passing the output coordinates back in as input. Checks consistency of shape.
        """
        mock_load_rna.return_value = (
            {"atom_to_token_idx": torch.zeros(1, 5, dtype=torch.long)},
            {
                "restype": torch.zeros(1, 5, dtype=torch.long),
                "profile": torch.zeros(1, 5, 10),
            },
        )

        # First inference pass
        coords_out_1 = run_stageD_diffusion(
            partial_coords=self.partial_coords,
            trunk_embeddings=self.trunk_embeddings,
            diffusion_config=self.diffusion_config,
            mode="inference",
            device=self.device,
        )

        self.assertIsInstance(coords_out_1, torch.Tensor)
        self.assertEqual(coords_out_1.shape[-1], 3)

        # Second inference pass, reusing the output of the first
        coords_out_2 = run_stageD_diffusion(
            partial_coords=coords_out_1.squeeze(1),  # shape alignment
            trunk_embeddings=self.trunk_embeddings,
            diffusion_config=self.diffusion_config,
            mode="inference",
            device=self.device,
        )
        self.assertIsInstance(coords_out_2, torch.Tensor)
        self.assertEqual(coords_out_2.shape[-1], 3)
        # Check final shapes for consistency
        self.assertEqual(coords_out_1.shape[-2], coords_out_2.shape[-2])

class TestRunStageDDiffusionFuzz(unittest.TestCase):
    """
    Property-based (fuzz) tests for run_stageD_diffusion. Ensures that
    arbitrary inputs do not crash the system. We do not assert correctness
    of the outputs, only that the function handles them gracefully.
    """

    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much]
    )
    @given(
        partial_coords=st.one_of(
            # Either empty or a small random shape. Using just st.builds(Tensor) can lead
            # to very large or invalid shapes that might cause crashes in PyTorch. We refine:
            st.builds(
                torch.randn, st.integers(min_value=1, max_value=4), st.integers(min_value=1, max_value=4)
            ),
            # Or just a zero-dim corner case
            st.just(torch.zeros(0, 0))
        ),
        trunk_embeddings=dictionaries(keys=text(), values=st.builds(Tensor), max_size=5),
        diffusion_config=dictionaries(keys=text(), values=st.integers(), max_size=5),
        mode=st.sampled_from(["inference", "train", "unknown_mode"]),
        device=st.sampled_from(["cpu"])  # can add "cuda" if available
    )
    @example(
        partial_coords=torch.randn(1, 3, 3),
        trunk_embeddings={"s_trunk": torch.randn(1, 3, 384)},
        diffusion_config={"c_atom": 128},
        mode="inference",
        device="cpu"
    )
    def test_fuzz_run_stageD_diffusion(
        self,
        partial_coords: Tensor,
        trunk_embeddings: dict,
        diffusion_config: dict,
        mode: str,
        device: str
    ) -> None:
        """
        Hypothesis-based fuzz test that calls run_stageD_diffusion with random
        arguments to see if it raises unexpected exceptions. Some combos may
        raise ValueError if mode is invalid or if shapes are truly impossible.
        """
        try:
            output = run_stageD_diffusion(
                partial_coords=partial_coords,
                trunk_embeddings=trunk_embeddings,
                diffusion_config=diffusion_config,
                mode=mode,
                device=device
            )
            # If the mode was recognized, we expect either a Tensor or a tuple.
            # It's valid if an exception wasn't raised.
            if mode not in ["inference", "train"]:
                # We expect a ValueError for invalid mode
                self.fail("Expected ValueError for invalid mode but got no exception.")
            else:
                if mode == "inference":
                    self.assertIsInstance(output, torch.Tensor)
                elif mode == "train":
                    self.assertIsInstance(output, tuple)
                    self.assertEqual(len(output), 3)
        except ValueError as e:
            # If we got here with an invalid mode, that's expected
            if mode in ["inference", "train"]:
                # If it's a valid mode, raising ValueError is unexpected
                self.fail(f"Valid mode '{mode}' unexpectedly raised ValueError: {e}")
        except Exception:
            # We catch broad exceptions to avoid test crashing from random shapes
            # but let unittest report them with a fail so they can be debugged.
            self.fail("run_stageD_diffusion raised an unexpected exception!")