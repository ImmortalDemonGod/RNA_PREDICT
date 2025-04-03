import unittest
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st
from torch import Tensor

# Import the class under test
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)


# Create a proper mock for the DiffusionModule class
class MockDiffusionModule(torch.nn.Module):
    """
    A proper mock for DiffusionModule that avoids the super() call issue.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Store kwargs for inspection in tests
        self.kwargs = kwargs
        # Set up any attributes that might be accessed
        self.initialization = kwargs.get("initialization", {})

    def to(self, device):
        self.device = device
        return self

    def forward(
        self,
        x_noisy,
        t_hat_noise_level,
        input_feature_dict,
        s_inputs,
        s_trunk,
        z_trunk,
        **kwargs,
    ):
        # Return a tensor with the expected output shape
        batch_size = x_noisy.shape[0] if len(x_noisy.shape) > 3 else 1
        n_sample = x_noisy.shape[1] if len(x_noisy.shape) > 3 else 1
        n_atom = x_noisy.shape[-2]
        # Instead of returning zeros directly, return a single tensor regardless of input structure
        return torch.zeros((batch_size, n_sample, n_atom, 3), device=x_noisy.device)


class TestProtenixDiffusionManagerInitialization(unittest.TestCase):
    """
    Tests the ProtenixDiffusionManager __init__ method to ensure it sets up
    the diffusion module properly, including handling default "initialization"
    config fields and device usage.
    """

    def setUp(self):
        # Set up the patch at the imported module level
        self.diffusion_module_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule",
            MockDiffusionModule,
        )
        self.mock_diffusion_module = self.diffusion_module_patcher.start()
        self.addCleanup(self.diffusion_module_patcher.stop)

    def test_init_adds_initialization_key_if_missing(self):
        """
        Verifies that if 'initialization' is not present in diffusion_config,
        ProtenixDiffusionManager adds an empty dict for it.
        """
        config = {}  # Missing "initialization"
        manager = ProtenixDiffusionManager(diffusion_config=config, device="cpu")

        # Check that initialization was added to the config
        self.assertIn(
            "initialization",
            manager.diffusion_module.kwargs,
            "Expected 'initialization' key to be injected into config",
        )
        self.assertEqual(
            manager.diffusion_module.kwargs["initialization"],
            {},
            "Expected an empty dict as the default 'initialization'",
        )
        self.assertEqual(str(manager.device), "cpu", "Device should be set to 'cpu'")

    def test_init_respects_given_device(self):
        """
        Checks that the manager's device is set to the one passed in.
        """
        manager = ProtenixDiffusionManager(
            diffusion_config={"initialization": {}}, device="cuda"
        )
        self.assertEqual(str(manager.device), "cuda", "Device should be set to 'cuda'")

    def test_init_with_filled_initialization(self):
        """
        Verifies that if 'initialization' is present in diffusion_config,
        it remains unchanged.
        """
        config = {"initialization": {"foo": "bar"}, "other_key": 123}
        manager = ProtenixDiffusionManager(diffusion_config=config, device="cpu")
        self.assertIn("initialization", manager.diffusion_module.kwargs)
        self.assertEqual(
            manager.diffusion_module.kwargs["initialization"], {"foo": "bar"}
        )
        # Verify other keys were also passed


class TestProtenixDiffusionManagerTrainDiffusionStep(unittest.TestCase):
    """
    Tests the train_diffusion_step method to ensure it properly handles
    noise sampling, calling the diffusion module, and returning correct outputs.
    """

    def setUp(self):
        # Patch the diffusion module at the import level
        self.diffusion_module_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule",
            MockDiffusionModule,
        )
        self.mock_diffusion_module = self.diffusion_module_patcher.start()
        self.addCleanup(self.diffusion_module_patcher.stop)

        # Patch the sample_diffusion_training function
        self.sample_diff_train_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion_training",
            autospec=True,
        )
        self.mock_sample_diff_train = self.sample_diff_train_patcher.start()
        self.addCleanup(self.sample_diff_train_patcher.stop)

        self.manager = ProtenixDiffusionManager(diffusion_config={}, device="cpu")

    @given(
        label_dict=st.dictionaries(
            keys=st.text(min_size=1, max_size=5), values=st.just(torch.zeros((1, 3)))
        ),
        input_feature_dict=st.dictionaries(
            keys=st.text(min_size=1, max_size=5), values=st.just(torch.ones((1, 3)))
        ),
        s_inputs=st.just(torch.randn((1, 3))),
        s_trunk=st.just(torch.randn((1, 3))),
        z_trunk=st.just(torch.randn((1, 3))),
        sampler_params=st.dictionaries(
            keys=st.text(), values=st.floats(min_value=0.1, max_value=10)
        ),
        N_sample=st.integers(min_value=1, max_value=3),
        diffusion_chunk_size=st.one_of(
            st.none(), st.integers(min_value=1, max_value=10)
        ),
    )
    @settings(max_examples=10)
    def test_train_diffusion_step_runs_without_error(
        self,
        label_dict: Dict[str, Tensor],
        input_feature_dict: Dict[str, Tensor],
        s_inputs: Tensor,
        s_trunk: Tensor,
        z_trunk: Tensor,
        sampler_params: Dict[str, float],
        N_sample: int,
        diffusion_chunk_size: Optional[int],
    ):
        """
        Uses Hypothesis to generate various inputs ensuring train_diffusion_step
        processes them without error and returns expected tensor outputs.
        """
        # Reset the mock for each test case to ensure call count is properly tracked
        self.mock_sample_diff_train.reset_mock()

        fake_x_gt_augment = torch.zeros((2, 2))
        fake_x_denoised = torch.ones((2, 2))
        fake_sigma = torch.tensor([0.5])
        self.mock_sample_diff_train.return_value = (
            fake_x_gt_augment,
            fake_x_denoised,
            fake_sigma,
        )

        x_gt_augment, x_denoised, sigma = self.manager.train_diffusion_step(
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            sampler_params=sampler_params,
            N_sample=N_sample,
            diffusion_chunk_size=diffusion_chunk_size,
        )

        self.assertTrue(torch.equal(x_gt_augment, fake_x_gt_augment))
        self.assertTrue(torch.equal(x_denoised, fake_x_denoised))
        self.assertTrue(torch.equal(sigma, fake_sigma))
        # Verify the mock was called exactly once
        self.mock_sample_diff_train.assert_called_once()


class TestProtenixDiffusionManagerMultiStepInference(unittest.TestCase):
    """
    Tests the multi_step_inference method to ensure it properly expands shapes,
    handles missing 's_trunk', and calls sample_diffusion as expected.
    """

    def setUp(self):
        # Patch the diffusion module at the import level
        self.diffusion_module_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule",
            MockDiffusionModule,
        )
        self.mock_diffusion_module = self.diffusion_module_patcher.start()
        self.addCleanup(self.diffusion_module_patcher.stop)

        # Patch the sample_diffusion function
        self.sample_diff_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion",
            autospec=True,
        )
        self.mock_sample_diffusion = self.sample_diff_patcher.start()
        self.addCleanup(self.sample_diff_patcher.stop)

        self.manager = ProtenixDiffusionManager(diffusion_config={}, device="cpu")

    def test_missing_s_trunk_raises_value_error(self):
        """
        Ensures a ValueError is raised if 's_trunk' is not provided in trunk_embeddings.
        """
        coords_init = torch.randn((1, 5, 3))
        trunk_embeddings = {}
        with self.assertRaises(ValueError):
            self.manager.multi_step_inference(
                coords_init=coords_init,
                trunk_embeddings=trunk_embeddings,
                inference_params={"N_sample": 1, "num_steps": 5},
            )

    def test_multi_step_inference_expands_for_multiple_samples(self):
        """
        Checks that multi_step_inference correctly expands tensor shapes when N_sample > 1
        and returns the expected coordinates.
        """
        coords_init = torch.randn((2, 10, 3))
        trunk_embeddings = {
            "s_trunk": torch.randn((2, 50, 32)),
            "s_inputs": torch.randn((2, 50, 16)),
            "pair": torch.randn((2, 50, 50, 8)),
        }
        fake_coords_final = torch.randn((2, 3, 10, 3))
        self.mock_sample_diffusion.return_value = fake_coords_final

        coords_final = self.manager.multi_step_inference(
            coords_init=coords_init,
            trunk_embeddings=trunk_embeddings,
            inference_params={"N_sample": 3, "num_steps": 4},
            debug_logging=False,
        )
        self.assertTrue(torch.equal(coords_final, fake_coords_final))
        self.mock_sample_diffusion.assert_called_once()


class TestProtenixDiffusionManagerCustomManualLoop(unittest.TestCase):
    """
    Tests the custom_manual_loop method to ensure it applies noise correctly and
    returns the expected (x_noisy, x_denoised) results.
    """

    def setUp(self):
        # Patch the diffusion module at the import level
        self.diffusion_module_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule",
            MockDiffusionModule,
        )
        self.mock_diffusion_module = self.diffusion_module_patcher.start()
        self.addCleanup(self.diffusion_module_patcher.stop)

        self.manager = ProtenixDiffusionManager(diffusion_config={}, device="cpu")
        # Create a specific mock for the forward method to track calls
        self.manager.diffusion_module.forward = MagicMock()
        # Set up a specific return value that matches the expected output shape
        self.manager.diffusion_module.forward.return_value = torch.zeros_like(
            torch.empty(3, 3)
        )

    @given(
        x_gt=st.lists(
            st.floats(
                min_value=-1000, max_value=1000, allow_infinity=False, allow_nan=False
            ),
            min_size=9,
            max_size=9,
        ).map(lambda arr: torch.tensor(arr).reshape(3, 3)),
        sigma=st.floats(
            min_value=0.01, max_value=10.0, allow_infinity=False, allow_nan=False
        ),
    )
    @settings(max_examples=5)
    @example(x_gt=torch.zeros((3, 3)), sigma=1.0)
    def test_custom_manual_loop(self, x_gt: Tensor, sigma: float):
        """
        Uses Hypothesis to verify custom_manual_loop processes numeric inputs for x_gt and sigma correctly.
        """
        # Reset mock to ensure we're only counting calls in this test
        self.manager.diffusion_module.forward.reset_mock()

        # Create a return value with the same shape as x_gt
        fake_x_denoised = torch.zeros_like(x_gt)
        self.manager.diffusion_module.forward.return_value = fake_x_denoised

        trunk_embeddings = {
            "s_trunk": torch.randn((1, 3)),
            "s_inputs": torch.randn((1, 3)),
            "pair": torch.randn((1, 3)),
        }

        x_noisy, x_denoised = self.manager.custom_manual_loop(
            x_gt=x_gt, trunk_embeddings=trunk_embeddings, sigma=sigma
        )
        self.assertEqual(x_noisy.shape, x_gt.shape)
        self.assertEqual(x_denoised.shape, x_gt.shape)
        # Verify the mock was called exactly once
        self.manager.diffusion_module.forward.assert_called_once()


class TestProtenixDiffusionManagerRoundTrip(unittest.TestCase):
    """
    Demonstrates a round-trip test covering train_diffusion_step, multi_step_inference,
    and custom_manual_loop, ensuring end-to-end behavior.
    """

    def setUp(self):
        # Patch the diffusion module at the import level
        self.diffusion_module_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule",
            MockDiffusionModule,
        )
        self.mock_diffusion_module = self.diffusion_module_patcher.start()
        self.addCleanup(self.diffusion_module_patcher.stop)

        self.train_patch = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion_training",
            autospec=True,
        )
        self.mock_train_diff = self.train_patch.start()
        self.addCleanup(self.train_patch.stop)

        self.inference_patch = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion",
            autospec=True,
        )
        self.mock_inference_diff = self.inference_patch.start()
        self.addCleanup(self.inference_patch.stop)

        self.manager = ProtenixDiffusionManager(diffusion_config={}, device="cpu")
        # Set up a mock return value for the diffusion module
        self.manager.diffusion_module.forward = MagicMock()

    def test_round_trip_pipeline(self):
        """
        Verifies a round-trip workflow:
        1) train_diffusion_step returns a tuple of tensors.
        2) multi_step_inference returns coordinates.
        3) custom_manual_loop applies noise and returns expected outputs.
        """
        # Setup mock returns
        x_gt_augment = torch.randn((2, 2))
        x_denoised_train = torch.randn((2, 2))
        train_sigma = torch.tensor([0.5])
        self.mock_train_diff.return_value = (
            x_gt_augment,
            x_denoised_train,
            train_sigma,
        )

        coords_final = torch.randn((1, 4, 3))
        self.mock_inference_diff.return_value = coords_final

        final_denoised = torch.randn((1, 4, 3))
        self.manager.diffusion_module.forward.return_value = final_denoised

        # Execute train_diffusion_step
        train_out = self.manager.train_diffusion_step(
            label_dict={"test": torch.zeros((1, 3))},
            input_feature_dict={"features": torch.ones((1, 3))},
            s_inputs=torch.randn((1, 3)),
            s_trunk=torch.randn((1, 3)),
            z_trunk=torch.randn((1, 3)),
            sampler_params={},
            N_sample=1,
            diffusion_chunk_size=None,
        )
        self.assertEqual(
            len(train_out), 3, "Expected a tuple of length 3 from train_diffusion_step"
        )

        # Execute multi_step_inference
        trunk_embeddings = {
            "s_trunk": torch.randn((1, 5, 32)),
            "s_inputs": torch.randn((1, 5, 16)),
            "pair": torch.randn((1, 5, 5, 8)),
        }
        coords_out = self.manager.multi_step_inference(
            coords_init=torch.randn((1, 5, 3)),
            trunk_embeddings=trunk_embeddings,
            inference_params={"N_sample": 1, "num_steps": 2},
        )
        self.assertTrue(torch.equal(coords_out, coords_final))

        # Execute custom_manual_loop
        x_noisy, x_denoised_manual = self.manager.custom_manual_loop(
            x_gt=coords_final, trunk_embeddings=trunk_embeddings, sigma=0.3
        )
        self.assertEqual(x_noisy.shape, coords_final.shape)
        self.assertEqual(x_denoised_manual.shape, coords_final.shape)

        self.mock_train_diff.assert_called_once()
        self.mock_inference_diff.assert_called_once()
        self.manager.diffusion_module.forward.assert_called()


if __name__ == "__main__":
    # Run the tests when the file is executed directly.
    unittest.main()
