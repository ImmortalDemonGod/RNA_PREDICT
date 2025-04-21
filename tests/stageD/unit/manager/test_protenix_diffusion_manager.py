import unittest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf, DictConfig # Import OmegaConf types

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# Import the class under test
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
# Import generator class needed for type checking in tests
from rna_predict.pipeline.stageD.diffusion.generator import (
    TrainingNoiseSampler,
)


# --- Helper function to create Stage D test configs ---
# (Copied from integration test, could be moved to a conftest.py later)
def create_stage_d_test_config(stage_overrides=None, model_overrides=None, noise_overrides=None) -> DictConfig:
    """Creates a base DictConfig for Stage D tests, allowing overrides."""
    if stage_overrides is None:
        stage_overrides = {}
    if model_overrides is None:
        model_overrides = {}
    if noise_overrides is None:
        noise_overrides = {}

    # Base structure matching stageD.diffusion.yaml and its expected nested groups
    base_config = {
        "stageD": {
            "diffusion": {
                "mode": "inference", # Default mode
                "device": "cpu",
                # Removed C-specific params like do_ring_closure
                "angle_representation": "cartesian", # Keep if relevant to diffusion input processing
                "use_metadata": False, # Keep if relevant
                "sigma_data": 16.0, # Diffusion param
                "gamma0": 0.8,
                "gamma_min": 1.0,
                "noise_scale_lambda": 1.003,
                "step_scale_eta": 1.5,
                "diffusion_chunk_size": None,
                "attn_chunk_size": None,
                "inplace_safe": False,
                "debug_logging": False,
                "training": { "batch_size": 1 },
                "inference": {
                    "num_steps": 2, # Keep low for testing
                    "temperature": 1.0,
                    "sampling": {"num_samples": 1, "seed": None, "use_deterministic": False}
                },
                "memory": {
                    "apply_memory_preprocess": False, # Default to False for most tests
                    "memory_preprocess_max_len": 25
                },
                # Sampler params needed for train_diffusion_step test
                "sampler": {
                    "p_mean": -1.2,
                    "p_std": 1.5,
                    "N_sample": 1
                }
            }
        },
        "diffusion_model": { # Corresponds to DiffusionModule args
             "c_atom": 128,
             "c_atompair": 16,
             "c_token": 384, # Adjusted based on dummy data below
             "c_s": 384,
             "c_z": 32, # Adjusted based on dummy data below
             "c_s_inputs": 384, # Match c_s for simplicity
             "c_noise_embedding": 128,
             "atom_encoder": {"n_blocks": 1, "n_heads": 1, "n_queries": 4, "n_keys": 8}, # Minimal
             "transformer": {"n_blocks": 1, "n_heads": 1}, # Minimal
             "atom_decoder": {"n_blocks": 1, "n_heads": 1, "n_queries": 4, "n_keys": 8}, # Minimal
             "sigma_data": 16.0, # Also needed by DiffusionModule
             "initialization": None # Explicitly None if not providing specifics
        },
        "noise_schedule": { # Example for linear schedule
             "schedule_type": "linear",
             "beta_start": 0.0001,
             "beta_end": 0.02
        }
    }
    cfg = OmegaConf.create(base_config)

    # Apply overrides selectively using OmegaConf.merge
    override_cfg = OmegaConf.create({})
    if stage_overrides:
        OmegaConf.update(override_cfg, "stageD.diffusion", stage_overrides, merge=True)
    if model_overrides:
        OmegaConf.update(override_cfg, "diffusion_model", model_overrides, merge=True)
    if noise_overrides:
         OmegaConf.update(override_cfg, "noise_schedule", noise_overrides, merge=True)

    # Use OmegaConf.merge which handles DictConfig correctly
    cfg = OmegaConf.merge(cfg, override_cfg)

    if not isinstance(cfg, DictConfig):
         # This check might be overly strict if merge sometimes returns other OmegaConf types
         # but we expect DictConfig based on input structure.
         raise TypeError(f"Merged config is not DictConfig: {type(cfg)}")
    return cfg


# Create a proper mock for the DiffusionModule class that can be instantiated
class MockDiffusionModule(torch.nn.Module):
    """A mock for DiffusionModule for isolated manager testing."""
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs # Store init args
        self._device = torch.device("cpu") # Default device

    def to(self, device):
        self._device = device # Record device
        return self

    def forward(self, *args, **kwargs):
        # Basic forward that returns zeros matching x_noisy shape if possible
        x_noisy = kwargs.get('x_noisy', args[0] if args else None)
        if x_noisy is not None:
            return torch.zeros_like(x_noisy, device=self._device)
        # Fallback return if x_noisy not found
        return torch.zeros(1, device=self._device)


class TestProtenixDiffusionManagerInitialization(unittest.TestCase):
    """Tests the ProtenixDiffusionManager __init__ method with Hydra config."""
    def setUp(self):
        # Patch the DiffusionModule class where it's imported by the manager
        self.diffusion_module_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule",
            autospec=True, # Use autospec to ensure mock matches signature
            # Return our MockDiffusionModule instance when the class is called
            return_value=MockDiffusionModule()
        )
        self.MockDiffusionModuleClass = self.diffusion_module_patcher.start()
        self.addCleanup(self.diffusion_module_patcher.stop)

    @given(
        c_atom=st.integers(min_value=16, max_value=256),
        c_z=st.integers(min_value=16, max_value=128),
        n_blocks=st.integers(min_value=1, max_value=4),
        n_heads=st.integers(min_value=1, max_value=8),
        device=st.just("cpu")  # Keep device as CPU for testing
    )
    @settings(max_examples=10, deadline=None)  # Limit examples for faster tests
    def test_init_with_basic_config(self, c_atom, c_z, n_blocks, n_heads, device):
        """Property-based: Verify manager initializes with various valid configs."""
        # Reset mock for each test case
        self.MockDiffusionModuleClass.reset_mock()

        # Create test config with Hypothesis-generated values
        test_cfg = create_stage_d_test_config(
            stage_overrides={
                "device": device,
                "c_atom": c_atom,
                "c_z": c_z,
                "transformer": {"n_blocks": n_blocks, "n_heads": n_heads}
            }
        )

        # Initialize manager with test config
        manager = ProtenixDiffusionManager(cfg=test_cfg)

        # Check the mocked DiffusionModule CLASS was called once to create instance
        self.MockDiffusionModuleClass.assert_called_once()
        _, kwargs_passed = self.MockDiffusionModuleClass.call_args

        # Verify specific config values were passed correctly
        self.assertEqual(kwargs_passed.get('c_atom'), c_atom,
                         f"[ERR-DIFFMAN-001] c_atom mismatch: expected {c_atom}, got {kwargs_passed.get('c_atom')}")
        self.assertEqual(kwargs_passed.get('c_z'), c_z,
                         f"[ERR-DIFFMAN-002] c_z mismatch: expected {c_z}, got {kwargs_passed.get('c_z')}")
        self.assertIn('transformer', kwargs_passed,
                      "[ERR-DIFFMAN-003] transformer config missing from DiffusionModule args")
        self.assertEqual(kwargs_passed['transformer'].get('n_blocks'), n_blocks,
                         f"[ERR-DIFFMAN-004] transformer.n_blocks mismatch: expected {n_blocks}, got {kwargs_passed['transformer'].get('n_blocks')}")
        self.assertEqual(kwargs_passed['transformer'].get('n_heads'), n_heads,
                         f"[ERR-DIFFMAN-005] transformer.n_heads mismatch: expected {n_heads}, got {kwargs_passed['transformer'].get('n_heads')}")

        # Verify device attribute on manager
        self.assertEqual(str(manager.device), device,
                         f"[ERR-DIFFMAN-006] device mismatch: expected {device}, got {str(manager.device)}")


    @given(
        missing_section=st.sampled_from(["stageD", "diffusion"]),
        c_s=st.integers(min_value=1, max_value=512),
        device=st.sampled_from(["cpu", "cuda"])
    )
    @settings(max_examples=4, deadline=None)  # Limit examples for faster tests
    def test_init_missing_required_groups_raises(self, missing_section, c_s, device):
        """Property-based: Check that errors are raised if essential config groups are missing."""
        # Reset mock for each test case
        self.MockDiffusionModuleClass.reset_mock()

        if missing_section == "stageD":
            # Missing stageD section
            cfg_missing = OmegaConf.create({"diffusion_model": {"c_s": c_s}})
            expected_error = "Config missing required 'stageD' group"
        else:  # missing_section == "diffusion"
            # Missing diffusion in stageD
            cfg_missing = OmegaConf.create({"stageD": {"device": device}})
            expected_error = "Config missing required 'diffusion' group in stageD"

        # Use the actual error message raised by the manager code
        with self.assertRaisesRegex(ValueError, expected_error,
                                    msg=f"[ERR-DIFFMAN-007] Expected error '{expected_error}' not raised"):
             ProtenixDiffusionManager(cfg=cfg_missing)


class TestProtenixDiffusionManagerTrainDiffusionStep(unittest.TestCase):
    """Tests the train_diffusion_step method with Hydra config."""
    def setUp(self):
        self.diffusion_module_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule",
            autospec=True, return_value=MockDiffusionModule()
        )
        self.MockDiffusionModuleClass = self.diffusion_module_patcher.start()
        self.addCleanup(self.diffusion_module_patcher.stop)

        self.sample_diff_train_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion_training",
            autospec=True,
        )
        self.mock_sample_diff_train = self.sample_diff_train_patcher.start()
        self.addCleanup(self.sample_diff_train_patcher.stop)

        # Create manager with default test config
        self.test_cfg = create_stage_d_test_config()
        self.manager = ProtenixDiffusionManager(cfg=self.test_cfg)

    @given(
        label_dict=st.dictionaries(keys=st.text(min_size=1, max_size=5), values=st.just(torch.zeros((1, 3)))),
        input_feature_dict=st.dictionaries(keys=st.text(min_size=1, max_size=5), values=st.just(torch.ones((1, 3)))),
        s_inputs=st.just(torch.randn((1, 5, 64))), # Match config dims
        s_trunk=st.just(torch.randn((1, 5, 64))),
        z_trunk=st.just(torch.randn((1, 5, 5, 32))),
        N_sample_override=st.one_of(st.none(), st.integers(min_value=1, max_value=3)),
        chunk_override=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
    )
    @settings(max_examples=10, deadline=None) # Disable deadline
    def test_train_diffusion_step_calls_sampler(
        self, label_dict, input_feature_dict, s_inputs, s_trunk, z_trunk,
        N_sample_override, chunk_override
    ):
        """
        Uses Hypothesis to ensure train_diffusion_step calls sample_diffusion_training
        with correct args derived from config.
        """
        self.mock_sample_diff_train.reset_mock()

        # Prepare config overrides for this run
        stage_cfg_overrides = {}
        # Note: N_sample override now goes under sampler
        if N_sample_override is not None:
            # Need to initialize sampler dict if overriding nested value
            stage_cfg_overrides['sampler'] = {'N_sample': N_sample_override}
        if chunk_override is not None:
             stage_cfg_overrides['diffusion_chunk_size'] = chunk_override

        current_cfg = create_stage_d_test_config(stage_overrides=stage_cfg_overrides)
        # Re-init manager with potentially overridden config for this test case
        current_manager = ProtenixDiffusionManager(cfg=current_cfg)


        # Set a deterministic return value for the mocked function
        fake_x_gt_augment = torch.zeros_like(s_inputs) # Example shape
        fake_x_denoised = torch.ones_like(s_inputs)
        fake_sigma = torch.tensor([0.5])
        self.mock_sample_diff_train.return_value = (fake_x_gt_augment, fake_x_denoised, fake_sigma)

        # Call the method under test (no extra args needed now)
        x_gt_out, x_den_out, sigma_out = current_manager.train_diffusion_step(
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
        )

        # Verify return values match mock
        self.assertTrue(torch.equal(x_gt_out, fake_x_gt_augment),
                       "[ERR-DIFFMAN-019] x_gt_augment output doesn't match expected value")
        self.assertTrue(torch.equal(x_den_out, fake_x_denoised),
                       "[ERR-DIFFMAN-020] x_denoised output doesn't match expected value")
        self.assertTrue(torch.equal(sigma_out, fake_sigma),
                       "[ERR-DIFFMAN-021] sigma output doesn't match expected value")

        # Verify sample_diffusion_training was called once
        self.mock_sample_diff_train.assert_called_once()

        # Verify args passed to sample_diffusion_training
        _, call_kwargs = self.mock_sample_diff_train.call_args
        self.assertIsInstance(call_kwargs.get('noise_sampler'), TrainingNoiseSampler,
                             "[ERR-DIFFMAN-022] noise_sampler is not a TrainingNoiseSampler")
        self.assertIs(call_kwargs.get('denoise_net'), current_manager.diffusion_module,
                     "[ERR-DIFFMAN-023] Wrong diffusion module passed to sample_diffusion_training")

        # Check config derived values were passed correctly
        expected_N_sample = N_sample_override if N_sample_override is not None else current_cfg.stageD.diffusion.sampler.N_sample
        expected_chunk_size = chunk_override if chunk_override is not None else current_cfg.stageD.diffusion.get('diffusion_chunk_size') # Use get for top level

        self.assertEqual(call_kwargs.get('N_sample'), expected_N_sample,
                        f"[ERR-DIFFMAN-024] N_sample mismatch: expected {expected_N_sample}, got {call_kwargs.get('N_sample')}")
        self.assertEqual(call_kwargs.get('diffusion_chunk_size'), expected_chunk_size,
                        f"[ERR-DIFFMAN-025] chunk_size mismatch: expected {expected_chunk_size}, got {call_kwargs.get('diffusion_chunk_size')}")

        # Check that input tensors were passed correctly
        self.assertTrue(torch.equal(call_kwargs.get('s_trunk'), s_trunk),
                       "[ERR-DIFFMAN-026] s_trunk tensor not passed correctly")
        self.assertTrue(torch.equal(call_kwargs.get('s_inputs'), s_inputs),
                       "[ERR-DIFFMAN-027] s_inputs tensor not passed correctly")
        self.assertTrue(torch.equal(call_kwargs.get('z_trunk'), z_trunk),
                       "[ERR-DIFFMAN-028] z_trunk tensor not passed correctly")


class TestProtenixDiffusionManagerMultiStepInference(unittest.TestCase):
    """Tests the multi_step_inference method with Hydra config."""

    def setUp(self):
        self.diffusion_module_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule",
            autospec=True, return_value=MockDiffusionModule()
        )
        self.MockDiffusionModuleClass = self.diffusion_module_patcher.start()
        self.addCleanup(self.diffusion_module_patcher.stop)

        self.sample_diff_patcher = patch(
            "rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion",
            autospec=True,
        )
        self.mock_sample_diffusion = self.sample_diff_patcher.start()
        self.addCleanup(self.sample_diff_patcher.stop)

        # Base config for tests
        self.test_cfg = create_stage_d_test_config()
        self.manager = ProtenixDiffusionManager(cfg=self.test_cfg)


    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=3, max_value=10),
        pair_dim=st.integers(min_value=4, max_value=16),
        include_other_keys=st.booleans()
    )
    @settings(max_examples=10, deadline=None)  # Limit examples for faster tests
    def test_missing_s_trunk_raises_value_error(self, batch_size, seq_len, pair_dim, include_other_keys):
        """Property-based: Ensures a ValueError is raised if 's_trunk' is missing."""
        # Create random coordinates tensor
        coords_init = torch.randn((batch_size, seq_len, 3))

        # Create trunk embeddings without s_trunk
        trunk_embeddings = {"pair": torch.randn(batch_size, seq_len, seq_len, pair_dim)}

        # Optionally add other keys to test robustness
        if include_other_keys:
            trunk_embeddings["s_inputs"] = torch.randn(batch_size, seq_len, 32)
            trunk_embeddings["atom_to_token_idx"] = torch.randint(0, seq_len, (batch_size, seq_len))

        # Verify that ValueError is raised with the expected message
        with self.assertRaisesRegex(ValueError, "requires a valid 's_trunk'",
                                    msg="[ERR-DIFFMAN-008] Expected error about missing s_trunk not raised"):
             self.manager.multi_step_inference(
                 coords_init=coords_init,
                 trunk_embeddings=trunk_embeddings
             )

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=5, max_value=15),
        feature_dim=st.integers(min_value=16, max_value=64),
        pair_dim=st.integers(min_value=8, max_value=32),
        num_samples=st.integers(min_value=1, max_value=5),
        include_override_features=st.booleans()
    )
    @settings(max_examples=10, deadline=None)  # Limit examples for faster tests
    def test_multi_step_inference_calls_sampler(self, batch_size, seq_len, feature_dim, pair_dim,
                                               num_samples, include_override_features):
        """
        Property-based: Checks that multi_step_inference calls sample_diffusion with args derived from config.
        """
        # Reset mock for each test case
        self.mock_sample_diffusion.reset_mock()

        # Create test input tensors with Hypothesis-generated dimensions
        coords_init = torch.randn((batch_size, seq_len, 3))
        trunk_embeddings = {
            "s_trunk": torch.randn((batch_size, seq_len, feature_dim)),
            "s_inputs": torch.randn((batch_size, seq_len, feature_dim)),
            "pair": torch.randn((batch_size, seq_len, seq_len, pair_dim)),
        }

        # Create optional override features
        override_input_features = None
        if include_override_features:
            override_input_features = {
                "atom_to_token_idx": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
                "extra_feature": torch.randn(batch_size, seq_len, 8)
            }

        # Expected return value matching num_samples
        # Create the tensor that sample_diffusion would return
        fake_sample_diffusion_output = torch.randn((batch_size, num_samples, seq_len, 3))
        self.mock_sample_diffusion.return_value = fake_sample_diffusion_output

        # Create the expected final output after potential squeezing
        if num_samples == 1:
            # For N_sample=1, the manager squeezes dimension 1
            fake_coords_final = fake_sample_diffusion_output.squeeze(1)
        else:
            fake_coords_final = fake_sample_diffusion_output

        # Create config with num_samples in the correct nested location
        cfg_multi_sample = create_stage_d_test_config(
            stage_overrides={"inference": {"sampling": {"num_samples": num_samples}}}
        )
        manager_multi = ProtenixDiffusionManager(cfg=cfg_multi_sample) # Use specific manager

        # Call the method under test
        coords_final = manager_multi.multi_step_inference(
            coords_init=coords_init,
            trunk_embeddings=trunk_embeddings,
            override_input_features=override_input_features
        )

        # Check return value and mock call
        self.assertTrue(torch.equal(coords_final, fake_coords_final),
                       f"[ERR-DIFFMAN-009] Return value doesn't match expected tensor")
        self.mock_sample_diffusion.assert_called_once()

        # Check args passed to sample_diffusion
        _, call_kwargs = self.mock_sample_diffusion.call_args
        self.assertIs(call_kwargs.get('denoise_net'), manager_multi.diffusion_module,
                     "[ERR-DIFFMAN-010] Wrong diffusion module passed to sample_diffusion")

        # Verify N_sample read correctly from nested config
        self.assertEqual(call_kwargs.get('N_sample'), num_samples,
                        f"[ERR-DIFFMAN-011] N_sample mismatch: expected {num_samples}, got {call_kwargs.get('N_sample')}")

        # Check noise_schedule is a tensor
        self.assertIsInstance(call_kwargs.get('noise_schedule'), torch.Tensor,
                             "[ERR-DIFFMAN-012] noise_schedule is not a tensor")

        # Check that embeddings were passed correctly
        self.assertTrue(torch.equal(call_kwargs.get('s_trunk'), trunk_embeddings['s_trunk'].to(manager_multi.device)),
                       "[ERR-DIFFMAN-013] s_trunk tensor not passed correctly")


# TestProtenixDiffusionManagerCustomManualLoop requires minimal changes if any
class TestProtenixDiffusionManagerCustomManualLoop(unittest.TestCase):
    """Tests the custom_manual_loop method (init needs cfg)."""
    def setUp(self):
        # Create a mock module that is callable
        self.mock_module = MagicMock()

        # Configure the mock to return a tensor when called
        def side_effect(**kwargs):
            x_noisy = kwargs.get('x_noisy')
            if x_noisy is not None:
                return torch.zeros_like(x_noisy)
            return torch.zeros(1)

        self.mock_module.side_effect = side_effect

        # Create a test config with small dimensions
        self.test_cfg = create_stage_d_test_config(model_overrides={"c_s":10, "c_z":10, "c_s_inputs":10})

        # Create the manager
        self.manager = ProtenixDiffusionManager(cfg=self.test_cfg)

        # Replace the diffusion_module with our mock
        self.manager.diffusion_module = self.mock_module


    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=3, max_value=15),
        feature_dim_multiplier=st.integers(min_value=1, max_value=4),
        sigma=st.floats(min_value=0.1, max_value=5.0)
    )
    @settings(max_examples=10, deadline=None)  # Limit examples for faster tests
    def test_custom_manual_loop(self, batch_size, seq_len, feature_dim_multiplier, sigma):
        """
        Property-based: Verify custom_manual_loop processes inputs and calls module's forward method.
        """
        # Create test data with Hypothesis-generated dimensions
        x_gt = torch.randn(batch_size, seq_len, 3)

        # Reset the mock
        self.mock_module.reset_mock()

        # Calculate feature dimensions based on multiplier for variety
        c_s = self.test_cfg.diffusion_model.c_s * feature_dim_multiplier
        c_s_inputs = self.test_cfg.diffusion_model.c_s_inputs * feature_dim_multiplier
        c_z = self.test_cfg.diffusion_model.c_z * feature_dim_multiplier

        # Dummy embeddings matching dims potentially used by model
        trunk_embeddings = {
            "s_trunk": torch.randn((batch_size, seq_len, c_s)),
            "s_inputs": torch.randn((batch_size, seq_len, c_s_inputs)),
            "pair": torch.randn((batch_size, seq_len, seq_len, c_z)),
        }

        # Configure mock to return appropriate tensor
        def mock_forward(**kwargs):
            x_noisy = kwargs.get('x_noisy')
            if x_noisy is not None:
                return torch.zeros_like(x_noisy)
            return torch.zeros(1)
        self.mock_module.side_effect = mock_forward

        # Call the method under test
        x_noisy, _ = self.manager.custom_manual_loop(
            x_gt=x_gt, trunk_embeddings=trunk_embeddings, sigma=sigma
        )

        # Verify the diffusion module was called once
        self.mock_module.assert_called_once()

        # Check shapes
        self.assertEqual(x_noisy.shape, x_gt.shape,
                        f"[ERR-DIFFMAN-014] x_noisy shape {x_noisy.shape} doesn't match x_gt shape {x_gt.shape}")

        # Check noise level passed to the module
        _, call_kwargs = self.mock_module.call_args
        noise_level_arg = call_kwargs.get('t_hat_noise_level')
        self.assertIsInstance(noise_level_arg, torch.Tensor,
                             "[ERR-DIFFMAN-015] Noise level is not a tensor")

        # Compare the float values directly
        self.assertAlmostEqual(noise_level_arg.item(), sigma, places=5,
                              msg=f"[ERR-DIFFMAN-016] Noise level {noise_level_arg.item()} differs from input sigma {sigma}")

        # Check that embeddings were passed correctly
        # Map the parameter names to the keys in trunk_embeddings
        param_to_key = {
            "s_trunk": "s_trunk",
            "s_inputs": "s_inputs",
            "z_trunk": "pair"  # In the implementation, pair is passed as z_trunk
        }

        for param_name, key in param_to_key.items():
            passed_tensor = call_kwargs.get(param_name)
            self.assertIsNotNone(passed_tensor,
                                f"[ERR-DIFFMAN-017] {param_name} not passed to diffusion module")
            self.assertEqual(passed_tensor.shape, trunk_embeddings[key].shape,
                           f"[ERR-DIFFMAN-018] {param_name} shape mismatch: expected {trunk_embeddings[key].shape}, got {passed_tensor.shape}")


if __name__ == "__main__":
    unittest.main()
