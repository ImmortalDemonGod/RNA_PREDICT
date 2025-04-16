"""
Comprehensive tests for protenix_diffusion_manager.py to improve test coverage.
"""

import torch
import unittest
from unittest.mock import patch, MagicMock
import warnings
from omegaconf import DictConfig, OmegaConf

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager


class TestProtenixDiffusionManagerComprehensive(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create mock tensors and embeddings
        self.batch_size = 1
        self.seq_len = 50
        self.feature_dim = 64

        # Create atom-level coordinates
        self.coords = torch.randn(self.batch_size, self.seq_len, 3)

        # Create trunk embeddings
        self.trunk_embeddings = {
            "s_trunk": torch.randn(self.batch_size, self.seq_len, self.feature_dim),
            "s_inputs": torch.randn(self.batch_size, self.seq_len, self.feature_dim * 2),
            "pair": torch.randn(self.batch_size, self.seq_len, self.seq_len, self.feature_dim // 2),
            "atom_to_token_idx": torch.randint(0, self.seq_len, (self.batch_size, self.seq_len)).long()
        }

        # Create a mock Hydra config
        self.cfg = OmegaConf.create({
            "stageD_diffusion": {
                "device": "cpu",
                "sigma_data": 16.0,
                "sampler": {
                    "p_mean": -1.2,
                    "p_std": 1.5,
                    "N_sample": 1
                },
                "inference": {
                    "num_steps": 2,  # Use a small value for testing
                    "sampling": {
                        "num_samples": 1
                    }
                }
            },
            "diffusion_model": {
                "c_atom": 128,
                "c_s": 64,
                "c_z": 32,
                "c_s_inputs": 128,
                "transformer": {
                    "n_blocks": 2,
                    "n_heads": 4
                },
                "atom_encoder": {},
                "atom_decoder": {},
                "sigma_data": 16.0
            },
            "noise_schedule": {
                "schedule_type": "linear"
            }
        })

        # Create a dictionary config for backward compatibility testing
        self.diffusion_config_dict = {
            "sigma_data": 16.0,
            "c_atom": 128,
            "c_s": 64,
            "c_z": 32,
            "c_s_inputs": 128,
            "transformer": {
                "n_blocks": 2,
                "n_heads": 4
            },
            "atom_encoder": {},
            "atom_decoder": {},
            "sampler": {
                "p_mean": -1.2,
                "p_std": 1.5,
                "N_sample": 1
            },
            "inference": {
                "num_steps": 2,
                "sampling": {
                    "num_samples": 1
                }
            },
            "noise_schedule": {
                "schedule_type": "linear"
            }
        }

        # Create label dict for training
        self.label_dict = {
            "coordinate": self.coords,
            "coordinate_mask": torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        }

        # Create input feature dict for training
        self.input_feature_dict = {
            "atom_mask": torch.ones(self.batch_size, self.seq_len),
            "residue_index": torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1)
        }

    def test_init_with_hydra_config(self):
        """Test initialization with Hydra config."""
        # Use a more targeted patch to avoid actual DiffusionModule initialization
        with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule') as mock_diffusion_module:
            # Setup mock
            mock_instance = MagicMock()
            mock_diffusion_module.return_value = mock_instance
            mock_instance.to.return_value = mock_instance

            # Initialize manager with Hydra config
            manager = ProtenixDiffusionManager(cfg=self.cfg)

            # Check that DiffusionModule was initialized with the correct arguments
            mock_diffusion_module.assert_called_once()

            # Check that the device was set correctly
            self.assertEqual(manager.device, torch.device("cpu"))

    def test_init_with_dict_config_backward_compatibility(self):
        """Test initialization with dictionary config (backward compatibility)."""
        # Use a more targeted patch to avoid actual DiffusionModule initialization
        with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule') as mock_diffusion_module:
            # Setup mock
            mock_instance = MagicMock()
            mock_diffusion_module.return_value = mock_instance
            mock_instance.to.return_value = mock_instance

            # Force warnings to be shown
            warnings.simplefilter('always', DeprecationWarning)

            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                # Initialize manager with dictionary config
                manager = ProtenixDiffusionManager(diffusion_config=self.diffusion_config_dict, device="cpu")

                # Check that a deprecation warning was issued
                self.assertTrue(len(w) > 0, "No warnings were captured")
                found_warning = False
                for warning in w:
                    if issubclass(warning.category, DeprecationWarning) and "Initializing ProtenixDiffusionManager with diffusion_config dict is deprecated" in str(warning.message):
                        found_warning = True
                        break
                self.assertTrue(found_warning, "Expected deprecation warning not found")

            # Check that DiffusionModule was initialized
            mock_diffusion_module.assert_called_once()

            # Check that the device was set correctly
            self.assertEqual(manager.device, torch.device("cpu"))

    def test_init_with_no_config(self):
        """Test initialization with no config."""
        # Check that ValueError is raised when no config is provided
        with self.assertRaises(ValueError) as context:
            ProtenixDiffusionManager()

        self.assertIn("Either cfg (Hydra config) or diffusion_config (dict) must be provided",
                      str(context.exception))

    def test_init_with_invalid_config_type(self):
        """Test initialization with invalid config type."""
        # Check that TypeError is raised when cfg is not an OmegaConf DictConfig
        with self.assertRaises(TypeError) as context:
            ProtenixDiffusionManager(cfg={"not": "a DictConfig"})

        self.assertIn("cfg argument must be an OmegaConf DictConfig", str(context.exception))

    def test_init_with_missing_stageD_diffusion_group(self):
        """Test initialization with missing stageD_diffusion group."""
        # Create config without stageD_diffusion group
        invalid_cfg = OmegaConf.create({
            "some_other_group": {}
        })

        # Check that ValueError is raised
        with self.assertRaises(ValueError) as context:
            ProtenixDiffusionManager(cfg=invalid_cfg)

        self.assertIn("Config missing required 'stageD_diffusion' group", str(context.exception))

    def test_train_diffusion_step(self):
        """Test train_diffusion_step method."""
        # Use more targeted patches to avoid actual model execution
        with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule') as mock_diffusion_module:
            with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion_training') as mock_sample_diffusion_training:
                # Setup mocks
                mock_instance = MagicMock()
                mock_diffusion_module.return_value = mock_instance
                mock_instance.to.return_value = mock_instance

                # Setup the mock to return a tuple (coords, loss)
                mock_instance.return_value = (torch.randn(self.batch_size, self.seq_len, 3), None)

                # Setup return value for sample_diffusion_training
                mock_sample_diffusion_training.return_value = (
                    torch.randn(self.batch_size, self.seq_len, 3),  # x_gt_augment
                    torch.randn(self.batch_size, self.seq_len, 3),  # x_denoised
                    torch.tensor(1.0)  # sigma
                )

                # Initialize manager
                manager = ProtenixDiffusionManager(cfg=self.cfg)

                # Call train_diffusion_step
                x_gt_augment, x_denoised, sigma = manager.train_diffusion_step(
                    label_dict=self.label_dict,
                    input_feature_dict=self.input_feature_dict,
                    s_inputs=self.trunk_embeddings["s_inputs"],
                    s_trunk=self.trunk_embeddings["s_trunk"],
                    z_trunk=self.trunk_embeddings["pair"]
                )

                # Check that sample_diffusion_training was called with the correct arguments
                mock_sample_diffusion_training.assert_called_once()

                # Check return values
                self.assertIsInstance(x_gt_augment, torch.Tensor)
                self.assertIsInstance(x_denoised, torch.Tensor)
                self.assertIsInstance(sigma, torch.Tensor)

    def test_train_diffusion_step_without_hydra_config(self):
        """Test train_diffusion_step method without Hydra config."""
        # Use a more targeted patch to avoid actual model execution
        with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule') as mock_diffusion_module:
            with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.OmegaConf.is_config', return_value=False) as mock_is_config:
                # Setup mocks
                mock_instance = MagicMock()
                mock_diffusion_module.return_value = mock_instance
                mock_instance.to.return_value = mock_instance

                # Initialize manager with dictionary config
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Ignore deprecation warning
                    manager = ProtenixDiffusionManager(diffusion_config=self.diffusion_config_dict, device="cpu")

                # Check that RuntimeError is raised when train_diffusion_step is called without Hydra config
                with self.assertRaises(RuntimeError) as context:
                    manager.train_diffusion_step(
                        label_dict=self.label_dict,
                        input_feature_dict=self.input_feature_dict,
                        s_inputs=self.trunk_embeddings["s_inputs"],
                        s_trunk=self.trunk_embeddings["s_trunk"],
                        z_trunk=self.trunk_embeddings["pair"]
                    )

            self.assertIn("train_diffusion_step requires Hydra config", str(context.exception))

    def test_multi_step_inference(self):
        """Test multi_step_inference method."""
        # Use more targeted patches to avoid actual model execution
        with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule') as mock_diffusion_module:
            with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion') as mock_sample_diffusion:
                # Setup mocks
                mock_instance = MagicMock()
                mock_diffusion_module.return_value = mock_instance
                mock_instance.to.return_value = mock_instance

                # Setup the mock to return a tuple (coords, loss)
                mock_instance.return_value = (torch.randn(self.batch_size, self.seq_len, 3), None)

                # Setup return value for sample_diffusion
                mock_sample_diffusion.return_value = torch.randn(self.batch_size, self.seq_len, 3)

                # Initialize manager
                manager = ProtenixDiffusionManager(cfg=self.cfg)

                # Call multi_step_inference
                refined_coords = manager.multi_step_inference(
                    coords_init=self.coords,
                    trunk_embeddings=self.trunk_embeddings
                )

                # Check that sample_diffusion was called
                mock_sample_diffusion.assert_called_once()

                # Check return value
                self.assertIsInstance(refined_coords, torch.Tensor)
                self.assertEqual(refined_coords.shape, (self.batch_size, self.seq_len, 3))

    def test_multi_step_inference_without_hydra_config(self):
        """Test multi_step_inference method without Hydra config."""
        # Use a more targeted patch to avoid actual model execution
        with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule') as mock_diffusion_module:
            # Setup mock
            mock_instance = MagicMock()
            mock_diffusion_module.return_value = mock_instance
            mock_instance.to.return_value = mock_instance

            # Setup the mock to return a tuple (coords, loss)
            mock_instance.return_value = (torch.randn(self.batch_size, self.seq_len, 3), None)

            # Initialize manager with dictionary config
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore deprecation warning
                manager = ProtenixDiffusionManager(diffusion_config=self.diffusion_config_dict, device="cpu")

            # Patch sample_diffusion to avoid the actual call
            with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion') as mock_sample_diffusion:
                # Setup return value for sample_diffusion
                mock_sample_diffusion.return_value = torch.randn(self.batch_size, self.seq_len, 3)

                # Call multi_step_inference
                refined_coords = manager.multi_step_inference(
                    coords_init=self.coords,
                    trunk_embeddings=self.trunk_embeddings
                )

                # Check that sample_diffusion was called
                mock_sample_diffusion.assert_called_once()

                # Check return value
                self.assertIsInstance(refined_coords, torch.Tensor)
                self.assertEqual(refined_coords.shape, (self.batch_size, self.seq_len, 3))

    def test_multi_step_inference_missing_s_trunk(self):
        """Test multi_step_inference method with missing s_trunk."""
        # Use a more targeted patch to avoid actual model execution
        with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule') as mock_diffusion_module:
            # Setup mocks
            mock_instance = MagicMock()
            mock_diffusion_module.return_value = mock_instance
            mock_instance.to.return_value = mock_instance

            # Initialize manager
            manager = ProtenixDiffusionManager(cfg=self.cfg)

            # Create trunk embeddings without s_trunk
            invalid_trunk_embeddings = {
                "s_inputs": self.trunk_embeddings["s_inputs"],
                "pair": self.trunk_embeddings["pair"]
            }

            # Check that ValueError is raised
            with self.assertRaises(ValueError) as context:
                manager.multi_step_inference(
                    coords_init=self.coords,
                    trunk_embeddings=invalid_trunk_embeddings
                )

            self.assertIn("requires a valid 's_trunk'", str(context.exception))

    def test_multi_step_inference_with_override_input_features(self):
        """Test multi_step_inference method with override_input_features."""
        # Use more targeted patches to avoid actual model execution
        with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule') as mock_diffusion_module:
            with patch('rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.sample_diffusion') as mock_sample_diffusion:
                # Setup mocks
                mock_instance = MagicMock()
                mock_diffusion_module.return_value = mock_instance
                mock_instance.to.return_value = mock_instance

                # Setup the mock to return a tuple (coords, loss)
                mock_instance.return_value = (torch.randn(self.batch_size, self.seq_len, 3), None)

                # Setup return value for sample_diffusion
                mock_sample_diffusion.return_value = torch.randn(self.batch_size, self.seq_len, 3)

                # Initialize manager
                manager = ProtenixDiffusionManager(cfg=self.cfg)

                # Create override input features
                override_input_features = {
                    "atom_mask": torch.ones(self.batch_size, self.seq_len),
                    "residue_index": torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1)
                }

                # Call multi_step_inference with override_input_features
                refined_coords = manager.multi_step_inference(
                    coords_init=self.coords,
                    trunk_embeddings=self.trunk_embeddings,
                    override_input_features=override_input_features
                )

                # Check that sample_diffusion was called
                mock_sample_diffusion.assert_called_once()

                # Check return value
                self.assertIsInstance(refined_coords, torch.Tensor)
                self.assertEqual(refined_coords.shape, (self.batch_size, self.seq_len, 3))


if __name__ == '__main__':
    unittest.main()
