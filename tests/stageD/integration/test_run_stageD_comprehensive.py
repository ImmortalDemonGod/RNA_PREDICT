"""
Comprehensive tests for run_stageD.py to improve test coverage.
"""

import torch
import unittest
from unittest.mock import patch, MagicMock
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import io

from rna_predict.pipeline.stageD.run_stageD import run_stageD, hydra_main


class TestRunStageDComprehensive(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create mock tensors and embeddings
        self.batch_size = 1
        self.seq_len = 50
        self.feature_dim = 64

        # Create atom-level coordinates
        self.atom_coords = torch.randn(self.batch_size, self.seq_len, 3)

        # Create atom-level embeddings with both tensor and non-tensor values
        self.atom_embeddings = {
            "s_trunk": torch.randn(self.batch_size, self.seq_len, self.feature_dim),
            "s_inputs": torch.randn(self.batch_size, self.seq_len, self.feature_dim * 2),
            "pair": torch.randn(self.batch_size, self.seq_len, self.seq_len, self.feature_dim // 2),
            "atom_to_token_idx": torch.randint(0, self.seq_len, (self.batch_size, self.seq_len)).long(),
            "non_tensor_value": "test_string",
            "list_value": [1, 2, 3],
            "dict_value": {"key": "value"}
        }

        # Create a mock Hydra config
        self.cfg = OmegaConf.create({
            "stageD_diffusion": {
                "device": "cpu",
                "memory": {
                    "apply_memory_preprocess": False,
                    "memory_preprocess_max_len": 25
                },
                "debug_logging": False,
                "inference": {
                    "num_steps": 2,  # Use a small value for testing
                    "sampling": {
                        "num_samples": 1
                    }
                }
            }
        })

        # Create a mock config with debug logging enabled
        self.cfg_with_debug = OmegaConf.create({
            "stageD_diffusion": {
                "device": "cpu",
                "memory": {
                    "apply_memory_preprocess": True,
                    "memory_preprocess_max_len": 25
                },
                "debug_logging": True,
                "inference": {
                    "num_steps": 2,  # Use a small value for testing
                    "sampling": {
                        "num_samples": 1
                    }
                }
            }
        })

    @patch('rna_predict.pipeline.stageD.run_stageD.ProtenixDiffusionManager')
    def test_run_stageD_basic(self, mock_manager_class):
        """Test basic functionality of run_stageD."""
        # Setup mock manager
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.multi_step_inference.return_value = torch.randn(self.batch_size, self.seq_len, 3)

        # Run the function
        result = run_stageD(
            cfg=self.cfg,
            atom_coords=self.atom_coords,
            atom_embeddings=self.atom_embeddings
        )

        # Check that the manager was initialized with the correct config
        mock_manager_class.assert_called_once_with(self.cfg)

        # Check that multi_step_inference was called with the correct arguments
        mock_manager.multi_step_inference.assert_called_once()

        # Check that the result is a tensor with the expected shape
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (self.batch_size, self.seq_len, 3))

    @patch('rna_predict.pipeline.stageD.run_stageD.ProtenixDiffusionManager')
    def test_run_stageD_with_debug_logging(self, mock_manager_class):
        """Test run_stageD with debug logging enabled."""
        # Setup mock manager
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.multi_step_inference.return_value = torch.randn(self.batch_size, self.seq_len, 3)

        # Capture stdout to check debug logs
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Run the function with debug logging enabled
            result = run_stageD(
                cfg=self.cfg_with_debug,
                atom_coords=self.atom_coords,
                atom_embeddings=self.atom_embeddings
            )

            # Check that debug logs were printed
            output = captured_output.getvalue()
            self.assertIn("[DEBUG] --- Running Stage D ---", output)
            self.assertIn(f"[DEBUG] Input coords shape: {self.atom_coords.shape}", output)
            self.assertIn("[DEBUG] Input embeddings keys:", output)
            self.assertIn("[DEBUG] Apply preprocessing: True", output)
            self.assertIn("[DEBUG] Preprocessing max length: 25", output)
            self.assertIn("[DEBUG] Processed coords shape:", output)
            self.assertIn("[DEBUG] Processed embeddings keys:", output)
            self.assertIn("[DEBUG] Final refined coords shape:", output)
            self.assertIn("[DEBUG] --- Stage D Complete ---", output)
        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__

    @patch('rna_predict.pipeline.stageD.run_stageD.ProtenixDiffusionManager')
    def test_run_stageD_with_preprocessing(self, mock_manager_class):
        """Test run_stageD with preprocessing enabled."""
        # Setup mock manager
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.multi_step_inference.return_value = torch.randn(self.batch_size, 25, 3)  # Reduced size

        # Create config with preprocessing enabled
        cfg_with_preprocessing = OmegaConf.create({
            "stageD_diffusion": {
                "device": "cpu",
                "memory": {
                    "apply_memory_preprocess": True,
                    "memory_preprocess_max_len": 25
                },
                "debug_logging": False,
                "inference": {
                    "num_steps": 2,
                    "sampling": {
                        "num_samples": 1
                    }
                }
            }
        })

        # Run the function
        result = run_stageD(
            cfg=cfg_with_preprocessing,
            atom_coords=self.atom_coords,
            atom_embeddings=self.atom_embeddings
        )

        # Check that the result has the expected shape after preprocessing
        self.assertEqual(result.shape, (self.batch_size, 25, 3))

    def test_run_stageD_missing_config_group(self):
        """Test run_stageD with missing stageD_diffusion config group."""
        # Create config without stageD_diffusion group
        invalid_cfg = OmegaConf.create({
            "some_other_group": {}
        })

        # Check that ValueError is raised
        with self.assertRaises(ValueError) as context:
            run_stageD(
                cfg=invalid_cfg,
                atom_coords=self.atom_coords,
                atom_embeddings=self.atom_embeddings
            )

        self.assertIn("Config missing required 'stageD_diffusion' group", str(context.exception))

    @patch('rna_predict.pipeline.stageD.run_stageD.run_stageD')
    def test_hydra_main(self, mock_run_stageD):
        """Test the hydra_main function."""
        # Setup mock run_stageD
        mock_run_stageD.return_value = torch.randn(1, 50, 3)

        # Create a more complete config that includes diffusion_model
        complete_cfg = OmegaConf.create({
            "stageD_diffusion": {
                "device": "cpu",
                "memory": {
                    "apply_memory_preprocess": False,
                    "memory_preprocess_max_len": 25
                },
                "debug_logging": False,
                "inference": {
                    "num_steps": 2,
                    "sampling": {
                        "num_samples": 1
                    }
                }
            },
            "diffusion_model": {
                "c_s": 384,
                "c_z": 64,
                "c_s_inputs": 449
            },
            "noise_schedule": {
                "schedule_type": "linear"
            }
        })

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Run hydra_main with our complete config
            hydra_main(complete_cfg)

            # Check that output contains expected strings
            output = captured_output.getvalue()
            self.assertIn("Running Stage D Standalone Demo with Hydra configuration:", output)
            self.assertIn("Resolved Hydra Config (Stage D Relevant Sections):", output)
            self.assertIn("Creating dummy data", output)
            self.assertIn("Calling run_stageD...", output)
            self.assertIn("Stage D Standalone Demo Output:", output)
            self.assertIn("Refined Coords Shape:", output)
        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__

    @patch('rna_predict.pipeline.stageD.run_stageD.run_stageD')
    def test_hydra_main_with_error(self, mock_run_stageD):
        """Test the hydra_main function when run_stageD raises an exception."""
        # Setup mock run_stageD to raise an exception
        mock_run_stageD.side_effect = RuntimeError("Test error")

        # Create a more complete config that includes diffusion_model
        complete_cfg = OmegaConf.create({
            "stageD_diffusion": {
                "device": "cpu",
                "memory": {
                    "apply_memory_preprocess": False,
                    "memory_preprocess_max_len": 25
                },
                "debug_logging": False,
                "inference": {
                    "num_steps": 2,
                    "sampling": {
                        "num_samples": 1
                    }
                }
            },
            "diffusion_model": {
                "c_s": 384,
                "c_z": 64,
                "c_s_inputs": 449
            },
            "noise_schedule": {
                "schedule_type": "linear"
            }
        })

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Run hydra_main with our complete config
            hydra_main(complete_cfg)

            # Check that error message is printed
            output = captured_output.getvalue()
            self.assertIn("Error during run_stageD execution: Test error", output)
        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__


if __name__ == '__main__':
    unittest.main()
