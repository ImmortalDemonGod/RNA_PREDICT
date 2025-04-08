import logging
import os
import unittest

import numpy as np
import torch

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestStageARFoldPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set up configuration and checkpoint path
        self.config = {"num_hidden": 128, "dropout": 0.3, "use_gpu": False}

        # Check for possible checkpoint paths
        possible_paths = [
            "checkpoints/RNAStralign_trainset_pretrained.pth",
            "rna_predict/pipeline/stageA/checkpoints/RNAStralign_trainset_pretrained.pth",
            "rna_predict/pipeline/stageA/adjacency/checkpoints/RNAStralign_trainset_pretrained.pth",
        ]

        self.checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.checkpoint_path = path
                logger.info(f"Found checkpoint at: {path}")
                break

        if self.checkpoint_path is None:
            logger.warning(
                "No checkpoint file found. Tests will run without loading weights."
            )

    def test_instantiation(self):
        """Test successful instantiation of StageARFoldPredictor."""
        try:
            StageARFoldPredictor(config=self.config)
            logger.info(
                "StageARFoldPredictor instantiated successfully without checkpoint."
            )
        except Exception as e:
            self.fail(f"StageARFoldPredictor instantiation failed: {e}")

    def test_instantiation_with_checkpoint(self):
        """Test instantiation with checkpoint if available."""
        if self.checkpoint_path is None:
            logger.warning(
                "Skipping test_instantiation_with_checkpoint as no checkpoint file was found."
            )
            return

        try:
            StageARFoldPredictor(
                config=self.config, checkpoint_path=self.checkpoint_path
            )
            logger.info(
                "StageARFoldPredictor instantiated successfully with checkpoint."
            )
        except Exception as e:
            self.fail(f"StageARFoldPredictor instantiation with checkpoint failed: {e}")

    def test_config_loading(self):
        """Test correct loading of configuration parameters."""
        predictor = StageARFoldPredictor(config=self.config)
        self.assertIsNotNone(predictor.model, "Model not loaded")
        self.assertIsNotNone(predictor.device, "Device not configured")
        logger.info("Model and device loaded successfully.")

    def test_model_weights_loading(self):
        """Test successful loading of pre-trained RFold model weights."""
        predictor = StageARFoldPredictor(config=self.config)
        # Check if model weights are loaded correctly
        for name, param in predictor.model.named_parameters():
            self.assertIsNotNone(
                param.data, f"Model weights not loaded for layer: {name}"
            )
            logger.info(f"Model weights loaded successfully for layer: {name}")
            break

    def test_device_configuration(self):
        """Test correct configuration of the computational device (CPU or GPU)."""
        predictor = StageARFoldPredictor(config=self.config)
        if torch.cuda.is_available() and self.config.get("use_gpu", True):
            self.assertEqual(
                str(predictor.device), "cuda", "Device not configured to GPU"
            )
        else:
            self.assertEqual(
                str(predictor.device), "cpu", "Device not configured to CPU"
            )
        logger.info(f"Device configured to: {predictor.device}")

    def test_predict_adjacency_method_exists(self):
        """Test the existence and accessibility of the predict_adjacency method."""
        predictor = StageARFoldPredictor(config=self.config)
        self.assertTrue(
            hasattr(predictor, "predict_adjacency"),
            "predict_adjacency method not found",
        )
        self.assertTrue(
            callable(predictor.predict_adjacency), "predict_adjacency is not callable"
        )
        logger.info("predict_adjacency method exists and is callable.")

    def test_predict_adjacency_accepts_sequence(self):
        """Test that the method accepts a standard RNA sequence string as input."""
        predictor = StageARFoldPredictor(config=self.config)
        try:
            sequence = "AUCGUACGA"
            predictor.predict_adjacency(sequence)
            logger.info("predict_adjacency method accepted RNA sequence.")
        except Exception as e:
            self.fail(f"predict_adjacency method failed to accept RNA sequence: {e}")

    def test_output_validation(self):
        """Test output validation."""
        predictor = StageARFoldPredictor(config=self.config)
        sequence = "AUCGUACGA"
        output = predictor.predict_adjacency(sequence)

        # Determine the data type of the returned output
        self.assertTrue(
            isinstance(output, (np.ndarray, torch.Tensor)),
            f"Output is not a NumPy array or PyTorch Tensor, got {type(output)}",
        )
        logger.info(f"Output data type: {type(output)}")

        # Verify that the output adjacency matrix has the expected shape of [N, N]
        N = len(sequence)
        if isinstance(output, np.ndarray):
            self.assertEqual(
                output.shape, (N, N), f"Output shape is not [N, N], got {output.shape}"
            )
        else:
            self.assertEqual(
                output.shape,
                torch.Size([N, N]),
                f"Output shape is not [N, N], got {output.shape}",
            )
        logger.info(f"Output shape: {output.shape}")

        # Assess the validity of values within the output matrix
        if isinstance(output, np.ndarray):
            self.assertTrue(
                np.all((output >= 0.0) & (output <= 1.0)),
                "Output values are not within the expected range",
            )
            self.assertFalse(np.any(np.isnan(output)), "Output contains NaN values")
            self.assertFalse(np.any(np.isinf(output)), "Output contains Inf values")
        else:
            self.assertTrue(
                torch.all((output >= 0.0) & (output <= 1.0)),
                "Output values are not within the expected range",
            )
            self.assertFalse(
                torch.any(torch.isnan(output)), "Output contains NaN values"
            )
            self.assertFalse(
                torch.any(torch.isinf(output)), "Output contains Inf values"
            )
        logger.info("Output values are valid.")

    def test_basic_operational_functionality(self):
        """Test basic operational functionality."""
        predictor = StageARFoldPredictor(config=self.config)
        sequence = "AUCGUACGA"
        try:
            result = predictor.predict_adjacency(sequence)
            logger.info(
                f"StageARFoldPredictor executed successfully. Result shape: {result.shape}"
            )
        except Exception as e:
            self.fail(f"StageARFoldPredictor failed to execute: {e}")


if __name__ == "__main__":
    unittest.main()
