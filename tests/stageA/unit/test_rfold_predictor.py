import logging
import os
import unittest
import pytest
from omegaconf import OmegaConf

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
        # Create a config that matches the expected structure from Hydra
        self.stage_cfg = OmegaConf.create({
            "num_hidden": 128,
            "dropout": 0.3,
            "min_seq_length": 80,
            "device": "cpu",
            "checkpoint_path": "RFold/checkpoints/RNAStralign_trainset_pretrained.pth",
            "checkpoint_url": "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1",
            "batch_size": 32,
            "lr": 0.001,
            "threshold": 0.5,
            "visualization": {
                "enabled": True,
                "varna_jar_path": "tools/varna-3-93.jar",
                "resolution": 8.0
            },
            "model": {
                "conv_channels": [64, 128, 256, 512],
                "residual": True,
                "c_in": 1,
                "c_out": 1,
                "c_hid": 32,
                "seq2map": {
                    "input_dim": 4,
                    "max_length": 3000,
                    "attention_heads": 8,
                    "attention_dropout": 0.1,
                    "positional_encoding": True,
                    "query_key_dim": 128,
                    "expansion_factor": 2.0,
                    "heads": 1
                },
                "decoder": {
                    "up_conv_channels": [256, 128, 64],
                    "skip_connections": True
                }
            }
        })

        # For backward compatibility with old tests
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
            device = torch.device("cpu")
            StageARFoldPredictor(stage_cfg=self.stage_cfg, device=device)
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
            # Update the stage_cfg with the actual checkpoint path
            import copy
            stage_cfg_with_checkpoint = copy.deepcopy(self.stage_cfg)
            stage_cfg_with_checkpoint.checkpoint_path = self.checkpoint_path
            device = torch.device("cpu")
            StageARFoldPredictor(stage_cfg=stage_cfg_with_checkpoint, device=device)
            logger.info(
                "StageARFoldPredictor instantiated successfully with checkpoint."
            )
        except Exception as e:
            self.fail(f"StageARFoldPredictor instantiation with checkpoint failed: {e}")

    def test_config_loading(self):
        """Test correct loading of configuration parameters."""
        device = torch.device("cpu")
        predictor = StageARFoldPredictor(stage_cfg=self.stage_cfg, device=device)
        self.assertIsNotNone(predictor.model, "Model not loaded")
        self.assertIsNotNone(predictor.device, "Device not configured")
        logger.info("Model and device loaded successfully.")

    def test_model_weights_loading(self):
        """Test successful loading of pre-trained RFold model weights."""
        device = torch.device("cpu")
        predictor = StageARFoldPredictor(stage_cfg=self.stage_cfg, device=device)
        # Check if model weights are loaded correctly
        for name, param in predictor.model.named_parameters():
            self.assertIsNotNone(
                param.data, f"Model weights not loaded for layer: {name}"
            )
            logger.info(f"Model weights loaded successfully for layer: {name}")
            break

    def test_device_configuration(self):
        """Test correct configuration of the computational device (CPU or GPU)."""
        device = torch.device("cpu")
        predictor = StageARFoldPredictor(stage_cfg=self.stage_cfg, device=device)
        self.assertEqual(
            str(predictor.device), "cpu", "Device not configured to CPU"
        )
        logger.info(f"Device configured to: {predictor.device}")

    def test_predict_adjacency_method_exists(self):
        """Test the existence and accessibility of the predict_adjacency method."""
        device = torch.device("cpu")
        predictor = StageARFoldPredictor(stage_cfg=self.stage_cfg, device=device)
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
        device = torch.device("cpu")
        predictor = StageARFoldPredictor(stage_cfg=self.stage_cfg, device=device)
        try:
            sequence = "AUCGUACGA"
            predictor.predict_adjacency(sequence)
            logger.info("predict_adjacency method accepted RNA sequence.")
        except Exception as e:
            self.fail(f"predict_adjacency method failed to accept RNA sequence: {e}")

    def test_output_validation(self):
        """Test output validation."""
        device = torch.device("cpu")
        predictor = StageARFoldPredictor(stage_cfg=self.stage_cfg, device=device)
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
        device = torch.device("cpu")
        predictor = StageARFoldPredictor(stage_cfg=self.stage_cfg, device=device)
        sequence = "AUCGUACGA"
        try:
            result = predictor.predict_adjacency(sequence)
            logger.info(
                f"StageARFoldPredictor executed successfully. Result shape: {result.shape}"
            )
        except Exception as e:
            self.fail(f"StageARFoldPredictor failed to execute: {e}")


@pytest.mark.usefixtures("caplog")
def test_debug_logging_emission(caplog):
    """Test that debug/info logs are emitted when debug_logging=True (StageA) [ERR-STAGEA-DEBUG-001]"""
    from rna_predict.pipeline.stageA.adjacency import rfold_predictor

    # Use the real logger object
    logger = rfold_predictor.logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    if hasattr(logger, "handlers"):
        logger.handlers.clear()
    # Compose a config with debug_logging True
    stage_cfg = OmegaConf.create({
        "debug_logging": True,
        "num_hidden": 128,
        "dropout": 0.3,
        "min_seq_length": 80,
        "device": "cpu",
        "checkpoint_path": None,
        "batch_size": 32,
        "lr": 0.001,
        "threshold": 0.5,
        "model": {
            "conv_channels": [64, 128],
            "decoder": {"up_conv_channels": [64], "skip_connections": True},
            "residual": True,
            "c_in": 1,
            "c_out": 1,
            "c_hid": 32,
            "seq2map": {
                "input_dim": 4,
                "max_length": 3000,
                "attention_heads": 8,
                "attention_dropout": 0.1,
                "positional_encoding": True,
                "query_key_dim": 128,
                "expansion_factor": 2.0,
                "heads": 1
            }
        },
        "decoder": {
            "up_conv_channels": [256, 128, 64],
            "skip_connections": True
        }
    })
    device = torch.device("cpu")
    caplog.set_level(logging.DEBUG)
    predictor = rfold_predictor.StageARFoldPredictor(stage_cfg=stage_cfg, device=device)
    # Trigger prediction to emit logs
    predictor.predict_adjacency("AUGCUAGU")
    # Assert on expected debug/info log lines
    log_text = caplog.text
    assert "Adjacency matrix shape" in log_text, "[ERR-STAGEA-DEBUG-002] Adjacency matrix shape info not logged."
    assert "Adjacency matrix data type" in log_text, "[ERR-STAGEA-DEBUG-003] Adjacency matrix dtype debug not logged."


if __name__ == "__main__":
    unittest.main()
