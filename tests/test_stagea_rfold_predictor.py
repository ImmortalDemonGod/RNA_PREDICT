import sys
import os
import torch
from omegaconf import OmegaConf

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor

# Use a DictConfig matching the real Hydra config structure
config_dict = {
    "min_seq_length": 16,
    "num_hidden": 128,
    "dropout": 0.1,
    "batch_size": 1,
    "lr": 0.001,
    "checkpoint_path": "dummy_path",
    "model": {
        "conv_channels": [1, 32, 64, 128],
        "residual": False,
        "c_in": 1,
        "c_out": 1,
        "c_hid": 32,
        "seq2map": {
            "input_dim": 4,
            "max_length": 1000,
            "attention_heads": 8,
            "attention_dropout": 0.1,
            "positional_encoding": True,
            "query_key_dim": 64,
            "expansion_factor": 2.0,
            "heads": 8,
        },
        "decoder": {
            "up_conv_channels": [128, 64, 32],
            "skip_connections": True,
        },
    }
}

config = OmegaConf.create(config_dict)

class TestStageARFoldPredictor:
    def setup_method(self):
        """Set up the test environment before each test method."""
        self.config = config
        self.device = torch.device("cpu")

    def test_instantiation(self):
        """Test that the StageARFoldPredictor can be instantiated successfully."""
        predictor = StageARFoldPredictor(self.config, self.device)
        assert isinstance(predictor, StageARFoldPredictor), "Predictor should be an instance of StageARFoldPredictor"

    def test_predict_adjacency(self):
        """Test that predict_adjacency produces output with the expected shape."""
        predictor = StageARFoldPredictor(self.config, self.device)
        seq = "AUGC"
        adjacency = predictor.predict_adjacency(seq)
        expected_shape = (len(seq), len(seq))
        assert adjacency.shape == expected_shape, f"Expected shape {expected_shape}, got {adjacency.shape}"
        # Check dtype in a way that works with both numpy and torch dtypes
        assert str(adjacency.dtype) == 'float32', f"Expected dtype float32, got {adjacency.dtype}"
        # Use numpy's all or torch's all depending on the type
        if hasattr(torch, 'is_tensor') and torch.is_tensor(adjacency):
            assert torch.all((adjacency >= 0) & (adjacency <= 1)), "Adjacency values should be in [0, 1]"
        else:
            import numpy as np
            assert np.all((adjacency >= 0) & (adjacency <= 1)), "Adjacency values should be in [0, 1]"
