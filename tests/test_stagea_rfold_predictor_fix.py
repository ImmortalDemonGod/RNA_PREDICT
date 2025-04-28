import sys
import os
import torch
from omegaconf import DictConfig, OmegaConf

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor

# Create a simple config
config = OmegaConf.create({
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
    },
})

# Create StageARFoldPredictor
print("Creating StageARFoldPredictor...")
predictor = StageARFoldPredictor(config, torch.device("cpu"))
print(f"Predictor type: {type(predictor)}")

# Test predict_adjacency
print("Testing predict_adjacency...")
adjacency = predictor.predict_adjacency("AUGC")
print(f"Adjacency shape: {adjacency.shape}")
print("Success!")
