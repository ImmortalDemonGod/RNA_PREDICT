import sys
import os
import types
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor

# Create a simple namespace object with the required attributes
class MockConfig:
    def __init__(self):
        self.min_seq_length = 16
        self.num_hidden = 128
        self.dropout = 0.1
        self.batch_size = 1
        self.lr = 0.001
        self.checkpoint_path = "dummy_path"
        self.model = types.SimpleNamespace()
        self.model.conv_channels = [1, 32, 64, 128]
        self.model.residual = False
        self.model.c_in = 1
        self.model.c_out = 1
        self.model.c_hid = 32
        self.model.seq2map = types.SimpleNamespace()
        self.model.seq2map.input_dim = 4
        self.model.seq2map.max_length = 1000
        self.model.seq2map.attention_heads = 8
        self.model.seq2map.attention_dropout = 0.1
        self.model.seq2map.positional_encoding = True
        self.model.seq2map.query_key_dim = 64
        self.model.seq2map.expansion_factor = 2.0
        self.model.seq2map.heads = 8
        self.model.decoder = types.SimpleNamespace()
        self.model.decoder.up_conv_channels = [128, 64, 32]
        self.model.decoder.skip_connections = True

# Create a mock config
config = MockConfig()

# Create StageARFoldPredictor
print("Creating StageARFoldPredictor...")
predictor = StageARFoldPredictor(config, torch.device("cpu"))
print(f"Predictor type: {type(predictor)}")

# Test predict_adjacency
print("Testing predict_adjacency...")
adjacency = predictor.predict_adjacency("AUGC")
print(f"Adjacency shape: {adjacency.shape}")
print("Success!")
