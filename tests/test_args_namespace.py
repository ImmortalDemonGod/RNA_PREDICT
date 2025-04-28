import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import args_namespace
from rna_predict.pipeline.stageA.adjacency.RFold_code import RFoldModel

# Create a simple config dict
config_dict = {
    "num_hidden": 16,
    "dropout": 0.1,
    "batch_size": 1,
    "lr": 0.001,
    "conv_channels": [1, 32, 64, 128],
    "residual": False,
    "c_in": 1,
    "c_out": 1,
    "c_hid": 32,
    "seq2map_input_dim": 4,
    "seq2map_max_length": 1000,
    "seq2map_attention_heads": 8,
    "seq2map_attention_dropout": 0.1,
    "seq2map_positional_encoding": True,
    "seq2map_query_key_dim": 64,
    "seq2map_expansion_factor": 2.0,
    "seq2map_heads": 8,
    "decoder_up_conv_channels": [128, 64, 32],
    "decoder_skip_connections": True,
}

# Create args namespace
print("Creating args namespace...")
args = args_namespace(config_dict)
print(f"Args type: {type(args)}")
print(f"Args attributes: {dir(args)}")

# Create RFoldModel
print("Creating RFoldModel...")
model = RFoldModel(args)
print(f"Model type: {type(model)}")
print("Success!")
