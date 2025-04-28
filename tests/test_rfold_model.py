import sys
import os
import types
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from rna_predict.pipeline.stageA.adjacency.RFold_code import RFoldModel

# Create a simple namespace object with the required attributes
args = types.SimpleNamespace()
args.num_hidden = 128
args.dropout = 0.1
args.use_gpu = False

# Create RFoldModel
print("Creating RFoldModel...")
model = RFoldModel(args)
print(f"Model type: {type(model)}")

# Test forward pass
print("Testing forward pass...")
seqs = torch.randint(0, 4, (1, 16))
output = model(seqs)
print(f"Output shape: {output.shape}")
print("Success!")
