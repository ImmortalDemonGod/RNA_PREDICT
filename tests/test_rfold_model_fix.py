import torch
import types

from rna_predict.pipeline.stageA.adjacency.RFold_code import RFoldModel

def test_rfold_model():
    # Create a simple namespace object with the required attributes
    args = types.SimpleNamespace()
    args.num_hidden = 128
    args.dropout = 0.1
    args.use_gpu = False
    args.device = "cpu"  # Add explicit device parameter

    # Create RFoldModel
    model = RFoldModel(args)
    assert isinstance(model, RFoldModel), "Failed to create RFoldModel instance"

    # Test forward pass
    seqs = torch.randint(0, 4, (1, 16))
    output = model(seqs)
    expected_shape = (1, 16, 16)  # Update with the expected output shape
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
