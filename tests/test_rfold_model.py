import sys
import os
from unittest import mock
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from rna_predict.pipeline.stageA.adjacency.RFold_code import RFoldModel

class TestRFoldModel:
    def setup_method(self):
        """
        Prepares mock model arguments before each test method.
        
        Initializes a mock object with attributes for model configuration, including
        number of hidden units, dropout rate, GPU usage flag, and device specification.
        """
        # Create a simple namespace object with the required attributes
        self.args = mock.MagicMock()
        self.args.num_hidden = 128
        self.args.dropout = 0.1
        self.args.use_gpu = False
        self.args.device = "cpu"  # Add explicit device parameter

    def test_model_instantiation(self):
        """Test that the RFoldModel can be instantiated successfully."""
        model = RFoldModel(self.args)
        assert isinstance(model, RFoldModel), "Model should be an instance of RFoldModel"

    def test_forward_pass(self):
        """Test that the forward pass produces output with the expected shape."""
        model = RFoldModel(self.args)
        seqs = torch.randint(0, 4, (1, 16))  # Random sequence batch of shape [1, 16]
        output = model(seqs)

        # Assert the output has the expected shape
        # Handle both torch tensors and numpy arrays
        if hasattr(torch, 'is_tensor') and torch.is_tensor(output):
            assert output.dim() == 3, f"Expected 3D output, got shape {output.shape}"
        else:
            # For numpy arrays, use ndim instead of dim
            assert len(output.shape) == 3, f"Expected 3D output, got shape {output.shape}"

        assert output.shape[0] == 1, f"Expected batch size 1, got {output.shape[0]}"
        assert output.shape[1] == 16, f"Expected sequence length 16, got {output.shape[1]}"
        assert output.shape[2] == 16, f"Expected output width 16, got {output.shape[2]}"
        # Use correct float dtype assertion
        assert output.dtype.is_floating_point, f"Expected float dtype, got {output.dtype}"
