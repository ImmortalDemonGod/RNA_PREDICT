"""
Comprehensive tests for memory_fix.py to improve test coverage.
"""

import torch
import unittest
import warnings
from rna_predict.pipeline.stageD.memory_optimization.memory_fix import (
    clear_memory,
    preprocess_inputs
)


class TestMemoryFixComprehensive(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create tensors with various shapes for testing
        self.batch_size = 2
        self.seq_len = 100
        self.max_seq_len = 25
        self.feature_dim = 64
        
        # 3D tensor [batch, seq_len, feature]
        self.coords_3d = torch.randn(self.batch_size, self.seq_len, 3)
        
        # 2D tensor [seq_len, feature]
        self.coords_2d = torch.randn(self.seq_len, 3)
        
        # Create embeddings dictionary with various tensor shapes and non-tensor values
        self.embeddings_dict = {
            # 3D tensor for s_trunk [batch, seq_len, feature]
            "s_trunk": torch.randn(self.batch_size, self.seq_len, self.feature_dim),
            
            # 2D tensor [seq_len, feature]
            "s_2d": torch.randn(self.seq_len, self.feature_dim),
            
            # 4D tensor for pair [batch, seq_len, seq_len, feature]
            "pair": torch.randn(self.batch_size, self.seq_len, self.seq_len, self.feature_dim // 2),
            
            # Non-tensor value
            "non_tensor": "test_string",
            
            # List value
            "list_value": [1, 2, 3],
            
            # Dictionary value
            "dict_value": {"key": "value"}
        }

    def test_clear_memory(self):
        """Test that clear_memory runs without errors."""
        # This is mostly a smoke test since we can't easily verify GC behavior
        clear_memory()
        self.assertTrue(True)  # If we got here, the function didn't crash

    def test_preprocess_inputs_3d_coords(self):
        """Test preprocessing of 3D coordinate tensors."""
        processed_coords, _ = preprocess_inputs(
            self.coords_3d,
            {"dummy": torch.tensor([0])},  # Empty dict to focus on coords
            max_seq_len=self.max_seq_len
        )
        
        # Check shape is truncated correctly
        self.assertEqual(processed_coords.shape, (self.batch_size, self.max_seq_len, 3))

    def test_preprocess_inputs_2d_coords(self):
        """Test preprocessing of 2D coordinate tensors with warning."""
        with warnings.catch_warnings(record=True) as w:
            processed_coords, _ = preprocess_inputs(
                self.coords_2d,
                {"dummy": torch.tensor([0])},  # Empty dict to focus on coords
                max_seq_len=self.max_seq_len
            )
            
            # Check warning was raised
            self.assertEqual(len(w), 1)
            self.assertTrue("truncating dim 0" in str(w[0].message))
            
            # Check shape is truncated correctly
            self.assertEqual(processed_coords.shape, (self.max_seq_len, 3))

    def test_preprocess_inputs_embeddings(self):
        """Test preprocessing of embeddings dictionary with various tensor types."""
        _, processed_embeddings = preprocess_inputs(
            torch.zeros(1, 1, 3),  # Dummy coords to focus on embeddings
            self.embeddings_dict,
            max_seq_len=self.max_seq_len
        )
        
        # Check tensor shapes are truncated correctly
        self.assertEqual(processed_embeddings["s_trunk"].shape, 
                         (self.batch_size, self.max_seq_len, self.feature_dim))
        
        # Check pair tensor is truncated in both dimensions
        self.assertEqual(processed_embeddings["pair"].shape, 
                         (self.batch_size, self.max_seq_len, self.max_seq_len, self.feature_dim // 2))
        
        # Check 2D tensor with warning
        with warnings.catch_warnings(record=True) as w:
            _, processed_embeddings = preprocess_inputs(
                torch.zeros(1, 1, 3),  # Dummy coords
                {"s_2d": self.embeddings_dict["s_2d"]},  # Only include 2D tensor
                max_seq_len=self.max_seq_len
            )
            
            # Check warning was raised
            self.assertEqual(len(w), 1)
            self.assertTrue("truncating dim 0" in str(w[0].message))
            
            # Check shape is truncated correctly
            self.assertEqual(processed_embeddings["s_2d"].shape, (self.max_seq_len, self.feature_dim))

    def test_preprocess_inputs_non_tensors(self):
        """Test that non-tensor values in embeddings are preserved."""
        _, processed_embeddings = preprocess_inputs(
            torch.zeros(1, 1, 3),  # Dummy coords
            self.embeddings_dict,
            max_seq_len=self.max_seq_len
        )
        
        # Check non-tensor values are preserved
        self.assertEqual(processed_embeddings["non_tensor"], "test_string")
        self.assertEqual(processed_embeddings["list_value"], [1, 2, 3])
        self.assertEqual(processed_embeddings["dict_value"], {"key": "value"})

    def test_preprocess_inputs_no_truncation_needed(self):
        """Test that tensors smaller than max_seq_len are not truncated."""
        small_coords = torch.randn(self.batch_size, self.max_seq_len - 5, 3)
        small_embeddings = {
            "s_trunk": torch.randn(self.batch_size, self.max_seq_len - 5, self.feature_dim),
            "pair": torch.randn(self.batch_size, self.max_seq_len - 5, self.max_seq_len - 5, self.feature_dim // 2)
        }
        
        processed_coords, processed_embeddings = preprocess_inputs(
            small_coords,
            small_embeddings,
            max_seq_len=self.max_seq_len
        )
        
        # Check shapes are unchanged
        self.assertEqual(processed_coords.shape, small_coords.shape)
        self.assertEqual(processed_embeddings["s_trunk"].shape, small_embeddings["s_trunk"].shape)
        self.assertEqual(processed_embeddings["pair"].shape, small_embeddings["pair"].shape)


if __name__ == '__main__':
    unittest.main()
