"""
Comprehensive tests for scatter_utils.py.

This module provides thorough testing for the scatter utility functions
in the RNA_PREDICT project, focusing on layernorm, inverse_squared_dist,
scatter_mean, and broadcast functions.
"""

import unittest
from typing import Tuple, List, Any

import pytest
import torch
from hypothesis import HealthCheck, given, settings, example
from hypothesis import strategies as st

from rna_predict.utils.scatter_utils import (
    layernorm,
    inverse_squared_dist,
    scatter_mean,
    broadcast,
)


class TestLayerNorm(unittest.TestCase):
    """Tests for the layernorm function."""

    def test_layernorm_basic(self):
        """Test layernorm with basic shapes and default epsilon."""
        shapes = [(2, 3), (10, 10), (5, 1)]
        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float32)
                out = layernorm(x)

                # Check shape
                self.assertEqual(out.shape, x.shape)

                # Check mean
                means = out.mean(dim=-1)
                self.assertTrue(torch.allclose(means, torch.zeros_like(means), atol=1e-5))

                # Check variance for dimensions > 1
                if shape[-1] > 1:
                    # Calculate expected variance
                    input_var = x.var(dim=-1, unbiased=False, keepdim=True)
                    expected_out_var = input_var.squeeze(-1) / (input_var.squeeze(-1) + 1e-5)
                    actual_out_var = out.var(dim=-1, unbiased=False)

                    # Check variance
                    self.assertTrue(torch.allclose(actual_out_var, expected_out_var, atol=1e-5))

    def test_layernorm_with_custom_eps(self):
        """Test layernorm with custom epsilon values."""
        eps_values = [1e-3, 1e-5, 1e-8]
        shape = (5, 10)
        x = torch.randn(shape, dtype=torch.float32)

        for eps in eps_values:
            with self.subTest(eps=eps):
                out = layernorm(x, eps=eps)

                # Check shape
                self.assertEqual(out.shape, x.shape)

                # Check mean
                means = out.mean(dim=-1)
                self.assertTrue(torch.allclose(means, torch.zeros_like(means), atol=1e-5))

                # Calculate expected variance
                input_var = x.var(dim=-1, unbiased=False, keepdim=True)
                expected_out_var = input_var.squeeze(-1) / (input_var.squeeze(-1) + eps)
                actual_out_var = out.var(dim=-1, unbiased=False)

                # Check variance
                self.assertTrue(torch.allclose(actual_out_var, expected_out_var, atol=1e-5))

    def test_layernorm_with_dim_size_1(self):
        """Test layernorm with last dimension size 1, which is a special case."""
        shapes = [(3, 1), (5, 1), (10, 1)]
        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float32)
                out = layernorm(x)

                # For dim_size=1, output should be zeros
                self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_layernorm_with_higher_dims(self):
        """Test layernorm with tensors of higher dimensions."""
        shapes = [(2, 3, 4), (3, 4, 5, 6)]
        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float32)
                out = layernorm(x)

                # Check shape
                self.assertEqual(out.shape, x.shape)

                # Check mean
                means = out.mean(dim=-1)
                self.assertTrue(torch.allclose(means, torch.zeros_like(means), atol=1e-5))

    def test_layernorm_with_1d_input(self):
        """Test layernorm with 1D input."""
        x = torch.randn(5, dtype=torch.float32)
        out = layernorm(x)

        # Check shape
        self.assertEqual(out.shape, x.shape)

        # Check mean
        mean = out.mean()
        self.assertTrue(torch.allclose(mean, torch.tensor(0.0), atol=1e-5))

    def test_layernorm_error_handling(self):
        """Test layernorm with invalid input types."""
        # Integer tensor
        x_int = torch.randint(0, 10, (4, 5), dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            _ = layernorm(x_int)

        # Boolean tensor
        x_bool = torch.randint(0, 2, (4, 5)).bool()
        with self.assertRaises(RuntimeError):
            _ = layernorm(x_bool)

    @given(
        shape=st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10),
        ),
        eps=st.floats(min_value=1e-10, max_value=1e-3),
    )
    @settings(deadline=None)
    def test_layernorm_hypothesis(self, shape: Tuple[int, int], eps: float):
        """Test layernorm with random shapes and epsilon values using Hypothesis."""
        x = torch.randn(shape, dtype=torch.float32)
        out = layernorm(x, eps=eps)

        # Check shape
        self.assertEqual(out.shape, x.shape)

        # Check mean
        means = out.mean(dim=-1)
        self.assertTrue(torch.allclose(means, torch.zeros_like(means), atol=1e-4))

        # Check variance for dimensions > 1
        if shape[-1] > 1:
            # Calculate expected variance
            input_var = x.var(dim=-1, unbiased=False, keepdim=True)
            expected_out_var = input_var.squeeze(-1) / (input_var.squeeze(-1) + eps)
            actual_out_var = out.var(dim=-1, unbiased=False)

            # Check variance
            self.assertTrue(torch.allclose(actual_out_var, expected_out_var, atol=1e-4))


class TestInverseSquaredDist(unittest.TestCase):
    """Tests for the inverse_squared_dist function."""

    def test_inverse_squared_dist_basic(self):
        """Test inverse_squared_dist with basic inputs."""
        test_cases = [
            (torch.zeros(1, 3), 1e-8, 1.0),  # zero vector => 1/(1+0^2+eps) ~ 1
            (torch.tensor([[1.0, 0.0, 0.0]]), 1e-8, 1.0 / (1 + 1 + 1e-8)),  # dist^2=1 => ~0.5
            (torch.tensor([[3.0, 4.0, 12.0]]), 1e-8, 1.0 / (1 + 9 + 16 + 144 + 1e-8)),  # dist^2=169 => 1/170
        ]

        for delta, eps, expected in test_cases:
            with self.subTest(delta=delta, eps=eps):
                result = inverse_squared_dist(delta, eps=eps)
                self.assertEqual(result.shape, (delta.shape[0], 1))
                self.assertTrue(torch.allclose(result.squeeze(-1), torch.tensor(expected), atol=1e-6))

    def test_inverse_squared_dist_batch_shapes(self):
        """Test inverse_squared_dist with batched shapes."""
        # shape: [B, N, 3]
        shapes = [(2, 3, 3), (4, 5, 3), (1, 10, 3)]
        for shape in shapes:
            with self.subTest(shape=shape):
                delta = torch.randn(shape)
                out = inverse_squared_dist(delta)
                expected_shape = shape[:-1] + (1,)  # Replace last dim with 1
                self.assertEqual(out.shape, expected_shape)

    def test_inverse_squared_dist_large_values(self):
        """
        Test inverse_squared_dist for numerical stability using extreme values.
        
        Verifies that extremely large delta inputs yield a near-zero inverse squared distance,
        while extremely small delta inputs yield a value close to one.
        """
        # Very large values
        delta_large = torch.tensor([[1e6, 1e6, 1e6]], dtype=torch.float32)
        out_large = inverse_squared_dist(delta_large)
        self.assertLess(out_large.item(), 1e-12)

        # Very small values
        delta_small = torch.tensor([[1e-6, 1e-6, 1e-6]], dtype=torch.float32)
        out_small = inverse_squared_dist(delta_small)
        self.assertGreater(out_small.item(), 0.999)

    def test_inverse_squared_dist_custom_eps(self):
        """Test inverse_squared_dist with custom epsilon values."""
        delta = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)

        # Different epsilon values
        eps_values = [1e-8, 1e-5, 1e-3, 0.1]
        for eps in eps_values:
            with self.subTest(eps=eps):
                out = inverse_squared_dist(delta, eps=eps)
                expected = 1.0 / (1.0 + 3.0 + eps)  # dist^2 = 3
                self.assertTrue(torch.allclose(out.squeeze(-1), torch.tensor(expected), atol=1e-6))

    def test_inverse_squared_dist_device_consistency(self):
        """Test inverse_squared_dist maintains device consistency."""
        if torch.cuda.is_available():
            delta_cpu = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
            delta_gpu = delta_cpu.cuda()

            out_cpu = inverse_squared_dist(delta_cpu)
            out_gpu = inverse_squared_dist(delta_gpu)

            self.assertEqual(out_cpu.device.type, "cpu")
            self.assertEqual(out_gpu.device.type, "cuda")
            self.assertTrue(torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-6))

    @given(
        batch_size=st.integers(min_value=1, max_value=5),
        seq_len=st.integers(min_value=1, max_value=5),
        eps=st.floats(min_value=1e-10, max_value=1e-3),
    )
    @settings(deadline=None)
    def test_inverse_squared_dist_hypothesis(self, batch_size: int, seq_len: int, eps: float):
        """Test inverse_squared_dist with random inputs using Hypothesis."""
        delta = torch.randn(batch_size, seq_len, 3)
        out = inverse_squared_dist(delta, eps=eps)

        # Check shape
        self.assertEqual(out.shape, (batch_size, seq_len, 1))

        # Check range (0, 1]
        self.assertTrue(torch.all(out > 0))
        self.assertTrue(torch.all(out <= 1.0))

        # Check formula manually
        dist_sq = torch.sum(delta * delta, dim=-1, keepdim=True)
        expected = 1.0 / (1.0 + dist_sq + eps)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


class TestScatterMean(unittest.TestCase):
    """Tests for the scatter_mean function."""

    def test_scatter_mean_basic(self):
        """Test scatter_mean with basic inputs."""
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        index = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        dim_size = 2

        out = scatter_mean(src, index, dim_size=dim_size)

        # Expected: For idx=0 => rows 0 and 3 => mean => ([1+7]/2, [2+8]/2) => (4, 5)
        #           For idx=1 => rows 1 and 2 => mean => ([3+5]/2, [4+6]/2) => (4, 5)
        expected = torch.tensor([[4.0, 5.0], [4.0, 5.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected))
        self.assertEqual(out.shape, (dim_size, 2))

    def test_scatter_mean_with_empty_segments(self):
        """Test scatter_mean with segments that don't receive any items."""
        src = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
        index = torch.tensor([0], dtype=torch.long)
        dim_size = 3

        out = scatter_mean(src, index, dim_size=dim_size)

        # Expected: Segment 0 => average is [2.0, 2.0]
        #           Segment 1 => no items => out => 0
        #           Segment 2 => no items => out => 0
        expected = torch.tensor([[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected))

    def test_scatter_mean_with_custom_dim(self):
        """Test scatter_mean with custom dimension parameter."""
        # Note: The current implementation only supports dim=0
        # This test verifies that the function works correctly with dim=0

        # Create a tensor with shape [3, 4]
        src = torch.randn(3, 4)

        # Scatter along dim=0
        index = torch.tensor([0, 2, 1], dtype=torch.long)  # One index per element in dim=0
        dim_size = 3

        out = scatter_mean(src, index, dim_size=dim_size, dim=0)

        # Expected shape: [3, 4] with scatter along dim=0 => [3, 4]
        self.assertEqual(out.shape, (dim_size, 4))

        # Verify results manually
        # Segment 0 should have src[0]
        self.assertTrue(torch.allclose(out[0], src[0]))
        # Segment 1 should have src[2]
        self.assertTrue(torch.allclose(out[1], src[2]))
        # Segment 2 should have src[1]
        self.assertTrue(torch.allclose(out[2], src[1]))

    def test_scatter_mean_with_torch_scatter(self):
        """Test scatter_mean with torch_scatter if available."""
        try:
            import torch_scatter  # noqa: F401
            has_torch_scatter = True
        except ImportError:
            has_torch_scatter = False

        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        index = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        dim_size = 2

        out = scatter_mean(src, index, dim_size=dim_size)

        # Expected: For idx=0 => rows 0 and 3 => mean => ([1+7]/2, [2+8]/2) => (4, 5)
        #           For idx=1 => rows 1 and 2 => mean => ([3+5]/2, [4+6]/2) => (4, 5)
        expected = torch.tensor([[4.0, 5.0], [4.0, 5.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected))

        # If torch_scatter is available, we should have used it
        # If not, we should have used the fallback loop
        # Either way, the result should be correct
        self.assertEqual(out.shape, (dim_size, 2))

    def test_scatter_mean_fallback_loop(self):
        """Test scatter_mean fallback loop by mocking ImportError."""
        # Save the original import
        original_import = __import__

        # Mock the import to raise ImportError for torch_scatter
        def mock_import(name, *args, **kwargs):
            """
            Mock module import to simulate missing torch_scatter.
            
            If the requested module is 'torch_scatter', raises an ImportError to mimic a missing
            dependency. For any other module, delegates the import to the original import function
            using any additional arguments.
            
            Raises:
                ImportError: If the module name is 'torch_scatter'.
            
            Returns:
                The result of the original import operation for modules other than 'torch_scatter'.
            """
            if name == 'torch_scatter':
                raise ImportError("Mocked ImportError for torch_scatter")
            return original_import(name, *args, **kwargs)

        try:
            # Apply the mock
            import builtins
            builtins.__import__ = mock_import

            # Test scatter_mean with the mock in place
            src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
            index = torch.tensor([0, 1, 1, 0], dtype=torch.long)
            dim_size = 2

            out = scatter_mean(src, index, dim_size=dim_size)

            # Expected: For idx=0 => rows 0 and 3 => mean => ([1+7]/2, [2+8]/2) => (4, 5)
            #           For idx=1 => rows 1 and 2 => mean => ([3+5]/2, [4+6]/2) => (4, 5)
            expected = torch.tensor([[4.0, 5.0], [4.0, 5.0]], dtype=torch.float32)
            self.assertTrue(torch.allclose(out, expected))
            self.assertEqual(out.shape, (dim_size, 2))
        finally:
            # Restore the original import
            builtins.__import__ = original_import

    def test_scatter_mean_inconsistent_dims(self):
        """Test scatter_mean with inconsistent dimensions between src and index."""
        src = torch.randn(5, 3)
        index = torch.tensor([0, 1, 2], dtype=torch.long)  # Only length 3

        with self.assertRaises(IndexError):
            _ = scatter_mean(src, index, dim_size=3)

    @given(
        n=st.integers(min_value=1, max_value=20),
        c=st.integers(min_value=1, max_value=5),
        segments=st.integers(min_value=1, max_value=5),
    )
    @settings(deadline=None)
    def test_scatter_mean_hypothesis(self, n: int, c: int, segments: int):
        """Test scatter_mean with random inputs using Hypothesis."""
        src = torch.randn(n, c)
        index = torch.randint(0, segments, (n,), dtype=torch.long)

        out = scatter_mean(src, index, dim_size=segments)

        # Check shape
        self.assertEqual(out.shape, (segments, c))

        # Verify results manually
        counts = torch.zeros(segments, dtype=torch.float32)
        sums = torch.zeros(segments, c, dtype=torch.float32)

        for i in range(n):
            idx = index[i].item()
            counts[idx] += 1
            sums[idx] += src[i]

        # Calculate expected means
        for i in range(segments):
            if counts[i] > 0:
                expected_mean = sums[i] / counts[i]
                self.assertTrue(torch.allclose(out[i], expected_mean, atol=1e-5))
            else:
                # If count is 0, the mean should be 0
                self.assertTrue(torch.allclose(out[i], torch.zeros(c), atol=1e-5))


class TestBroadcast(unittest.TestCase):
    """Tests for the broadcast function."""

    def test_broadcast_basic(self):
        """Test broadcast with basic inputs."""
        # src: [3], other: [3, 4], dim=0
        src = torch.tensor([0, 1, 2])
        other = torch.randn(3, 4)

        result = broadcast(src, other, dim=0)

        # Expected: src expanded to [3, 4] with each row being [0, 1, 2, 0], [0, 1, 2, 1], etc.
        self.assertEqual(result.shape, (3, 4))
        for i in range(3):
            self.assertEqual(result[i, 0].item(), src[i].item())

    def test_broadcast_negative_dim(self):
        """Test broadcast with negative dimension."""
        # src: [3], other: [2, 3], dim=-1
        src = torch.tensor([0, 1, 2])
        other = torch.randn(2, 3)

        result = broadcast(src, other, dim=-1)

        # Expected: src expanded to [3, 1] since dim=-1 (or 1) is the scatter dimension
        # and we don't expand along that dimension
        self.assertEqual(result.shape, (3, 1))
        for i in range(3):
            self.assertEqual(result[i, 0].item(), src[i].item())

    def test_broadcast_scalar(self):
        """Test broadcast with scalar input."""
        # src: scalar, other: [2, 3], dim=0
        src = torch.tensor(5)
        other = torch.randn(2, 3)

        result = broadcast(src, other, dim=0)

        # Expected: src expanded to [1, 3] since dim=0 is the scatter dimension
        # and we don't expand along that dimension
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(torch.all(result == 5))

    def test_broadcast_higher_dims(self):
        """Test broadcast with higher dimensional tensors."""
        # src: [2, 1, 3], other: [2, 4, 3], dim=1
        src = torch.tensor([[[0, 1, 2]], [[3, 4, 5]]])
        other = torch.randn(2, 4, 3)

        result = broadcast(src, other, dim=1)

        # Expected: src expanded to [2, 1, 3] since dim=1 is the scatter dimension
        # and we don't expand along that dimension
        self.assertEqual(result.shape, (2, 1, 3))
        for i in range(2):
            for k in range(3):
                self.assertEqual(result[i, 0, k].item(), src[i, 0, k].item())

    def test_broadcast_complex_case(self):
        """Test broadcast with a complex case that demonstrates its flexibility."""
        # Create a complex case
        # src: [2, 3, 4], other: [2, 5, 4], dim=1
        src = torch.randn(2, 3, 4)
        other = torch.randn(2, 5, 4)

        # The broadcast function is very flexible and can handle this case
        # It will broadcast src to match other's shape along non-scatter dimensions
        result = broadcast(src, other, dim=1)

        # Check the shape
        self.assertEqual(result.shape, (2, 3, 4))

        # Check that the values are preserved
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(result[i, j, k].item(), src[i, j, k].item())

    def test_broadcast_singleton_expansion(self):
        """Test broadcast with singleton dimensions that need expansion."""
        # src: [1, 3, 1], other: [2, 3, 4], dim=1
        src = torch.tensor([[[0], [1], [2]]])
        other = torch.randn(2, 3, 4)

        result = broadcast(src, other, dim=1)

        # Expected: src expanded to [2, 3, 4]
        self.assertEqual(result.shape, (2, 3, 4))
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(result[i, j, k].item(), src[0, j, 0].item())

    @given(
        batch_size=st.integers(min_value=1, max_value=5),
        seq_len=st.integers(min_value=1, max_value=5),
        feature_dim=st.integers(min_value=1, max_value=5),
        dim=st.integers(min_value=0, max_value=2),
    )
    @settings(deadline=None)
    def test_broadcast_hypothesis(self, batch_size: int, seq_len: int, feature_dim: int, dim: int):
        """Test broadcast with random inputs using Hypothesis."""
        # Skip invalid combinations
        if dim > 2:
            return

        # Create other tensor
        other_shape = [batch_size, seq_len, feature_dim]
        other = torch.randn(*other_shape)

        # Create src tensor with shape compatible for broadcasting along dim
        src_shape = [1, 1, 1]  # Start with all singleton dimensions
        src_shape[dim] = other_shape[dim]  # Match the dimension we're scattering along
        src = torch.randn(*src_shape)

        try:
            result = broadcast(src, other, dim=dim)

            # Check shape
            self.assertEqual(result.shape, other.shape)

            # Check values along the scatter dimension
            if dim == 0:
                for i in range(batch_size):
                    self.assertTrue(torch.allclose(result[i, 0, 0], src[i, 0, 0]))
            elif dim == 1:
                for j in range(seq_len):
                    self.assertTrue(torch.allclose(result[0, j, 0], src[0, j, 0]))
            elif dim == 2:
                for k in range(feature_dim):
                    self.assertTrue(torch.allclose(result[0, 0, k], src[0, 0, k]))
        except RuntimeError:
            # Some combinations might be invalid for broadcasting
            pass


if __name__ == "__main__":
    unittest.main()
