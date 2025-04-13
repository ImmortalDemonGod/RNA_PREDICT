"""
Comprehensive tests for geometry_utils.py.

This module provides thorough testing for the geometry utility functions
in the RNA_PREDICT project, focusing on angle conversion and other
geometric operations.
"""

import unittest
from typing import List, Tuple

import numpy as np
import torch
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

from rna_predict.pipeline.stageC.mp_nerf.protein_utils.geometry_utils import (
    to_zero_two_pi,
)


class TestToZeroTwoPi(unittest.TestCase):
    """Tests for the to_zero_two_pi function."""

    def setUp(self):
        """Set up test fixtures."""
        # Define common test cases as (input, expected) pairs
        self.test_cases_float: List[Tuple[float, float]] = [
            (0.0, 0.0),  # Zero remains zero
            (np.pi, np.pi),  # π remains π
            (2 * np.pi, 0.0),  # 2π wraps to 0
            (3 * np.pi, np.pi),  # 3π wraps to π
            (4 * np.pi, 0.0),  # 4π wraps to 0
            (-np.pi, np.pi),  # -π wraps to π
            (-2 * np.pi, 0.0),  # -2π wraps to 0
            (-3 * np.pi, np.pi),  # -3π wraps to π
            (0.5 * np.pi, 0.5 * np.pi),  # π/2 remains π/2
            (2.5 * np.pi, 0.5 * np.pi),  # 2.5π wraps to 0.5π
            (-0.5 * np.pi, 1.5 * np.pi),  # -0.5π wraps to 1.5π
        ]

    def test_float_inputs(self):
        """Test to_zero_two_pi with float inputs."""
        for input_val, expected in self.test_cases_float:
            with self.subTest(input_val=input_val):
                result = to_zero_two_pi(input_val)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, expected, places=10)

    def test_int_inputs(self):
        """Test to_zero_two_pi with integer inputs."""
        # Test with integer inputs (should be converted to float)
        int_test_cases = [
            (0, 0.0),  # Zero remains zero
            (3, 3.0 % (2 * np.pi)),  # 3 radians wraps to its modulo
            (-3, ((-3) % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)),  # -3 radians wraps
        ]
        for input_val, expected in int_test_cases:
            with self.subTest(input_val=input_val):
                result = to_zero_two_pi(input_val)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, expected, places=10)

    def test_numpy_array_inputs(self):
        """Test to_zero_two_pi with numpy array inputs."""
        # Create a numpy array from the test cases
        inputs = np.array([case[0] for case in self.test_cases_float])
        expected = np.array([case[1] for case in self.test_cases_float])
        
        result = to_zero_two_pi(inputs)
        
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_torch_tensor_inputs(self):
        """Test to_zero_two_pi with torch tensor inputs."""
        # Create a torch tensor from the test cases
        inputs = torch.tensor([case[0] for case in self.test_cases_float], dtype=torch.float32)
        expected = torch.tensor([case[1] for case in self.test_cases_float], dtype=torch.float32)
        
        result = to_zero_two_pi(inputs)
        
        self.assertIsInstance(result, torch.Tensor)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_multidimensional_numpy_array(self):
        """Test to_zero_two_pi with multidimensional numpy arrays."""
        # Create a 2D numpy array
        inputs_2d = np.array([
            [0.0, np.pi, 2 * np.pi],
            [-np.pi, -2 * np.pi, 3 * np.pi]
        ])
        expected_2d = np.array([
            [0.0, np.pi, 0.0],
            [np.pi, 0.0, np.pi]
        ])
        
        result_2d = to_zero_two_pi(inputs_2d)
        
        self.assertIsInstance(result_2d, np.ndarray)
        np.testing.assert_allclose(result_2d, expected_2d, rtol=1e-10)

    def test_multidimensional_torch_tensor(self):
        """Test to_zero_two_pi with multidimensional torch tensors."""
        # Create a 2D torch tensor
        inputs_2d = torch.tensor([
            [0.0, np.pi, 2 * np.pi],
            [-np.pi, -2 * np.pi, 3 * np.pi]
        ], dtype=torch.float32)
        expected_2d = torch.tensor([
            [0.0, np.pi, 0.0],
            [np.pi, 0.0, np.pi]
        ], dtype=torch.float32)
        
        result_2d = to_zero_two_pi(inputs_2d)
        
        self.assertIsInstance(result_2d, torch.Tensor)
        torch.testing.assert_close(result_2d, expected_2d, rtol=1e-5, atol=1e-5)

    def test_unsupported_type(self):
        """Test to_zero_two_pi raises TypeError for unsupported input types."""
        with self.assertRaises(TypeError):
            to_zero_two_pi("not a number")
        
        with self.assertRaises(TypeError):
            to_zero_two_pi([1.0, 2.0, 3.0])  # List is not supported
        
        with self.assertRaises(TypeError):
            to_zero_two_pi((1.0, 2.0))  # Tuple is not supported

    @given(x=st.floats(min_value=-100 * np.pi, max_value=100 * np.pi, allow_nan=False, allow_infinity=False))
    @settings(deadline=None)
    def test_property_output_range(self, x: float):
        """Test that output is always in [0, 2π) range."""
        result = to_zero_two_pi(x)
        self.assertGreaterEqual(result, 0.0)
        self.assertLess(result, 2 * np.pi)

    @given(x=st.floats(min_value=-100 * np.pi, max_value=100 * np.pi, allow_nan=False, allow_infinity=False))
    @settings(deadline=None)
    def test_property_modular_equivalence(self, x: float):
        """Test that input and output are equivalent modulo 2π."""
        result = to_zero_two_pi(x)
        # Check that sin and cos of both angles are the same (modular equivalence)
        self.assertAlmostEqual(np.sin(x), np.sin(result), places=10)
        self.assertAlmostEqual(np.cos(x), np.cos(result), places=10)

    @given(
        x=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=10),
            elements=st.floats(min_value=-100 * np.pi, max_value=100 * np.pi, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(deadline=None)
    def test_property_numpy_array_output_range(self, x: np.ndarray):
        """Test that numpy array output is always in [0, 2π) range."""
        result = to_zero_two_pi(x)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result < 2 * np.pi))

    @given(
        x=st.lists(
            st.floats(min_value=-100 * np.pi, max_value=100 * np.pi, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10,
        )
    )
    @settings(deadline=None)
    def test_property_torch_tensor_output_range(self, x: List[float]):
        """Test that torch tensor output is always in [0, 2π) range."""
        tensor_x = torch.tensor(x, dtype=torch.float32)
        result = to_zero_two_pi(tensor_x)
        self.assertTrue(torch.all(result >= 0.0))
        self.assertTrue(torch.all(result < 2 * np.pi))

    @given(
        x=st.lists(
            st.floats(min_value=-100 * np.pi, max_value=100 * np.pi, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10,
        )
    )
    @settings(deadline=None)
    def test_property_torch_tensor_modular_equivalence(self, x: List[float]):
        """Test that torch tensor input and output are equivalent modulo 2π."""
        tensor_x = torch.tensor(x, dtype=torch.float32)
        result = to_zero_two_pi(tensor_x)
        # Check that sin and cos of both angles are the same (modular equivalence)
        torch.testing.assert_close(torch.sin(tensor_x), torch.sin(result), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(torch.cos(tensor_x), torch.cos(result), rtol=1e-5, atol=1e-5)

    def test_edge_cases(self):
        """Test edge cases for to_zero_two_pi."""
        # Very large positive number
        large_pos = 1000 * np.pi
        result_large_pos = to_zero_two_pi(large_pos)
        self.assertGreaterEqual(result_large_pos, 0.0)
        self.assertLess(result_large_pos, 2 * np.pi)
        
        # Very large negative number
        large_neg = -1000 * np.pi
        result_large_neg = to_zero_two_pi(large_neg)
        self.assertGreaterEqual(result_large_neg, 0.0)
        self.assertLess(result_large_neg, 2 * np.pi)
        
        # Exactly 2π should wrap to 0
        exactly_2pi = 2 * np.pi
        result_exactly_2pi = to_zero_two_pi(exactly_2pi)
        self.assertAlmostEqual(result_exactly_2pi, 0.0, places=10)
        
        # Exactly -2π should wrap to 0
        exactly_neg_2pi = -2 * np.pi
        result_exactly_neg_2pi = to_zero_two_pi(exactly_neg_2pi)
        self.assertAlmostEqual(result_exactly_neg_2pi, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
