"""
===============================================================================
 Comprehensive Test Suite for `utils.py`
===============================================================================

This module contains an extensive suite of unittests for the `utils.py` module.
It integrates lessons learned from multiple versions:

1. **Organization by Function**:
   Each tested function in `utils.py` has its own test class, so you can quickly 
   locate all relevant tests.

2. **Shared Fixture**:
   A `BaseUtilsTest` class sets up commonly used tensors, dictionaries, and masks 
   to avoid duplication in test code.

3. **Mocking External Calls**:
   We mock `scipy.spatial.transform.Rotation.random` in `TestUniformRandomRotation` 
   to ensure we can isolate its behavior and verify the usage in `uniform_random_rotation`.

4. **Hypothesis-Based Testing**:
   Critical numeric functions (e.g., `centre_random_augmentation`, `rot_vec_mul`) 
   are tested with Hypothesis. We generate random shapes, random numeric ranges, etc.
   This approach helps us uncover edge cases that typical "fixed-value" tests might miss.

5. **Detailed Docstrings**:
   Each test class and method includes docstrings describing the purpose and 
   methodology, making it suitable for maintainers who want to understand the 
   rationale behind each test.

6. **Running and Coverage**:
   Use `python -m unittest test_utils.py` to run all tests. For coverage, install 
   `coverage` (e.g., `pip install coverage`) and then:
     coverage run -m unittest test_utils.py
     coverage report -m

   This should provide a coverage metric indicating how much of `utils.py` is exercised.

7. **Extendability**:
   Add new test methods (or entire classes) as `utils.py` grows. If a new function 
   is introduced, create a `TestNewFunctionName` class and follow the same pattern.

Enjoy the fully integrated test suite that aims for a thorough and robust 
verification of the `utils.py` functionalities.

===============================================================================
"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn as nn

# Hypothesis & related imports
from hypothesis import given, strategies as st, example
from hypothesis.strategies import floats, integers
from hypothesis.extra.numpy import arrays

# NOTE: Adjust this import if needed, e.g.:
from rna_predict.pipeline.stageA.input_embedding.current import utils
#import utils


class BaseUtilsTest(unittest.TestCase):
    """
    Base class providing shared setup data (e.g., small coordinate tensors, masks).
    Inheritors can reference self.coords, self.mask, etc. for convenience.
    """
    def setUp(self):
        """
        Common fixture creation. Called before each test method:
          - self.coords: A small 3-atom coordinate array for geometry tests.
          - self.mask: A mask that excludes the middle atom.
          - self.lower_bins / self.upper_bins: Bins for one_hot tests.
          - self.dict_list: A sample list of dictionaries for merging metrics.
          - self.x_token / self.atom_to_token_idx: For broadcast/aggregate tests.
          - self.msa_feat_dict / self.dim_dict: For sampling MSA features.
        """
        self.coords = torch.tensor(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]], dtype=torch.float
        )
        self.mask = torch.tensor([1, 0, 1], dtype=torch.float)

        # Basic bins for one_hot tests
        self.lower_bins = torch.tensor([-1.0, 3.0], dtype=torch.float)
        self.upper_bins = torch.tensor([1.0, 5.0], dtype=torch.float)

        # Example dictionary list for merging
        self.dict_list = [
            {"loss": 0.5, "accuracy": 0.8},
            {"loss": 0.4, "accuracy": 0.85},
        ]

        # Simple token embedding of shape [1, 3, 2]
        self.x_token = torch.tensor(
            [[[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]], dtype=torch.float
        )
        # Each atom i -> token i
        self.atom_to_token_idx = torch.tensor([0, 1, 2], dtype=torch.long)

        # MSA-like dictionary for sampling
        self.n_msa = 5
        self.msa_feat_dict = {
            "msa": torch.randn(self.n_msa, 10),
            "xyz": torch.randn(self.n_msa, 3),
        }
        self.dim_dict = {"msa": 0, "xyz": 0}


class TestCentreRandomAugmentation(BaseUtilsTest):
    """
    Tests for the `centre_random_augmentation` function in utils.
    This function:
      - Centers coordinates around mean or masked mean.
      - Optionally applies random rotations/translations if centre_only=False.
    """

    def test_centre_only(self):
        """
        Verify that with centre_only=True, we simply shift coords so their mean is ~ 0.
        """
        out = utils.centre_random_augmentation(self.coords, centre_only=True)
        # Expect shape => (1, N_atom=3, 3)
        self.assertEqual(out.shape, (1, 3, 3))
        means = out[0].mean(dim=0)
        for val in means:
            self.assertAlmostEqual(val.item(), 0.0, places=5)

    def test_centre_with_mask(self):
        """
        Check that masking excludes certain atoms from the center calculation.
        """
        out = utils.centre_random_augmentation(self.coords, centre_only=True, mask=self.mask)
        # Expect (1,3,3)
        self.assertEqual(out.shape, (1, 3, 3))

        # Only atoms [0] and [2] used => average is (4,5,6).
        expected = torch.tensor([[-3., -3., -3.],
                                 [ 0.,  0.,  0.],
                                 [ 3.,  3.,  3.]])
        self.assertTrue(torch.allclose(out[0], expected, atol=1e-5))

    @given(coords=arrays(dtype=np.float32, shape=(5, 3), elements=floats(-100, 100)))
    @example(coords=np.array([[0,0,0],[1,1,1],[5,5,5],[5,5,5],[2,2,2]], dtype=np.float32))
    def test_random_augmentation_hypothesis(self, coords):
        """
        Fuzz test using Hypothesis. We create random Nx3 coords,
        call centre_random_augmentation, ensuring shape correctness and no crash.
        """
        tensor_coords = torch.from_numpy(coords)
        # e.g. N_sample=2 => shape => [2, N, 3]
        out = utils.centre_random_augmentation(tensor_coords, N_sample=2, centre_only=False)
        self.assertEqual(out.shape, (2, coords.shape[0], 3))


class TestUniformRandomRotation(BaseUtilsTest):
    """
    Tests for `uniform_random_rotation` in utils.
    This function generates random 3x3 rotation matrices (orthonormal, determinant ~1).
    """

    def test_basic_shape(self):
        """
        With N_sample=3, expect shape (3,3,3).
        """
        mats = utils.uniform_random_rotation(N_sample=3)
        self.assertEqual(mats.shape, (3, 3, 3))

    def test_orthonormal_and_det(self):
        """
        Verify each matrix is orthonormal (R@R^T=I) and determinant is close to 1.
        """
        N = 5
        mats = utils.uniform_random_rotation(N_sample=N)
        for i in range(N):
            R = mats[i]
            # Check orthogonality
            I_approx = R @ R.transpose(-1, -2)
            self.assertTrue(torch.allclose(I_approx, torch.eye(3), atol=1e-5))
            # Check determinant
            det = torch.det(R)
            self.assertAlmostEqual(det.item(), 1.0, places=4)

    @patch("utils.Rotation.random")
    def test_mock_rotation_call(self, mock_rotation):
        """
        Mock the underlying call to Rotation.random(num=N_sample) to test 
        that we pass correct args and properly handle the result.
        """
        class FakeRotation:
            def as_matrix(self):
                return np.array([[[1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]
                                  ]
                                ]
                ).astype(np.float32)

        mock_rotation.return_value = FakeRotation()
        out = utils.uniform_random_rotation(N_sample=1)
        self.assertEqual(out.shape, (1,3,3))
        self.assertTrue(torch.allclose(out[0], torch.eye(3), atol=1e-5))
        mock_rotation.assert_called_once_with(num=1)


class TestRotVecMul(BaseUtilsTest):
    """
    Tests for `rot_vec_mul` in utils.
    Applies a rotation matrix ([...,3,3]) to a set of vectors ([...,3]).
    """

    def test_identity(self):
        """
        Rotating coords by identity should yield the same coords.
        """
        identity = torch.eye(3).unsqueeze(0)  # shape => [1,3,3]
        out = utils.rot_vec_mul(identity, self.coords.unsqueeze(0))
        self.assertTrue(torch.allclose(out[0], self.coords, atol=1e-5))

    def test_90_deg_z(self):
        """
        Rotate (1,0,0) by 90 deg around z => (0,1,0).
        """
        rot_z_90 = torch.tensor([[0, -1, 0],
                                 [1,  0, 0],
                                 [0,  0, 1]], dtype=torch.float).unsqueeze(0)
        point = torch.tensor([[1.,0.,0.]])
        out = utils.rot_vec_mul(rot_z_90, point)
        expected = torch.tensor([[0.,1.,0.]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    @given(batch=integers(min_value=1, max_value=5))
    def test_various_shapes_with_hypothesis(self, batch):
        """
        Random shapes: (batch,3,3) rotation, (batch,3) coords => output => (batch,3).
        """
        r = torch.randn(batch,3,3)
        t = torch.randn(batch,3)
        out = utils.rot_vec_mul(r, t)
        self.assertEqual(out.shape, (batch, 3))


class TestPermuteFinalDims(BaseUtilsTest):
    """
    Tests for `permute_final_dims`.
    This reorders the final dimensions of a tensor as per the provided indices.
    """

    def test_swap_last_two(self):
        """
        For shape [2,3,4], swapping last dims => [2,4,3].
        """
        x = torch.randn(2,3,4)
        out = utils.permute_final_dims(x, [1,0])
        self.assertEqual(out.shape, (2,4,3))

    def test_noop_permutation(self):
        """
        If we permute final dims [0,1], it does nothing to shape [2,3,4].
        """
        x = torch.randn(2,3,4)
        out = utils.permute_final_dims(x, [0,1])
        self.assertTrue(torch.allclose(x, out))


class TestFlattenFinalDims(BaseUtilsTest):
    """
    Tests for `flatten_final_dims`.
    This flattens the last `num_dims` dimensions into a single dimension.
    """

    def test_flatten_last_two(self):
        """
        [2,3,4] => flatten final 2 => shape => [2, 12].
        """
        x = torch.randn(2,3,4)
        out = utils.flatten_final_dims(x, 2)
        self.assertEqual(out.shape, (2, 12))

    def test_flatten_one_dim(self):
        """
        Flattening the last single dimension is effectively a no-op in shape.
        """
        x = torch.randn(2,3)
        out = utils.flatten_final_dims(x, 1)
        self.assertTrue(torch.allclose(x, out))


class TestOneHot(BaseUtilsTest):
    """
    Tests for `one_hot`.
    This uses `lower_bins` and `upper_bins` to produce a one-hot embedding.
    """

    def test_basic_case(self):
        """
        If x => [-2,0,4,6], bins => [-1,3],[1,5], check correctness.
        """
        x = torch.tensor([-2., 0., 4., 6.], dtype=torch.float)
        out = utils.one_hot(x, self.lower_bins, self.upper_bins)
        expected = torch.tensor([[0.,0.],
                                 [1.,0.],
                                 [0.,1.],
                                 [0.,0.]])
        self.assertTrue(torch.allclose(out, expected))


class TestBatchedGather(BaseUtilsTest):
    """
    Tests for `batched_gather`.
    Gathers data from a specified dimension using provided indices.
    """

    def test_1d_simple(self):
        """
        Gather from 1D data => dim=0 => standard usage.
        """
        data = torch.tensor([10,20,30,40], dtype=torch.float)
        inds = torch.tensor([1,3], dtype=torch.long)
        out = utils.batched_gather(data, inds, dim=0, no_batch_dims=0)
        self.assertTrue(torch.allclose(out, torch.tensor([20.,40.])))

    def test_2d_batched(self):
        """
        If data is 2D => shape [B,K], and inds is [B,N], gather from dim=1.
        """
        data = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float)
        inds = torch.tensor([[0,2],[1,0]])
        out = utils.batched_gather(data, inds, dim=1, no_batch_dims=1)
        # row0 => gather [0,2] => [1,3]; row1 => gather [1,0] => [5,4]
        expected = torch.tensor([[1,3],[5,4]], dtype=torch.float)
        self.assertTrue(torch.allclose(out, expected))


class TestBroadcastTokenToAtom(BaseUtilsTest):
    """
    Tests for `broadcast_token_to_atom`.
    This expands token-level embeddings to per-atom embeddings 
    according to a mapping tensor.
    """

    def test_basic_broadcast(self):
        """
        If each atom i => token i, expect the broadcast to replicate x_token's rows 
        in the corresponding atom positions.
        """
        out = utils.broadcast_token_to_atom(self.x_token, self.atom_to_token_idx)
        self.assertEqual(out.shape, (1,3,2))
        self.assertTrue(torch.allclose(out, self.x_token))

    def test_round_trip(self):
        """
        Round-trip check: broadcast => aggregate => x_token 
        (when each atom is uniquely mapped).
        """
        x_atom = utils.broadcast_token_to_atom(self.x_token, self.atom_to_token_idx)
        x_token_2 = utils.aggregate_atom_to_token(
            x_atom, self.atom_to_token_idx, n_token=3, reduce="mean"
        )
        self.assertTrue(torch.allclose(x_token_2, self.x_token))


class TestAggregateAtomToToken(BaseUtilsTest):
    """
    Tests for `aggregate_atom_to_token`.
    This aggregates per-atom embeddings back into per-token embeddings 
    via sum/mean or other strategies.
    """

    def test_basic_mean(self):
        """
        If each atom i => token i => the output matches the original coords.
        """
        out = utils.aggregate_atom_to_token(
            self.coords.unsqueeze(0), self.atom_to_token_idx, n_token=3, reduce="mean"
        )
        self.assertEqual(out.shape, (1,3,3))
        self.assertTrue(torch.allclose(out[0], self.coords))

    def test_sum_aggregation(self):
        """
        If all atoms map to the same token=0 => we get sum of all coords in that slot.
        """
        all_zeros = torch.zeros_like(self.atom_to_token_idx)
        out = utils.aggregate_atom_to_token(
            self.coords.unsqueeze(0), all_zeros, n_token=3, reduce="sum"
        )
        # sum => (1+4+7, 2+5+8, 3+6+9) => (12,15,18)
        exp = torch.tensor([12.,15.,18.], dtype=torch.float)
        self.assertTrue(torch.allclose(out[0,0], exp, atol=1e-5))
        self.assertTrue(torch.allclose(out[0,1], torch.zeros(3)))
        self.assertTrue(torch.allclose(out[0,2], torch.zeros(3)))


class TestSampleIndices(BaseUtilsTest):
    """
    Tests for `sample_indices`.
    This function returns random or 'topk' indices from [0..n-1].
    """

    def test_random_strategy(self):
        """
        'random' => expects valid subset of [0..n-1].
        """
        idx = utils.sample_indices(n=5, strategy="random")
        self.assertTrue(0 <= idx.min().item() < 5)
        self.assertTrue(0 <= idx.max().item() < 5)

    def test_topk_strategy(self):
        """
        'topk' => expects first k indices in ascending order.
        """
        idx = utils.sample_indices(n=5, strategy="topk")
        # length can be in [1..5], but if not empty, first is 0
        if idx.numel() > 0:
            self.assertEqual(idx[0].item(), 0)

    def test_bad_strategy(self):
        """
        Invalid strategy => triggers an assertion error.
        """
        with self.assertRaises(AssertionError):
            utils.sample_indices(n=5, strategy="unknown")


class TestSampleMsaFeatureDictRandomWithoutReplacement(BaseUtilsTest):
    """
    Tests for `sample_msa_feature_dict_random_without_replacement`.
    This samples MSA features for data augmentation.
    """

    def test_basic_usage(self):
        """
        With cutoff=3, check shapes do not exceed 3 in dimension 0 
        and remain consistent in feature dims.
        """
        out = utils.sample_msa_feature_dict_random_without_replacement(
            self.msa_feat_dict, self.dim_dict, cutoff=3, lower_bound=1, strategy="random"
        )
        self.assertIn("msa", out)
        self.assertIn("xyz", out)
        self.assertLessEqual(out["msa"].shape[0], 3)
        self.assertLessEqual(out["xyz"].shape[0], 3)
        self.assertEqual(out["msa"].shape[1], 10)
        self.assertEqual(out["xyz"].shape[1], 3)


class TestExpandAtDim(BaseUtilsTest):
    """
    Tests for `expand_at_dim`.
    This function inserts a new dimension at `dim` and expands it `n` times.
    """

    def test_basic_expand(self):
        """
        [3] => expand at dim=0 by n=2 => shape => [2,3].
        """
        x = torch.tensor([1.,2.,3.])
        out = utils.expand_at_dim(x, dim=0, n=2)
        self.assertEqual(out.shape, (2,3))
        self.assertTrue(torch.allclose(out[0], x))
        self.assertTrue(torch.allclose(out[1], x))


class TestPadAtDim(BaseUtilsTest):
    """
    Tests for `pad_at_dim`.
    This pads a tensor at a specified dimension on both sides (left/right).
    """

    def test_pad_front_and_back(self):
        """
        [3] => pad (1 left, 2 right) => total length => 6.
        """
        x = torch.tensor([10.,20.,30.])
        out = utils.pad_at_dim(x, dim=0, pad_length=(1,2), value=-1)
        exp = torch.tensor([-1.,10.,20.,30.,-1.,-1.])
        self.assertTrue(torch.allclose(out, exp))


class TestReshapeAtDim(BaseUtilsTest):
    """
    Tests for `reshape_at_dim`.
    Reshape a single dimension into a new shape tuple.
    """

    def test_basic_reshape(self):
        """
        [2,3,4,5] => reshape dimension -2 => new shape => [12], => [2, 12, 5].
        """
        x = torch.randn(2,3,4,5)
        out = utils.reshape_at_dim(x, dim=-2, target_shape=[12])
        self.assertEqual(out.shape, (2,12,5))


class TestMoveFinalDimToDim(BaseUtilsTest):
    """
    Tests for `move_final_dim_to_dim`.
    Moves the final dimension to a user-specified dimension index.
    """

    def test_basic_move(self):
        """
        [2,3,4], move final dim => dim=1 => shape => [2,4,3].
        """
        x = torch.randn(2,3,4)
        out = utils.move_final_dim_to_dim(x, dim=1)
        self.assertEqual(out.shape, (2,4,3))


class TestSimpleMergeDictList(BaseUtilsTest):
    """
    Tests for `simple_merge_dict_list`.
    This merges lists of dictionaries by concatenating numeric values.
    """

    def test_merge(self):
        """
        Basic scenario => merges numeric data into arrays for each key.
        """
        merged = utils.simple_merge_dict_list(self.dict_list)
        self.assertIn("loss", merged)
        self.assertIn("accuracy", merged)
        self.assertEqual(merged["loss"].shape, (2,))
        self.assertEqual(merged["accuracy"].shape, (2,))
        self.assertAlmostEqual(merged["loss"][0], 0.5)
        self.assertAlmostEqual(merged["loss"][1], 0.4)
        self.assertAlmostEqual(merged["accuracy"][0], 0.8)
        self.assertAlmostEqual(merged["accuracy"][1], 0.85)

    def test_unsupported_type(self):
        """
        If we pass an unsupported type (like a string), we expect a ValueError.
        """
        with self.assertRaises(ValueError):
            utils.simple_merge_dict_list([{"unexpected": "string_value"}])


if __name__ == "__main__":
    # Run all tests: python -m unittest test_utils.py
    unittest.main()