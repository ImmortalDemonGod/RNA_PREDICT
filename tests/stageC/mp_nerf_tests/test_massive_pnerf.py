import unittest
from unittest.mock import patch

import numpy as np
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rna_predict.pipeline.stageC.mp_nerf import massive_pnerf

# =============================================================================
# Helper Strategies for Hypothesis Tests
# =============================================================================
# 'point3d' generates a list of 3 floats (representing 3D coordinates) and then maps them to a torch.Tensor.
point3d = st.lists(
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=3,
).map(lambda coords: torch.tensor(coords, dtype=torch.float32))

# 'angle' generates angles within the valid range for trigonometric functions.
angle = st.floats(
    min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False
)

# 'bond_length' generates bond lengths ensuring they are strictly positive and finite.
bond_length = st.floats(
    min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
)


# =============================================================================
# Test Class for get_axis_matrix Function
# =============================================================================
class TestGetAxisMatrix(unittest.TestCase):
    """
    Comprehensive tests for the get_axis_matrix function from massive_pnerf.

    This test suite verifies:
      - Basic functionality using fixed inputs.
      - Correct behavior when normalization is enabled vs. disabled.
      - Robustness under property-based random input testing.

    The get_axis_matrix function is expected to return an orthonormal or unnormalized
    basis (depending on the 'norm' flag) constructed from three input points.
    """

    def setUp(self):
        # Initialize fixed vectors for baseline testing
        # Use vectors that do NOT initially form an orthonormal basis
        self.a = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.b = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
        self.c = torch.tensor([2.0, 3.0, 0.0], dtype=torch.float32)

    def test_normalization_true(self):
        """
        Test that when normalization is enabled (norm=True), get_axis_matrix returns a basis
        where each vector has a unit length.
        """
        basis = massive_pnerf.get_axis_matrix(self.a, self.b, self.c, norm=True)
        self.assertEqual(basis.shape, (3, 3), "Expected basis to be of shape (3,3)")
        for idx, vec in enumerate(basis):
            norm_val = torch.norm(vec).item()
            self.assertAlmostEqual(
                norm_val,
                1.0,
                places=5,
                msg=f"Vector at index {idx} not normalized (norm = {norm_val})",
            )

    def test_normalization_false(self):
        """
        Test that when normalization is disabled (norm=False), the output basis is not normalized.
        The result should differ from the normalized version.
        """
        basis_norm = massive_pnerf.get_axis_matrix(self.a, self.b, self.c, norm=True)
        basis_no_norm = massive_pnerf.get_axis_matrix(
            self.a, self.b, self.c, norm=False
        )
        # Ensure the normalized and unnormalized bases differ significantly.
        self.assertFalse(
            torch.allclose(basis_norm, basis_no_norm, atol=1e-5),
            "The normalized and unnormalized bases should differ when norm is False.",
        )

    @given(a=point3d, b=point3d, c=point3d, norm=st.booleans())
    def test_hypothesis_axis_matrix(self, a, b, c, norm):
        """
        Property-based test for get_axis_matrix using randomly generated 3D points.

        The test ensures:
          - The function returns a (3, 3) tensor.
          - For non-collinear inputs, if normalization is True, each vector has unit length.
        """
        # Ensure that the input points are non-collinear by checking the cross product norm.
        v1 = c - b
        v2 = b - a
        cp = torch.cross(v1, v2)
        assume(torch.norm(cp) > 1e-3)
        basis = massive_pnerf.get_axis_matrix(a, b, c, norm=norm)
        self.assertEqual(
            basis.shape, (3, 3), "Hypothesis test: Basis should be of shape (3,3)"
        )
        if norm:
            for idx, vec in enumerate(basis):
                self.assertAlmostEqual(
                    torch.norm(vec).item(),
                    1.0,
                    places=5,
                    msg=f"Hypothesis test: Vector {idx} not unit norm when normalized.",
                )


# =============================================================================
# Test Class for mp_nerf_torch Function
# =============================================================================
class TestMpNerfTorch(unittest.TestCase):
    """
    Comprehensive tests for the mp_nerf_torch function from massive_pnerf.

    This test suite includes:
      - Basic functionality and output shape tests using fixed inputs.
      - Consistency tests when parameters are provided as floats versus tensors.
      - Property-based tests covering a range of randomly generated inputs.
      - Tests for degenerate input handling and idempotency.
      - Use of mocks to ensure that internal clamping is applied correctly.
    """

    def setUp(self):
        # Initialize fixed vectors for reproducible testing
        self.a = torch.tensor([0.0, 0.0, 0.0])
        self.b = torch.tensor([1.0, 0.0, 0.0])
        self.c = torch.tensor([1.0, 1.0, 0.0])

    def test_basic_output_shape(self):
        """
        Verify that mp_nerf_torch returns a tensor of shape (3,) for typical fixed inputs.
        """
        params = massive_pnerf.MpNerfParams(self.a, self.b, self.c, 1.0, 0.5, 0.3)
        result = massive_pnerf.mp_nerf_torch(params)
        self.assertEqual(result.shape, (3,), "Output tensor shape should be (3,)")

    def test_consistency_float_tensor(self):
        """
        Ensure that providing bond_length, theta, and chi as floats yields the same result as
        providing them as torch.Tensors.
        """
        l_val, theta_val, chi_val = 2.0, 0.7, 0.2
        params_float = massive_pnerf.MpNerfParams(
            self.a, self.b, self.c, l_val, theta_val, chi_val
        )
        result_float = massive_pnerf.mp_nerf_torch(params_float)

        l_tensor = torch.tensor(l_val, dtype=torch.float32)
        theta_tensor = torch.tensor(theta_val, dtype=torch.float32)
        chi_tensor = torch.tensor(chi_val, dtype=torch.float32)
        params_tensor = massive_pnerf.MpNerfParams(
            self.a, self.b, self.c, l_tensor, theta_tensor, chi_tensor
        )
        result_tensor = massive_pnerf.mp_nerf_torch(params_tensor)

        self.assertTrue(
            torch.allclose(result_float, result_tensor, atol=1e-5),
            "Results should be equivalent whether using float or tensor inputs.",
        )

    @given(
        a=point3d,
        b=point3d,
        c=point3d,
        bond=st.one_of(
            bond_length,
            st.lists(bond_length, min_size=1, max_size=1).map(lambda length_list: length_list[0]),
        ),
        theta=angle,
        chi=angle,
    )
    @settings(max_examples=50)
    def test_hypothesis_mp_nerf(self, a, b, c, bond, theta, chi):
        """
        Property-based test for mp_nerf_torch:
        Verifies that for randomly generated valid 3D points and parameters, the output is a tensor of shape (3,)
        and does not contain any NaN values.
        """
        assume(torch.norm(b - a) > 1e-3 and torch.norm(c - b) > 1e-3)
        params = massive_pnerf.MpNerfParams(a, b, c, bond, theta, chi)
        result = massive_pnerf.mp_nerf_torch(params)
        self.assertEqual(
            result.shape, (3,), "Hypothesis test: Output shape must be (3,)"
        )
        self.assertFalse(
            torch.isnan(result).any(), "Output must not contain NaN values."
        )

    def test_degenerate_vectors(self):
        """
        Test that mp_nerf_torch handles degenerate cases gracefully.
        Specifically, when the difference between input points is nearly zero, the function should
        not crash and must still return a valid tensor of shape (3,).
        """
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([1e-12, 0.0, 0.0])
        c = torch.tensor([0.0, 1.0, 0.0])
        params = massive_pnerf.MpNerfParams(a, b, c, 1.0, 0.0, 0.0)
        result = massive_pnerf.mp_nerf_torch(params)
        self.assertEqual(
            result.shape,
            (3,),
            "Degenerate input should still yield an output shape of (3,)",
        )

    def test_idempotency(self):
        """
        Verify that repeated calls to mp_nerf_torch with the same inputs produce identical outputs,
        ensuring deterministic behavior.
        """
        params = massive_pnerf.MpNerfParams(self.a, self.b, self.c, 1.0, 0.5, 0.3)
        result1 = massive_pnerf.mp_nerf_torch(params)
        result2 = massive_pnerf.mp_nerf_torch(params)
        self.assertTrue(
            torch.allclose(result1, result2, atol=1e-5),
            "Repeated calls with identical inputs should yield the same result.",
        )

    @patch(
        "rna_predict.pipeline.stageC.mp_nerf.massive_pnerf.torch.clamp",
        wraps=torch.clamp,
    )
    def test_theta_clamping(self, mock_clamp):
        """
        Use a mock to verify that when theta values exceed the valid range, mp_nerf_torch uses torch.clamp
        to restrict them to the proper limits.
        """
        params = massive_pnerf.MpNerfParams(self.a, self.b, self.c, 1.0, 4 * np.pi, 0.3)
        result = massive_pnerf.mp_nerf_torch(params)
        mock_clamp.assert_called()
        self.assertEqual(
            result.shape, (3,), "Output shape must be (3,) even when clamping is used."
        )

    def test_near_zero_cb_vector(self):
        """
        Test mp_nerf_torch when the vector c - b has near-zero magnitude.
        This specifically targets the perturbation logic for cb (line 98).
        """
        a = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        # Make c very close to b to trigger the cb_norm < 1e-10 condition
        c = torch.tensor([1.0, 1e-12, 0.0], dtype=torch.float32)
        params = massive_pnerf.MpNerfParams(a, b, c, 1.0, 0.5, 0.3)

        # Ensure the function runs without error and returns the correct shape
        result = massive_pnerf.mp_nerf_torch(params)
        self.assertEqual(
            result.shape,
            (3,),
            "Output tensor shape should be (3,) even with near-zero cb vector.",
        )
        self.assertFalse(
            torch.isnan(result).any(), "Output must not contain NaN values."
        )
        # Optional: Add more specific assertions about the expected output if known


# =============================================================================
# Test Class for scn_rigid_index_mask Function
# =============================================================================
class TestScnRigidIndexMask(unittest.TestCase):
    """
    Comprehensive tests for the scn_rigid_index_mask function from massive_pnerf.

    This test suite checks:
      - Behavior for sequences shorter than the minimum required length.
      - Correct output when using both C-alpha and full-atom (N, CA, C) modes.
      - Property-based testing with various sequence lengths and randomness.
    """

    def test_short_sequence(self):
        """
        Verify that when the input sequence is shorter than 3 characters, the function returns
        an empty tensor.
        """
        seq = "AB"
        result = massive_pnerf.scn_rigid_index_mask(seq, c_alpha=True)
        self.assertEqual(
            result.numel(), 0, "Sequences shorter than 3 should return an empty tensor."
        )

    def test_calpha_true(self):
        """
        Verify that when c_alpha is True, the function returns the expected indices for the first
        three C-alpha atoms.

        For a typical sequence, expected indices are computed as:
          - 0*14+1, 1*14+1, 2*14+1 -> [1, 15, 29]
        """
        seq = "ABCDEFG"
        expected = torch.tensor([1, 15, 29], dtype=torch.long)
        result = massive_pnerf.scn_rigid_index_mask(seq, c_alpha=True)
        self.assertTrue(
            torch.equal(result, expected),
            "C-alpha true should return indices [1, 15, 29] for the first 3 residues.",
        )

    def test_calpha_false(self):
        """
        Verify that when c_alpha is False, the function returns indices corresponding to the
        N, CA, and C atoms of the first residue.

        For a typical sequence, expected indices are:
          - 0*14+0, 0*14+1, 0*14+2 -> [0, 1, 2]
        """
        seq = "ABCDEFG"
        expected = torch.tensor([0, 1, 2], dtype=torch.long)
        result = massive_pnerf.scn_rigid_index_mask(seq, c_alpha=False)
        self.assertTrue(
            torch.equal(result, expected),
            "C-alpha false should return indices [0, 1, 2] for the first residue.",
        )

    @given(seq=st.text(min_size=3, max_size=100), c_alpha=st.booleans())
    def test_hypothesis_rigid_mask(self, seq, c_alpha):
        """
        Property-based test for scn_rigid_index_mask:
        Ensures that for any valid sequence (of length at least 3), the function returns a tensor
        of exactly 3 long integer indices that are non-negative.
        """
        result = massive_pnerf.scn_rigid_index_mask(seq, c_alpha)
        self.assertEqual(
            result.shape[0], 3, "Result tensor should contain exactly 3 indices."
        )
        self.assertEqual(
            result.dtype, torch.long, "Indices must be of type torch.long."
        )
        self.assertTrue((result >= 0).all(), "Indices should be non-negative.")


# =============================================================================
# Main Entry Point for Test Execution
# =============================================================================
if __name__ == "__main__":
    unittest.main()
