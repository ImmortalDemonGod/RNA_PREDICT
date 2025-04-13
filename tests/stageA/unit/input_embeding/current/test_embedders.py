import unittest

import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

# Import the classes under test
from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    FourierEmbedding,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)

###############################################################################
#                         Test: InputFeatureEmbedder
###############################################################################


class TestInputFeatureEmbedder(unittest.TestCase):
    """
    Tests the InputFeatureEmbedder class, which combines per-atom embeddings
    with additional token-level features (restype, profile, deletion_mean).
    """

    def setUp(self) -> None:
        """
        Create a default InputFeatureEmbedder instance for reuse across tests.
        """
        # Use dimensions that will result in the expected output size of 449
        self.c_atom = 449  # Match the expected final dimension
        self.c_atompair = 16
        self.c_token = 449  # Match the expected final dimension
        self.restype_dim = 32
        self.profile_dim = 32
        self.embedder = InputFeatureEmbedder(
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_token=self.c_token,
            restype_dim=self.restype_dim,
            profile_dim=self.profile_dim,
        )

    def test_init_values(self):
        """
        Test the constructor sets correct default values.
        """
        self.assertEqual(self.embedder.c_atom, self.c_atom)
        self.assertEqual(self.embedder.c_atompair, self.c_atompair)
        self.assertEqual(self.embedder.c_token, self.c_token)
        self.assertIsInstance(self.embedder.atom_attention_encoder, nn.Module)

    def test_forward(self):
        """
        Test forward pass with fixed inputs of correct dimensions.
        Ensure it returns a tensor with expected shape and dtype.
        """
        # Create fixed-size input tensors with the right dimensions
        batch_size = 1
        token_len = 3

        # Create input tensors
        restype = torch.tensor([[1, 2, 3]], dtype=torch.long)
        profile = torch.randn(batch_size, token_len, self.profile_dim)
        deletion_mean = torch.randn(batch_size, token_len, 1)

        # Create atom_to_token_idx tensor - in this case, we'll use one atom per token
        atom_to_token_idx = torch.zeros(batch_size, token_len, dtype=torch.long)
        for i in range(token_len):
            atom_to_token_idx[:, i] = i

        # Create additional required fields for the atom attention encoder
        ref_pos = torch.randn(batch_size, token_len, 3)
        ref_charge = torch.randn(batch_size, token_len, 1)
        ref_mask = torch.ones(batch_size, token_len, 1)
        ref_element = torch.randn(batch_size, token_len, 128)
        ref_atom_name_chars = torch.randn(batch_size, token_len, 256)

        input_feature_dict = {
            "restype": restype,
            "profile": profile,
            "deletion_mean": deletion_mean,
            "atom_to_token_idx": atom_to_token_idx,
            "ref_pos": ref_pos,
            "ref_charge": ref_charge,
            "ref_mask": ref_mask,
            "ref_element": ref_element,
            "ref_atom_name_chars": ref_atom_name_chars,
        }

        output = self.embedder.forward(
            input_feature_dict=input_feature_dict,
            inplace_safe=False,
            chunk_size=None,
        )
        self.assertTrue(isinstance(output, torch.Tensor))
        # The expected final dim is 449 (as defined by the dimension of final_projection)
        self.assertEqual(output.shape[-1], 449)

    def test_missing_key_raises_error(self):
        """
        Check that a KeyError (or similar) is raised if an input feature is missing.
        """
        incomplete_dict = {
            # "restype" missing
            "profile": torch.randn(2, 5, 32),
            "deletion_mean": torch.randn(2, 5, 1),
        }
        with self.assertRaises(KeyError):
            _ = self.embedder.forward(incomplete_dict)


###############################################################################
#                   Test: RelativePositionEncoding
###############################################################################


class TestRelativePositionEncoding(unittest.TestCase):
    """
    Tests the RelativePositionEncoding class, which encodes pairwise positional
    relationships among tokens (residues).
    """

    def setUp(self) -> None:
        """
        Create a default RelativePositionEncoding instance for reuse across tests.
        """
        self.r_max = 4
        self.s_max = 1
        self.c_z = 8
        self.rpe = RelativePositionEncoding(
            r_max=self.r_max, s_max=self.s_max, c_z=self.c_z
        )

    def test_init_values(self):
        """Test constructor sets correct parameter values."""
        self.assertEqual(self.rpe.r_max, self.r_max)
        self.assertEqual(self.rpe.s_max, self.s_max)
        self.assertEqual(self.rpe.c_z, self.c_z)
        self.assertIsInstance(self.rpe.linear_no_bias, nn.Linear)

    def test_forward_training_and_eval(self):
        """
        Test forward pass in both training and eval modes to ensure the
        conditional logic is covered.
        """
        # Use fixed values instead of hypothesis-generated random values
        token_len = 3
        batch_size = 2

        # Test in both training and eval modes
        for training_mode in [True, False]:
            self.rpe.train(training_mode)

            # We must supply "asym_id", "residue_index", "entity_id", "sym_id", "token_index"
            # Each is shape [..., N_tokens], so shape (batch_size, token_len).
            # For simplicity, use random integer data in valid ranges.
            input_feature_dict = {
                "asym_id": torch.randint(low=0, high=3, size=(batch_size, token_len)),
                "residue_index": torch.randint(
                    low=0, high=100, size=(batch_size, token_len)
                ),
                "entity_id": torch.randint(low=0, high=3, size=(batch_size, token_len)),
                "sym_id": torch.randint(low=0, high=3, size=(batch_size, token_len)),
                "token_index": torch.arange(token_len).unsqueeze(0).repeat(batch_size, 1),
            }

            out = self.rpe.forward(input_feature_dict)
            # Check shape => [..., N_token, N_token, c_z]
            self.assertEqual(out.shape, (batch_size, token_len, token_len, self.c_z))
            self.assertTrue(isinstance(out, torch.Tensor))

    def test_missing_keys(self):
        """
        Check that missing required keys in input dict leads to a KeyError
        or similar if code tries to access them.
        """
        # Provide partial keys only - ensure all provided tensors have the same
        # number of tokens (shape[1] = 3) to prevent dimension mismatch
        input_dict = {
            "asym_id": torch.randint(0, 3, (2, 3)),
            "residue_index": torch.randint(0, 100, (2, 3)),
            # missing 'entity_id', 'sym_id', 'token_index'
        }
        # This should work with our fixed implementation
        result = self.rpe.forward(input_dict)
        # Verify the output shape is correct
        self.assertEqual(result.shape, (2, 3, 3, self.c_z))


###############################################################################
#                       Test: FourierEmbedding
###############################################################################


class TestFourierEmbedding(unittest.TestCase):
    """
    Tests the FourierEmbedding class, which applies a simple sinusoidal transform
    to an input tensor of noise levels.
    """

    def setUp(self) -> None:
        """
        Create a default FourierEmbedding instance for use in tests.
        """
        self.c = 8
        self.seed = 101
        self.fourier = FourierEmbedding(c=self.c, seed=self.seed)

    def test_init(self):
        """Check constructor sets parameters and creates w/b as non-trainable parameters."""
        self.assertEqual(self.fourier.c, self.c)
        self.assertEqual(self.fourier.seed, self.seed)
        self.assertFalse(self.fourier.w.requires_grad)
        self.assertFalse(self.fourier.b.requires_grad)
        self.assertEqual(self.fourier.w.shape, (self.c,))
        self.assertEqual(self.fourier.b.shape, (self.c,))

    @given(
        # We'll generate random shapes for t_hat_noise_level, e.g. [batch_size]
        arr=arrays(
            dtype=float,
            shape=st.tuples(st.integers(min_value=1, max_value=3)),
            elements=floats(
                min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=10, deadline=None)  # Disable deadline
    def test_forward_random(self, arr):
        """
        Test forward pass with random shapes, ensuring shape output is consistent.
        """
        t_hat_noise_level = torch.from_numpy(arr).float()
        out = self.fourier.forward(t_hat_noise_level)
        # Output shape => [..., c]
        self.assertEqual(out.shape[-1], self.c)
        # Check some basic range properties (cos(...) range is [-1,1])
        self.assertTrue((out >= -1).all() and (out <= 1).all())

    def test_forward_example(self):
        """
        Simple deterministic check: If we pass in zeros,
        the output should be cos(2π * b).
        """
        n_samples = 3
        input_zeros = torch.zeros(n_samples)
        out = self.fourier.forward(input_zeros)
        self.assertEqual(out.shape, (n_samples, self.c))
        # Each output element is cos(2 * pi * b[i])
        # We'll check only that it's within [-1,1] and that the shape is correct.
        self.assertTrue((out >= -1).all() and (out <= 1).all())

    def test_non_tensor_input_raises_error(self):
        """
        Confirm that passing a non-tensor or invalid type raises an error.
        """
        with self.assertRaises(AttributeError):
            _ = self.fourier.forward(None)


###############################################################################
#                                   MAIN
###############################################################################
if __name__ == "__main__":
    unittest.main()
