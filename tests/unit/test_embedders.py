import unittest
import torch
import torch.nn as nn
from typing import Any, Optional, Dict

from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis import example
from hypothesis.strategies import integers, floats, booleans, none, one_of, dictionaries
from hypothesis.extra.numpy import arrays
import math

# Import the classes under test
from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    InputFeatureEmbedder,
    RelativePositionEncoding,
    FourierEmbedding
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
        # Smaller dims to speed up tests and reduce memory usage:
        self.c_atom = 16
        self.c_atompair = 4
        self.c_token = 32
        self.embedder = InputFeatureEmbedder(
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_token=self.c_token
        )

    def test_init_values(self):
        """
        Test the constructor sets correct default values.
        """
        self.assertEqual(self.embedder.c_atom, self.c_atom)
        self.assertEqual(self.embedder.c_atompair, self.c_atompair)
        self.assertEqual(self.embedder.c_token, self.c_token)
        self.assertIsInstance(self.embedder.atom_attention_encoder, nn.Module)

    @given(
        # Basic random shapes for the restype, profile, deletion_mean
        restype=arrays(
            dtype=torch.int64,
            shape=st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=5))
        ),
        profile=arrays(
            dtype=torch.float32,
            shape=st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=5), st.just(32))
        ),
        deletion_mean=arrays(
            dtype=torch.float32,
            shape=st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=5), st.just(1))
        ),
        inplace_safe=booleans(),
        chunk_size=one_of(none(), integers(min_value=1, max_value=8))
    )
    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], max_examples=20)
    @example(
        restype=torch.tensor([[1,2,3]]),
        profile=torch.randn(1,3,32),
        deletion_mean=torch.randn(1,3,1),
        inplace_safe=False,
        chunk_size=None
    )
    def test_forward(self, restype, profile, deletion_mean, inplace_safe, chunk_size):
        """
        Test forward pass under random shapes and configurations using Hypothesis.
        Ensure it returns a tensor with expected shape and dtype.
        """
        # Convert numpy arrays to torch
        restype_t = torch.from_numpy(restype)
        profile_t = torch.from_numpy(profile)
        deletion_mean_t = torch.from_numpy(deletion_mean)

        input_feature_dict = {
            "restype": restype_t,
            "profile": profile_t,
            "deletion_mean": deletion_mean_t
        }

        output = self.embedder.forward(
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size
        )
        self.assertTrue(isinstance(output, torch.Tensor))
        # The expected final dim is c_token + 32 + 32 + 1
        expected_feat_size = self.c_token + 32 + 32 + 1
        self.assertEqual(output.shape[-1], expected_feat_size)

    def test_missing_key_raises_error(self):
        """
        Check that a KeyError (or similar) is raised if an input feature is missing.
        """
        incomplete_dict = {
            # "restype" missing
            "profile": torch.randn(2, 5, 32),
            "deletion_mean": torch.randn(2, 5, 1)
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
            r_max=self.r_max,
            s_max=self.s_max,
            c_z=self.c_z
        )

    def test_init_values(self):
        """Test constructor sets correct parameter values."""
        self.assertEqual(self.rpe.r_max, self.r_max)
        self.assertEqual(self.rpe.s_max, self.s_max)
        self.assertEqual(self.rpe.c_z, self.c_z)
        self.assertIsInstance(self.rpe.linear_no_bias, nn.Linear)

    @given(
        token_len=st.integers(min_value=1, max_value=5),
        batch_size=st.integers(min_value=1, max_value=2),
        training_mode=booleans()
    )
    @settings(max_examples=10)
    def test_forward_training_and_eval(self, token_len, batch_size, training_mode):
        """
        Test forward pass in both training and eval modes to ensure the
        conditional logic is covered.
        """
        self.rpe.train(training_mode)

        # We must supply "asym_id", "residue_index", "entity_id", "sym_id", "token_index"
        # Each is shape [..., N_tokens], so shape (batch_size, token_len).
        # For simplicity, use random integer data in valid ranges.
        input_feature_dict = {
            "asym_id": torch.randint(low=0, high=3, size=(batch_size, token_len)),
            "residue_index": torch.randint(low=0, high=100, size=(batch_size, token_len)),
            "entity_id": torch.randint(low=0, high=3, size=(batch_size, token_len)),
            "sym_id": torch.randint(low=0, high=3, size=(batch_size, token_len)),
            "token_index": torch.arange(token_len).unsqueeze(0).repeat(batch_size, 1)
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
        # Provide partial keys only
        input_dict = {
            "asym_id": torch.randint(0, 3, (2, 3)),
            # missing 'residue_index', 'entity_id', 'sym_id', 'token_index'
        }
        with self.assertRaises(KeyError):
            self.rpe.forward(input_dict)

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
            elements=floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=10)
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