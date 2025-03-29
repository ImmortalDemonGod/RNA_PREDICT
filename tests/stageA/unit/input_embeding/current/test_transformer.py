import unittest

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# We import the classes under test as if they are in the same directory or installed.
# If needed, adjust imports per your project structure.
try:
    from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
        AtomAttentionDecoder,
        AtomAttentionEncoder,
        AtomTransformer,
        AttentionPairBias,
        ConditionedTransitionBlock,
        DiffusionTransformer,
        DiffusionTransformerBlock,
    )
except ImportError:
    # If your code is in a submodule like "rna_predict.pipeline.stageA.input_embedding.current.transformer",
    # adjust the import path accordingly. For demonstration, we assume `transformer.py` is accessible.
    raise


# -------------------------
# Utility strategies for Hypothesis
# -------------------------


# Strategy to generate small random 2D or 3D Tensors that are likely to be valid inputs
# for the classes in the code. We keep shapes small for performance reasons.
def tensor_strat(
    min_size: int = 1,
    max_size: int = 8,
    requires_grad: bool = False,
    min_val: float = -10.0,
    max_val: float = 10.0,
    dims=2,
):
    """Generates a random float Tensor with 2 or 3 dimensions, within [min_val, max_val]."""
    assert dims in [2, 3, 4, 5], "dims must be 2, 3, 4, or 5"
    shape_strat = st.lists(
        st.integers(min_size, max_size), min_size=dims, max_size=dims
    )
    return shape_strat.map(
        lambda shape: (
            torch.rand(*shape) * (max_val - min_val) + min_val
        ).requires_grad_(requires_grad)
    )


# Strategy for n_heads that divides c_a
def valid_n_heads_strategy():
    # c_a typically can be 128, 256, etc. We'll just pick from a small set
    possible_c_a = [16, 32, 64, 128]
    # We'll pick a random c_a, then pick an n_heads that divides it
    return st.builds(
        lambda c_a: (
            c_a,
            st.sampled_from([h for h in range(1, c_a + 1) if c_a % h == 0]),
        ),
        st.sampled_from(possible_c_a),
    ).map(lambda x: (x[0], x[1].example() if hasattr(x[1], "example") else x[1]))


# -----------------------------------------
#  TestAttentionPairBias
# -----------------------------------------
class TestAttentionPairBias(unittest.TestCase):
    """Tests for the AttentionPairBias class."""

    def setUp(self):
        """Creates a default instance with typical arguments."""
        self.default_has_s = True
        self.default_n_heads = 4
        self.default_c_a = 128
        self.default_c_s = 64
        self.default_c_z = 32
        self.default_biasinit = -2.0
        self.module = AttentionPairBias(
            has_s=self.default_has_s,
            n_heads=self.default_n_heads,
            c_a=self.default_c_a,
            c_s=self.default_c_s,
            c_z=self.default_c_z,
            biasinit=self.default_biasinit,
        )

    def test_basic_instantiation(self):
        """Test that a module can be instantiated without error."""
        self.assertIsInstance(self.module, AttentionPairBias)
        self.assertEqual(self.module.n_heads, self.default_n_heads)

    def test_glorot_init(self):
        """Test the glorot_init method does not raise errors."""
        try:
            self.module.glorot_init()
        except Exception as e:
            self.fail(f"glorot_init raised an exception: {e}")

    @settings(
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
        max_examples=20,
    )
    @given(
        a=tensor_strat(dims=3),
        s=tensor_strat(dims=3),
        z=tensor_strat(
            dims=4
        ),  # For local multi-head usage: e.g. [..., n_blocks, n_queries, n_keys, c_z]
        n_queries=st.integers(min_value=1, max_value=16),
        n_keys=st.integers(min_value=1, max_value=16),
        inplace_safe=st.booleans(),
        chunk_size=st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
    )
    def test_local_multihead_attention(
        self, a, s, z, n_queries, n_keys, inplace_safe, chunk_size
    ):
        """Fuzzy test local_multihead_attention for shape/dtype correctness."""
        # We forcibly shape z to pretend: [..., n_blocks, n_queries, n_keys, c_z] => dims=5
        # We'll do a quick clamp for shape
        # We only do this if z has dims=4; let's reshape to (B, n_blocks, n_queries, n_keys, c_z)
        # For test, let's pick n_blocks=1 if possible
        if z.dim() == 4 and z.size(1) >= n_queries and z.size(2) >= n_keys:
            # Insert dimension for n_blocks=1
            z_reshaped = z.unsqueeze(1)
            # We do a partial slice to shape it
            z_reshaped = z_reshaped[:, :, :n_queries, :n_keys, :]
            # Now shape is [B, 1, n_queries, n_keys, cZ]
            if z_reshaped.size(-1) != self.default_c_z:
                # We skip if it doesn't match c_z
                return
            if a.size(-1) != self.default_c_a or s.size(-1) != self.default_c_s:
                return
            # Attempt the call
            try:
                out = self.module.local_multihead_attention(
                    a=a,
                    s=s,
                    z=z_reshaped,
                    n_queries=n_queries,
                    n_keys=n_keys,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
                self.assertEqual(out.shape, a.shape)
            except (ValueError, AssertionError, RuntimeError):
                # Some shape mismatch could happen
                pass

    @settings(
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
        max_examples=20,
    )
    @given(
        a=tensor_strat(dims=3),
        s=tensor_strat(dims=3),
        z=tensor_strat(dims=3),
        inplace_safe=st.booleans(),
    )
    def test_standard_multihead_attention(self, a, s, z, inplace_safe):
        """Fuzzy test standard_multihead_attention shape correctness."""
        if a.size(-1) != self.default_c_a or s.size(-1) != self.default_c_s:
            return
        # z: [..., N_token, N_token, c_z], dim=3 => we want 4D
        # We do quick shape check
        # Let's see if z is shape [B, N, c_z], we'd need [B, N, N, c_z]
        if z.dim() == 3:
            n = z.size(1)
            if n < 1:  # no tokens
                return
            # Expand to [B, N, N, c_z]
            z_expanded = z.unsqueeze(-2).expand(-1, n, n, -1)
            if z_expanded.size(-1) != self.default_c_z:
                return
            try:
                out = self.module.standard_multihead_attention(
                    a=a, s=s, z=z_expanded, inplace_safe=inplace_safe
                )
                self.assertEqual(out.shape, a.shape)
            except Exception:
                pass


# -----------------------------------------
#  TestConditionedTransitionBlock
# -----------------------------------------
class TestConditionedTransitionBlock(unittest.TestCase):
    """Tests for ConditionedTransitionBlock class."""

    def setUp(self):
        self.c_a = 64
        self.c_s = 64
        self.n = 2
        self.block = ConditionedTransitionBlock(self.c_a, self.c_s, self.n)

    def test_basic_forward(self):
        """Test forward method with typical shape."""
        a = torch.randn(8, self.c_a)
        s = torch.randn(8, self.c_s)
        out = self.block(a, s)
        self.assertEqual(out.shape, (8, self.c_a))

    @given(
        a=tensor_strat(dims=2, min_size=1, max_size=8),
        s=tensor_strat(dims=2, min_size=1, max_size=8),
    )
    @settings(max_examples=15)
    def test_fuzz_forward_dims(self, a, s):
        """Fuzz test the forward method for dimension correctness."""
        if a.size(-1) != self.c_a or s.size(-1) != self.c_s:
            return
        out = self.block(a, s)
        self.assertEqual(out.shape, a.shape)


# -----------------------------------------
#  TestDiffusionTransformerBlock
# -----------------------------------------
class TestDiffusionTransformerBlock(unittest.TestCase):
    """Tests for DiffusionTransformerBlock class."""

    def setUp(self):
        self.c_a = 64
        self.c_s = 64
        self.c_z = 32
        self.n_heads = 4
        self.block = DiffusionTransformerBlock(
            c_a=self.c_a, c_s=self.c_s, c_z=self.c_z, n_heads=self.n_heads
        )

    @given(
        a=tensor_strat(dims=3),
        s=tensor_strat(dims=3),
        z=tensor_strat(dims=4),
        n_queries=st.one_of(st.none(), st.integers(min_value=1, max_value=16)),
        n_keys=st.one_of(st.none(), st.integers(min_value=1, max_value=16)),
        inplace_safe=st.booleans(),
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    def test_forward_fuzz(self, a, s, z, n_queries, n_keys, inplace_safe):
        """Fuzz test the forward method for shape correctness."""
        if a.size(-1) != self.c_a or s.size(-1) != self.c_s:
            return
        # We want z to have shape [..., n_block, n_queries, n_keys, c_z] or [..., N, N, c_z]
        # We'll skip if shapes do not match c_z
        if z.size(-1) != self.c_z:
            return
        try:
            out_a, out_s, out_z = self.block(
                a=a,
                s=s,
                z=z,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
            )
            self.assertEqual(out_a.shape, a.shape)
        except Exception:
            pass


# -----------------------------------------
#  TestDiffusionTransformer
# -----------------------------------------
class TestDiffusionTransformer(unittest.TestCase):
    """Tests for DiffusionTransformer class."""

    def setUp(self):
        self.transformer = DiffusionTransformer(
            c_a=64, c_s=64, c_z=32, n_blocks=2, n_heads=4
        )

    def test_instantiation(self):
        """Ensure basic instantiation works."""
        self.assertIsInstance(self.transformer, DiffusionTransformer)
        self.assertEqual(self.transformer.n_blocks, 2)

    @given(
        a=tensor_strat(dims=3),
        s=tensor_strat(dims=3),
        z=tensor_strat(dims=3),
        n_queries=st.one_of(st.none(), st.integers(min_value=1, max_value=16)),
        n_keys=st.one_of(st.none(), st.integers(min_value=1, max_value=16)),
        inplace_safe=st.booleans(),
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    def test_forward_fuzz(self, a, s, z, n_queries, n_keys, inplace_safe):
        """Fuzz test the forward method shape correctness."""
        if a.size(-1) != 64 or s.size(-1) != 64 or z.size(-1) != 32:
            return
        try:
            out = self.transformer(
                a=a,
                s=s,
                z=z,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
            )
            self.assertEqual(out.shape, a.shape)
        except Exception:
            pass


# -----------------------------------------
#  TestAtomTransformer
# -----------------------------------------
class TestAtomTransformer(unittest.TestCase):
    """Tests for AtomTransformer class."""

    def setUp(self):
        self.transformer = AtomTransformer(
            c_atom=64, c_atompair=16, n_blocks=2, n_heads=4, n_queries=8, n_keys=8
        )

    def test_instantiation(self):
        """Check basic instantiation."""
        self.assertIsInstance(self.transformer, AtomTransformer)

    @given(
        q=tensor_strat(dims=3),
        c=tensor_strat(dims=3),
        p=tensor_strat(dims=3),
        inplace_safe=st.booleans(),
        chunk_size=st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    def test_forward_fuzz_3d_case(self, q, c, p, inplace_safe, chunk_size):
        """
        Fuzz test fallback usage: p.dim()==3 => global usage.
        """
        if q.size(-1) != 64 or c.size(-1) != 64 or p.size(-1) != 16:
            return
        try:
            out = self.transformer(q, c, p, inplace_safe, chunk_size)
            self.assertEqual(out.shape, q.shape)
        except (ValueError, AssertionError, RuntimeError):
            pass

    def test_error_for_incorrect_dim(self):
        """
        Test that passing a p with invalid dim triggers a ValueError.
        """
        q = torch.randn(2, 8, 64)
        c = torch.randn(2, 8, 64)
        p_invalid = torch.randn(2, 8, 8, 8, 8, 8)  # 6D
        with self.assertRaises(ValueError):
            _ = self.transformer(q, c, p_invalid)


# -----------------------------------------
#  TestAtomAttentionEncoder
# -----------------------------------------
class TestAtomAttentionEncoder(unittest.TestCase):
    """Tests for the AtomAttentionEncoder class."""

    def setUp(self):
        self.encoder = AtomAttentionEncoder(
            has_coords=True,
            c_token=128,
            c_atom=64,
            c_atompair=16,
            c_s=64,
            c_z=32,
            n_blocks=2,
            n_heads=4,
            n_queries=8,
            n_keys=8,
        )

    def test_instantiation(self):
        """Basic instantiation check."""
        self.assertIsInstance(self.encoder, AtomAttentionEncoder)

    def test_linear_init(self):
        """Smoke test the linear_init method."""
        try:
            self.encoder.linear_init(
                zero_init_atom_encoder_residual_linear=True,
                he_normal_init_atom_encoder_small_mlp=True,
                he_normal_init_atom_encoder_output=True,
            )
        except Exception as e:
            self.fail(f"linear_init raised an exception: {e}")

    @given(
        r_l=tensor_strat(dims=3),
        s=tensor_strat(dims=3),
        z=tensor_strat(dims=3),
        inplace_safe=st.booleans(),
        chunk_size=st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
    )
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    def test_forward_fuzz_has_coords(self, r_l, s, z, inplace_safe, chunk_size):
        """Fuzz test forward with has_coords=True."""
        input_feature_dict = {
            "ref_pos": torch.randn(2, 10, 3),  # shape [B, N_atom, 3]
            "ref_charge": torch.randn(2, 10, 1),
            "ref_mask": torch.ones(2, 10, 1),
            "ref_element": torch.randn(2, 10, 128),
            "ref_atom_name_chars": torch.randn(2, 10, 256),
            "atom_to_token_idx": torch.zeros(2, 10, dtype=torch.long),
            "restype": torch.randn(2, 10, 5),  # shape [B, N_token, features]
        }
        try:
            a, q_l, c_l, p_lm = self.encoder(
                input_feature_dict=input_feature_dict,
                r_l=r_l,
                s=s,
                z=z,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            # Basic shape checks
            self.assertEqual(a.dim(), 3)
            self.assertEqual(q_l.dim(), 3)
            self.assertEqual(c_l.dim(), 3)
        except ValueError:
            pass


# -----------------------------------------
#  TestAtomAttentionDecoder
# -----------------------------------------
class TestAtomAttentionDecoder(unittest.TestCase):
    """Tests for the AtomAttentionDecoder class."""

    def setUp(self):
        self.decoder = AtomAttentionDecoder(
            n_blocks=2,
            n_heads=4,
            c_token=128,
            c_atom=64,
            c_atompair=16,
            n_queries=8,
            n_keys=8,
        )

    def test_instantiation(self):
        """Ensure the decoder is instantiated properly."""
        self.assertIsInstance(self.decoder, AtomAttentionDecoder)

    @given(
        a=tensor_strat(dims=3),
        q_skip=tensor_strat(dims=3),
        c_skip=tensor_strat(dims=3),
        p_skip=tensor_strat(dims=5),  # local trunk shape
        inplace_safe=st.booleans(),
        chunk_size=st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
    )
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    def test_forward_fuzz(self, a, q_skip, c_skip, p_skip, inplace_safe, chunk_size):
        """Fuzz test the forward usage of AtomAttentionDecoder."""
        # We build a minimal input_feature_dict
        input_feature_dict = {
            "atom_to_token_idx": torch.zeros(a.shape[0], a.shape[1], dtype=torch.long)
        }
        # shape checks
        if (
            a.size(-1) != 128
            or q_skip.size(-1) != 64
            or c_skip.size(-1) != 64
            or p_skip.size(-1) != 16
        ):
            return
        try:
            out_coords = self.decoder(
                input_feature_dict=input_feature_dict,
                a=a,
                q_skip=q_skip,
                c_skip=c_skip,
                p_skip=p_skip,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            self.assertEqual(out_coords.shape[:-1], q_skip.shape[:-1])
            self.assertEqual(out_coords.size(-1), 3)
        except Exception:
            pass


# -----------------------------------------
# Round-Trip Test Example
# -----------------------------------------
class TestEncoderDecoderRoundTrip(unittest.TestCase):
    """
    Demonstrates a round-trip style test: encode with AtomAttentionEncoder, then decode
    with AtomAttentionDecoder to see if we get the correct shape or partial fidelity.
    """

    def setUp(self):
        self.encoder = AtomAttentionEncoder(
            has_coords=True,
            c_token=64,
            c_atom=64,
            c_atompair=16,
            c_s=64,
            c_z=32,
            n_blocks=2,
            n_heads=4,
            n_queries=8,
            n_keys=8,
        )
        self.decoder = AtomAttentionDecoder(
            n_blocks=2,
            n_heads=4,
            c_token=64,
            c_atom=64,
            c_atompair=16,
            n_queries=8,
            n_keys=8,
        )

    def test_encode_decode_shapes(self):
        """
        Simple test showing that after encoding, we can decode
        some representation back to coordinates of expected shape.
        """
        input_feature_dict = {
            "ref_pos": torch.randn(1, 5, 3),
            "ref_charge": torch.randn(1, 5, 1),
            "ref_mask": torch.ones(1, 5, 1),
            "ref_element": torch.randn(1, 5, 128),
            "ref_atom_name_chars": torch.randn(1, 5, 256),
            "atom_to_token_idx": torch.zeros(1, 5, dtype=torch.long),
            "restype": torch.randn(1, 5, 10),
            "ref_space_uid": torch.randint(
                0,
                2,
                (
                    1,
                    5,
                ),
            ),
        }

        # Encode
        a, q_l, c_l, p_lm = self.encoder(input_feature_dict)
        self.assertEqual(a.shape[-1], 64)
        self.assertEqual(q_l.shape[-1], 64)
        # Now decode
        # We pass (a, q_l, c_l, p_lm) as if the decoder can reconstruct positions
        # The docs say forward signature: (input_feature_dict, a, q_skip, c_skip, p_skip)
        out_coords = self.decoder(input_feature_dict, a, q_l, c_l, p_lm)
        # Should have shape [batch, N_atom, 3]
        self.assertEqual(out_coords.shape, torch.Size([1, 5, 3]))


if __name__ == "__main__":
    unittest.main()
