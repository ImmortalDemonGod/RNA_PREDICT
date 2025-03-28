import pytest
import torch

"""
This file now tests only the *current* AtomTransformer from
rna_predict.pipeline.stageA.input_embedding.current.transformer
All legacy and block-specific tests have been removed.
"""

def test_atom_transformer_full_stack():
    """
    Test the full AtomTransformer with multiple blocks (using the current code).
    Also checks 3D fallback vs. 5D trunk usage.
    """
    from rna_predict.pipeline.stageA.input_embedding.current.transformer import AtomTransformer

    torch.manual_seed(43)
    c_atom = 32
    num_heads = 4
    # We'll keep n_blocks=3 for the 3D fallback usage:
    n_blocks_3d = 3
    n_atom = 10
    c_pair = 8

    # 3D fallback path
    x = torch.randn(n_atom, c_atom)
    c = torch.randn(n_atom, c_atom)
    p_3d = torch.randn(n_atom, n_atom, c_pair)

    model_3d = AtomTransformer(
        c_atom=c_atom,
        c_atompair=c_pair,
        n_blocks=n_blocks_3d,
        n_heads=num_heads
    )
    out_3d = model_3d(x, c, p_3d)
    assert out_3d.shape == (n_atom, c_atom), f"3D fallback => got {out_3d.shape}"

    # 5D trunk path:
    # We'll set n_blocks=2, n_queries=4, n_keys=4
    n_blocks_5d = 2
    n_queries_5d = 4
    n_keys_5d = 4
    x5 = x.unsqueeze(0)
    c5 = c.unsqueeze(0)
    # shape [batch=1, n_blocks=2, n_queries=4, n_keys=4, c_pair=8]
    p_5d = torch.randn(1, n_blocks_5d, n_queries_5d, n_keys_5d, c_pair)

    model_5d = AtomTransformer(
        c_atom=c_atom,
        c_atompair=c_pair,
        n_blocks=n_blocks_5d,
        n_heads=num_heads,
        n_queries=n_queries_5d,
        n_keys=n_keys_5d
    )
    out_5d = model_5d(x5, c5, p_5d)
    # Should produce shape [1, n_atom, c_atom]
    assert out_5d.shape == (1, n_atom, c_atom), f"5D trunk => got {out_5d.shape}"


class TestAtomTransformer:
    """
    Tests for the multi-layer AtomTransformer from current code (transformer.py)
    """

    def setup_method(self):
        self.c_atom = 64
        self.num_heads = 4
        self.n_blocks = 3
        from rna_predict.pipeline.stageA.input_embedding.current.transformer import AtomTransformer
        self.AtomTransformerClass = AtomTransformer
        # Common test data
        self.n_atoms = 5
        self.c_pair = 32
        self.x = torch.randn(self.n_atoms, self.c_atom)
        self.c_embed = torch.randn(self.n_atoms, self.c_atom)
        self.pair_emb = torch.randn(self.n_atoms, self.n_atoms, self.c_pair)

    def test_forward_consistency(self):
        model = self.AtomTransformerClass(
            c_atom=self.c_atom,
            c_atompair=self.c_pair,
            n_blocks=self.n_blocks,
            n_heads=self.num_heads
        )
        out = model(self.x, self.c_embed, self.pair_emb)
        assert out.shape == (self.n_atoms, self.c_atom)
        assert not torch.isnan(out).any(), "No NaNs expected in output."

    def test_transformer_random_shapes(self):
        """
        Check random shapes for the full multi-block AtomTransformer.
        """
        for n_atoms in [1, 2, 6, 10]:
            x = torch.randn(n_atoms, self.c_atom)
            pair_emb = torch.randn(n_atoms, n_atoms, self.c_pair)
            model = self.AtomTransformerClass(
                c_atom=self.c_atom,
                c_atompair=self.c_pair,
                n_blocks=self.n_blocks,
                n_heads=self.num_heads
            )
            out = model(x, x, pair_emb)
            assert out.shape == (n_atoms, self.c_atom), f"Expected shape ({n_atoms}, {self.c_atom})."

    def test_transformer_round_trip(self):
        """
        Repeated forward pass to ensure shape is preserved and numeric stability is maintained.
        """
        model = self.AtomTransformerClass(
            c_atom=self.c_atom,
            c_atompair=self.c_pair,
            n_blocks=self.n_blocks,
            n_heads=self.num_heads
        )
        out1 = model(self.x, self.c_embed, self.pair_emb)
        out2 = model(out1, out1, self.pair_emb)
        assert out2.shape == (self.n_atoms, self.c_atom)
        assert torch.isfinite(out2).all()