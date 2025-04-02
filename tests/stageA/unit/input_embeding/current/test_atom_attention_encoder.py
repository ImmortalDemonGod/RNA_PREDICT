import pytest
import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
    AtomAttentionEncoder,
    AtomAttentionConfig,
)


@pytest.mark.xfail(
    reason="Indices exceed the token range => scatter out-of-bounds error."
)
def test_atom_encoder_no_coords_original_fail():
    """
    Original test that incorrectly sets atom_to_token_idx in [0..47]
    for only 12 tokens, guaranteeing an out-of-bounds error during scatter.
    """
    encoder = AtomAttentionEncoder.from_args(
        has_coords=False,
        c_atom=128,
        c_atompair=16,
        c_token=384,
    )
    input_feature_dict = {
        # WRONG: max index is 47 for only 12 tokens => out-of-bounds
        "atom_to_token_idx": torch.arange(48).unsqueeze(-1),
        "ref_pos": torch.randn(48, 3),
        "ref_charge": torch.zeros(48, 1),
        "ref_mask": torch.ones(48, 1, dtype=torch.bool),
        "ref_element": torch.zeros(48, 128),
        "ref_atom_name_chars": torch.zeros(48, 256),
        "ref_space_uid": torch.zeros(48, 1),
        # nominal tokens=12
        "restype": torch.zeros(1, 12, 32),
        "profile": torch.zeros(1, 12, 32),
        "deletion_mean": torch.zeros(1, 12, 1),
    }

    # This call triggers the out-of-bounds error
    a, q_l, c_l, p_lm = encoder(input_feature_dict)
    # We won't reach here if the aggregator does scatter_add_ with index >= 12


def test_atom_encoder_no_coords_fixed():
    """
    Corrected version: ensures each of the 48 atoms maps to an index in [0..11].
    We have 48 atoms, 12 tokens => 4 atoms per token. No out-of-bounds in scatter.
    """
    encoder = AtomAttentionEncoder.from_args(
        has_coords=False,
        c_atom=128,
        c_atompair=16,
        c_token=384,
    )

    # Group 48 atoms into 12 tokens => index range [0..11]
    token_map = torch.repeat_interleave(torch.arange(12), 4)  # shape [48]
    assert token_map.shape[0] == 48

    input_feature_dict = {
        "atom_to_token_idx": token_map.unsqueeze(-1),  # shape [48, 1], all in [0..11]
        "ref_pos": torch.randn(48, 3),
        "ref_charge": torch.zeros(48, 1),
        "ref_mask": torch.ones(48, 1, dtype=torch.bool),
        "ref_element": torch.zeros(48, 128),
        "ref_atom_name_chars": torch.zeros(48, 256),
        "ref_space_uid": torch.zeros(48, 1),
        "restype": torch.zeros(1, 12, 32),  # batch=1, 12 tokens
        "profile": torch.zeros(1, 12, 32),
        "deletion_mean": torch.zeros(1, 12, 1),
    }

    a, q_l, c_l, p_lm = encoder.forward_legacy(input_feature_dict)

    # With has_coords=False, trunk logic is skipped => p_lm should be None
    assert p_lm is None, "Expected no trunk-based aggregator when has_coords=False"

    # a => shape [1, 12, 384] or [12, 384], check last dimension
    assert a.shape[-1] == 384, f"Token embedding last dim must be 384, got {a.shape}"

    # q_l, c_l => per-atom embeddings => shape [48,128] or [1,48,128]
    assert q_l.shape[-1] == 128
    assert c_l.shape[-1] == 128

    print("test_atom_encoder_no_coords_fixed: PASS - no out-of-bounds error.")
