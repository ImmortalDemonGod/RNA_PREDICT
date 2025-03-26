import pytest
import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer import AtomAttentionEncoder

def test_atom_encoder_no_coords():
    """
    Verify that if has_coords=False, the trunk logic is skipped and no AssertionError arises.
    """
    encoder = AtomAttentionEncoder(
        has_coords=False,
        c_atom=128,
        c_atompair=16,
        c_token=384,
    )
    input_feature_dict = {
        "atom_to_token_idx": torch.arange(48).unsqueeze(-1),
        "ref_pos": torch.randn(48, 3),
        "ref_charge": torch.zeros(48, 1),
        "ref_mask": torch.ones(48, 1, dtype=torch.bool),
        "ref_element": torch.zeros(48, 128),
        "ref_atom_name_chars": torch.zeros(48, 256),
        "ref_space_uid": torch.zeros(48, 1),
        "restype": torch.zeros(1, 12, 32),    # [batch=1, N_token=12, 32-dim feature]
        "profile": torch.zeros(1, 12, 32),
        "deletion_mean": torch.zeros(1, 12, 1),
    }

    a, q_l, c_l, p_lm = encoder(input_feature_dict)
    # trunk logic was skipped => p_lm should be None
    assert p_lm is None, "Expected p_lm=None if has_coords=False"
    assert a.shape == (1, 12, 384), f"Token shape mismatch, got {a.shape}"
    assert q_l.shape[-1] == 128, "Atom embedding last dim mismatch"
    assert c_l.shape == q_l.shape, "c_l should match q_l when skipping trunk"

def test_atom_encoder_has_coords():
    """
    Verify that if has_coords=True, the trunk logic runs, dimension mismatch is avoided by unifying -2.
    """
    encoder = AtomAttentionEncoder(
        has_coords=True,
        c_atom=128,
        c_atompair=16,
        c_token=384,
    )
    input_feature_dict = {
        "atom_to_token_idx": torch.arange(48).unsqueeze(-1),
        "ref_pos": torch.randn(48, 3),
        "ref_charge": torch.zeros(48, 1),
        "ref_mask": torch.ones(48, 1, dtype=torch.bool),
        "ref_element": torch.zeros(48, 128),
        "ref_atom_name_chars": torch.zeros(48, 256),
        "ref_space_uid": torch.zeros(48, 1),
        "restype": torch.zeros(1, 12, 32),
        "profile": torch.zeros(1, 12, 32),
        "deletion_mean": torch.zeros(1, 12, 1),
    }

    a, q_l, c_l, p_lm = encoder(input_feature_dict)
    # trunk logic used => p_lm should be non-None
    assert p_lm is not None, "Expected trunk-based pair embedding if has_coords=True"
    assert a.shape == (1, 12, 384), f"Expected token shape (1,12,384), got {a.shape}"
    assert q_l.shape[-1] == 128 and c_l.shape[-1] == 128

    # p_lm is chunk-based => shape might be [*, n_trunks, n_queries, n_keys, c_atompair]
    # We won't fully check, just ensure it's 5D or more
    assert p_lm.ndim >= 5, f"Expected p_lm to be at least 5D, got shape {p_lm.shape}"

if __name__ == "__main__":
    pytest.main([__file__])