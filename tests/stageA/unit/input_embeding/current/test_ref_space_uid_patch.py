import pytest
import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
    AtomAttentionEncoder,
)


def test_ref_space_uid_patch():
    """
    Checks that ref_space_uid with shape [B, N_atom] is automatically unsqueezed
    to [B, N_atom, 1], preventing dimension mismatch in rearrange_qk_to_dense_trunk.
    """
    encoder = AtomAttentionEncoder(
        c_atom=128, c_atompair=16, c_token=384, has_coords=True
    )

    B, N_atom = 1, 5
    # Make a minimal input feature dict
    ref_pos = torch.randn((B, N_atom, 3), dtype=torch.float32)
    ref_space_uid = torch.arange(N_atom).unsqueeze(
        0
    )  # shape [1, 5], triggers the patch
    input_feature_dict = {
        "ref_pos": ref_pos,
        "ref_space_uid": ref_space_uid,
        "ref_charge": torch.zeros((B, N_atom, 1), dtype=torch.float32),
        "ref_mask": torch.ones((B, N_atom, 1), dtype=torch.bool),
        "ref_element": torch.zeros((B, N_atom, 128), dtype=torch.float32),
        "ref_atom_name_chars": torch.zeros((B, N_atom, 256), dtype=torch.float32),
        "atom_to_token_idx": torch.arange(N_atom).unsqueeze(0),  # shape [1, 5]
        # Minimal token-level features:
        "restype": torch.zeros((B, 3, 32), dtype=torch.float32),
        "profile": torch.zeros((B, 3, 32), dtype=torch.float32),
        "deletion_mean": torch.zeros((B, 3, 1), dtype=torch.float32),
    }

    # Attempt forward pass
    # Without the patch, we used to get AssertionError. Now it should pass.
    try:
        a, q_l, c_l, p_lm = encoder(input_feature_dict)
    except AssertionError:
        pytest.fail(
            "AssertionError still triggered after shape patch for ref_space_uid!"
        )

    # If we get here, no dimension mismatch.
    assert a is not None, "No token-level output produced."
    assert a.dim() == 3, f"Expected 3D output, got shape {a.shape}."
    print("test_ref_space_uid_patch passed: dimension mismatch is resolved.")
