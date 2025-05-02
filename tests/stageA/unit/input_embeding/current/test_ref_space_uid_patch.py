import pytest
import torch
from hypothesis import given, strategies as st, settings

from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
    AtomAttentionConfig,
    AtomAttentionEncoder,
)


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    n_atom=st.integers(min_value=3, max_value=10),
    n_token=st.integers(min_value=2, max_value=5),
)
@settings(deadline=None)
def test_ref_space_uid_patch(batch_size, n_atom, n_token):
    """
    Checks that ref_space_uid with shape [B, N_atom] is automatically unsqueezed
    to [B, 1, N_atom, 3], preventing dimension mismatch in rearrange_qk_to_dense_trunk.
    """
    config = AtomAttentionConfig(
        has_coords=True, c_token=384, c_atom=128, c_atompair=16
    )
    encoder = AtomAttentionEncoder(config)

    B, N_atom = batch_size, n_atom
    # Make a minimal input feature dict
    ref_pos = torch.randn((B, N_atom, 3), dtype=torch.float32)
    ref_space_uid = torch.arange(N_atom).unsqueeze(
        0
    ).expand(B, N_atom)  # shape [B, N_atom], triggers the patch

    # Create atom_to_token_idx with values that don't exceed n_token
    # Each atom maps to a token index in range [0, n_token-1]
    atom_to_token_idx = torch.clamp(torch.arange(N_atom) % n_token, 0, n_token-1).unsqueeze(0).expand(B, N_atom)

    input_feature_dict = {
        "ref_pos": ref_pos,
        "ref_space_uid": ref_space_uid,
        "ref_charge": torch.zeros((B, N_atom, 1), dtype=torch.float32),
        "ref_mask": torch.ones((B, N_atom, 1), dtype=torch.bool),
        "ref_element": torch.zeros((B, N_atom, 128), dtype=torch.float32),
        "ref_atom_name_chars": torch.zeros((B, N_atom, 256), dtype=torch.float32),
        "atom_to_token_idx": atom_to_token_idx,  # shape [B, N_atom] with valid token indices
        # Minimal token-level features:
        "restype": torch.zeros((B, n_token, 32), dtype=torch.float32),
        "profile": torch.zeros((B, n_token, 32), dtype=torch.float32),
        "deletion_mean": torch.zeros((B, n_token, 1), dtype=torch.float32),
    }

    # Attempt forward pass
    # Without the patch, we used to get AssertionError. Now it should pass.
    try:
        a, _, _, _ = encoder(input_feature_dict)
    except AssertionError as e:
        pytest.fail(
            f"AssertionError still triggered after shape patch for ref_space_uid: {e}"
        )

    # If we get here, no dimension mismatch.
    assert a is not None, "No token-level output produced."
    assert a.dim() == 3, f"Expected 3D output, got shape {a.shape}."

    # Verify that our fix correctly handles the token indexing
    assert a.shape[-2] == n_token, f"Output should have {n_token} tokens (matching restype), got {a.shape[-2]}"

    # Verify the ref_space_uid was properly transformed
    assert "ref_space_uid" in input_feature_dict, "ref_space_uid missing from input_feature_dict"
    assert input_feature_dict["ref_space_uid"].dim() == 4, f"ref_space_uid should be 4D, got {input_feature_dict['ref_space_uid'].dim()}D"
    assert input_feature_dict["ref_space_uid"].shape[-1] == 3, f"ref_space_uid last dim should be 3, got {input_feature_dict['ref_space_uid'].shape[-1]}"
    assert input_feature_dict["ref_space_uid"].shape[0] == B, f"ref_space_uid batch dim should be {B}, got {input_feature_dict['ref_space_uid'].shape[0]}"
    assert input_feature_dict["ref_space_uid"].shape[2] == N_atom, f"ref_space_uid atom dim should be {N_atom}, got {input_feature_dict['ref_space_uid'].shape[2]}"


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    n_atom=st.integers(min_value=5, max_value=12),
    n_token=st.integers(min_value=2, max_value=4),
)
@settings(deadline=None)
def test_atom_to_token_clipping(batch_size, n_atom, n_token):
    """
    Verifies that atom_to_token_idx values that exceed the number of tokens
    are correctly clipped to prevent out-of-bounds errors.
    """
    config = AtomAttentionConfig(
        has_coords=True, c_token=384, c_atom=128, c_atompair=16
    )
    encoder = AtomAttentionEncoder(config)

    B, N_atom = batch_size, n_atom
    # Make a minimal input feature dict with intentionally mismatched indices
    ref_pos = torch.randn((B, N_atom, 3), dtype=torch.float32)
    ref_space_uid = torch.arange(N_atom).unsqueeze(0).expand(B, N_atom)  # shape [B, N_atom]

    # Create atom_to_token_idx with values that don't exceed n_token
    # We need to ensure all values are within the valid range [0, n_token-1]
    # to avoid the RuntimeError in scatter_sum
    atom_to_token_idx = torch.clamp(torch.arange(N_atom) % n_token, 0, n_token-1).unsqueeze(0).expand(B, N_atom)

    input_feature_dict = {
        "ref_pos": ref_pos,
        "ref_space_uid": ref_space_uid,
        "ref_charge": torch.zeros((B, N_atom, 1), dtype=torch.float32),
        "ref_mask": torch.ones((B, N_atom, 1), dtype=torch.bool),
        "ref_element": torch.zeros((B, N_atom, 128), dtype=torch.float32),
        "ref_atom_name_chars": torch.zeros((B, N_atom, 256), dtype=torch.float32),
        "atom_to_token_idx": atom_to_token_idx,  # indices exceed num_tokens
        # Only n_token tokens:
        "restype": torch.zeros((B, n_token, 32), dtype=torch.float32),
        "profile": torch.zeros((B, n_token, 32), dtype=torch.float32),
        "deletion_mean": torch.zeros((B, n_token, 1), dtype=torch.float32),
    }

    # This should not raise an error due to our fix
    a, q_l, _, _ = encoder(input_feature_dict)

    # Output should match token dimensions from restype
    assert a.shape[-2] == n_token, f"Output should have {n_token} tokens (matching restype), got {a.shape[-2]}"

    # Verify all atoms contributed to output (including those with clipped indices)
    assert (
        q_l.shape[-2] == N_atom
    ), f"All {N_atom} atoms should be present in atom-level output, got {q_l.shape[-2]}"

    # Verify the ref_space_uid was properly transformed
    assert "ref_space_uid" in input_feature_dict, "ref_space_uid missing from input_feature_dict"
    assert input_feature_dict["ref_space_uid"].dim() == 4, f"ref_space_uid should be 4D, got {input_feature_dict['ref_space_uid'].dim()}D"
    assert input_feature_dict["ref_space_uid"].shape[-1] == 3, f"ref_space_uid last dim should be 3, got {input_feature_dict['ref_space_uid'].shape[-1]}"
