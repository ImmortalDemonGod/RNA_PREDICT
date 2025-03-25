# tests/test_token_feature_shape.py
import torch
import pytest
from rna_predict.pipeline.stageA.input_embedding.current.embedders import InputFeatureEmbedder

def test_input_feature_embedder_deletion_mean_shape():
    """
    Test that InputFeatureEmbedder properly handles 'deletion_mean'.
    Expected token features should have shape (batch, num_tokens, feature_dim)
    where num_tokens is derived from the 'restype' feature.
    
    We simulate a bug by providing a 'deletion_mean' tensor with shape (batch, 40)
    when num_tokens should be 10.
    
    When the bug is still present, the embedder will raise a RuntimeError due to an
    invalid reshape, and we mark the test as xfail. When the bug is fixed, the embedder
    returns a tensor with shape (batch, num_tokens, 449).
    """
    batch_size = 1
    num_tokens = 10
    num_atoms = 5  # arbitrary number for atom-level features

    # Token-level features
    restype = torch.zeros((batch_size, num_tokens, 32))
    profile = torch.zeros((batch_size, num_tokens, 32))
    # Simulate bug: incorrect shape for deletion_mean (should be (batch, num_tokens, 1))
    deletion_mean = torch.zeros((batch_size, 40))
    
    # Atom-level features required by AtomAttentionEncoder (has_coords is False)
    atom_to_token_idx = torch.zeros((batch_size, num_atoms), dtype=torch.long)  # dummy mapping
    ref_pos = torch.zeros((batch_size, num_atoms, 3))
    ref_charge = torch.zeros((batch_size, num_atoms, 1))
    ref_mask = torch.ones((batch_size, num_atoms, 1))
    ref_element = torch.zeros((batch_size, num_atoms, 128))
    ref_atom_name_chars = torch.zeros((batch_size, num_atoms, 256))
    # Provide required key 'ref_space_uid'
    ref_space_uid = torch.arange(num_atoms, device=ref_pos.device).unsqueeze(0)
    
    input_feature_dict = {
        "atom_to_token_idx": atom_to_token_idx,
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_mask": ref_mask,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "restype": restype,
        "profile": profile,
        "deletion_mean": deletion_mean,
    }
    
    embedder = InputFeatureEmbedder(c_atom=128, c_atompair=16, c_token=384)
    
    try:
        out = embedder(input_feature_dict)
    except RuntimeError as e:
        pytest.xfail("Bug not fixed: invalid deletion_mean shape raises RuntimeError")
    else:
        # When the bug is fixed, the output should have shape (batch, num_tokens, 449)
        expected_shape = (batch_size, num_tokens, 449)
        assert out.shape == expected_shape, f"Expected output shape {expected_shape}, but got {out.shape}"