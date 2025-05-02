import torch

# Import the config class as well
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention_encoder import (
    AtomAttentionConfig,
    AtomAttentionEncoder,
)


def test_atom_encoder_smoke():
    # Create the configuration object
    config = AtomAttentionConfig(has_coords=False, c_token=384)
    # Instantiate the model with the config object
    model = AtomAttentionEncoder(config=config)

    # Calculate the total input feature dimension based on model.input_feature
    # This ensures our test inputs match what the model expects
    total_feature_dim = sum(model.input_feature.values())

    # Create a minimal dict f with correct dimensions
    # Important: All tensors need to have the same batch dimension structure
    batch_size = 1
    n_atoms = 5
    n_tokens = 2

    # Create atom-to-token mapping that maps each atom to a token index
    # Format: [batch_size, n_atoms] - each value is a token index
    atom_to_token_idx = torch.tensor([[0, 0, 1, 1, 0]])  # [1, 5]

    f = {
        # Add batch dimension to all atom-level features
        "ref_pos": torch.randn(batch_size, n_atoms, 3),  # [1, 5, 3]
        "ref_charge": torch.zeros(batch_size, n_atoms, 1),  # [1, 5, 1]
        "ref_mask": torch.ones(batch_size, n_atoms, 1, dtype=torch.bool),  # [1, 5, 1]
        "ref_element": torch.zeros(batch_size, n_atoms, 128),  # [1, 5, 128]
        "ref_atom_name_chars": torch.zeros(batch_size, n_atoms, 256),  # [1, 5, 256]

        # Atom to token mapping - CRITICAL: must be [batch_size, n_atoms] not [batch_size, n_atoms, 1]
        "atom_to_token_idx": atom_to_token_idx,  # [1, 5]

        # Token-level features
        "restype": torch.zeros(batch_size, n_tokens, 32),  # [1, 2, 32]

        # Space UID with batch dimension
        "ref_space_uid": torch.zeros(batch_size, n_atoms, 1),  # [1, 5, 1]
    }

    # Verify that our input features match the expected dimensions
    expected_feature_dim = 3 + 1 + 1 + 128 + 256  # Sum of all feature dimensions
    assert expected_feature_dim == total_feature_dim, f"Input feature dimensions mismatch: got {expected_feature_dim}, expected {total_feature_dim}"

    # Run the model
    out = model(f)  # Pass the dictionary directly

    # Verify outputs
    assert len(out) == 4, "Expected 4 output tensors"  # returns (a_token, q_atom, c_atom0, p_lm=None)
    assert out[0].shape[1] == n_tokens, f"Expected {n_tokens} tokens, got {out[0].shape[1]}"  # With batch dim, shape is [1, 2, 384]
    # Patch: Pair embedding is now a zero tensor, not None
    assert torch.all(out[3] == 0), "Pair embedding should be a zero tensor when has_coords=False"
