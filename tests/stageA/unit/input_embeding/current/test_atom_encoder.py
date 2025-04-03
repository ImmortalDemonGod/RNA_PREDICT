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
    # Create a minimal dict f
    f = {
        "ref_pos": torch.randn(5, 3),
        "ref_charge": torch.zeros(5, 1),
        "ref_mask": torch.ones(5, 1, dtype=torch.bool),
        "ref_element": torch.zeros(5, 128),
        "ref_atom_name_chars": torch.zeros(5, 256),
        "atom_to_token_idx": torch.tensor([0, 0, 1, 1, 0]),
        "restype": torch.zeros(
            1, 2, 32
        ),  # batch=1, 2 tokens (corrected based on atom_to_token_idx)
        # Add missing required features based on AtomAttentionEncoder._setup_feature_dimensions
        "ref_space_uid": torch.zeros(5, 3),  # Added missing feature
    }
    # Add block_index only if needed by the specific path taken in forward (depends on has_coords)
    # Since has_coords=False, it takes the _process_simple_embedding path which doesn't use block_index
    # "block_index": torch.randint(0, 5, (5, 2)), # Removed as not needed for has_coords=False path

    out = model(f)  # Pass the dictionary directly
    assert len(out) == 4  # returns (a_token, q_atom, c_atom0, p_lm=None)
    assert (
        out[0].shape[0] == 2
    )  # 2 tokens expected based on atom_to_token_idx and restype
    assert out[3] is None  # Pair embedding should be None when has_coords=False
