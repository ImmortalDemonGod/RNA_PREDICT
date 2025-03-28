import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer import AtomAttentionEncoder
from rna_predict.pipeline.stageA.input_embedding.current.embedders import InputFeatureEmbedder as AtomEncoderConfig


def test_atom_encoder_smoke():
    model = AtomAttentionEncoder(has_coords=False, c_token=384)
    # Create a minimal dict f
    f = {
        "ref_pos": torch.randn(5, 3),
        "ref_charge": torch.zeros(5, 1),
        "ref_mask": torch.ones(5, 1, dtype=torch.bool),
        "ref_element": torch.zeros(5, 128),
        "ref_atom_name_chars": torch.zeros(5, 256),
        "atom_to_token_idx": torch.tensor([0, 0, 1, 1, 0]),
        "restype": torch.zeros(1, 12, 32),    # batch=1, 12 tokens
        "block_index": torch.randint(0, 5, (5, 2)),
    }
    out = model(f)
    assert len(out) == 4  # returns (a_token, q_atom, c_atom0, p_lm)
    assert out[0].shape[0] == 2  # 2 tokens expected