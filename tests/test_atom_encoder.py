import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer import AtomAttentionEncoder
from rna_predict.pipeline.stageA.input_embedding.current.embedders import InputFeatureEmbedder as AtomEncoderConfig


def test_atom_encoder_smoke():
    config = AtomEncoderConfig()
    model = AtomAttentionEncoder(config)
    # Create a minimal dict f
    f = {
        "ref_pos": torch.randn(5, 3),
        "ref_charge": torch.zeros(5),
        "ref_element": torch.randn(5, 128),
        "ref_atom_name_chars": torch.randn(5, 16),
        "atom_to_token": torch.tensor([0, 0, 1, 1, 0]),
        "restype": torch.randn(2, 32),  # minimal token dimension
        "block_index": torch.randint(0, 5, (5, 2)),
    }
    out = model(f)
    assert len(out) == 4  # returns (a_token, q_atom, c_atom0, p_lm)
    assert out[0].shape[0] == 2  # 2 tokens expected