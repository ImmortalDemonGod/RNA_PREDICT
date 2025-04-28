import pytest
import numpy as np
import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer.encoder_components import forward_logic

# Dummy encoder object for testing (mocks required attributes)
class DummyEncoder:
    def __init__(self):
        self.c_atom = 8
        self.c_token = 4
        self.c_atompair = 4
        self.c_s = 4
        self.has_coords = True

@pytest.fixture
def dummy_input_feature_dict():
    return {
        "atom_type": torch.tensor([0, 1, 2, 3]),
        "token_type": torch.tensor([0, 1, 2, 3]),
        "coords": torch.rand(4, 3),
        "style": torch.rand(4, 4),
    }

def test_get_atom_to_token_idx_valid():
    idx = torch.tensor([0, 1, 2, 3])
    out = forward_logic.get_atom_to_token_idx({"atom_to_token_idx": idx}, num_tokens=4)
    assert torch.equal(out, idx)

def test_get_atom_to_token_idx_none():
    out = forward_logic.get_atom_to_token_idx({}, num_tokens=4)
    assert out is None

def test_process_simple_embedding(dummy_input_feature_dict):
    encoder = DummyEncoder()
    result = forward_logic._process_simple_embedding(encoder, dummy_input_feature_dict, None)
    assert isinstance(result, tuple)
    assert result[0].shape[0] == 4  # token embeddings
    assert result[1].shape[0] == 4  # atom embeddings

def test_process_coordinate_encoding(dummy_input_feature_dict):
    encoder = DummyEncoder()
    coords = dummy_input_feature_dict["coords"]
    token_emb = torch.rand(4, encoder.c_token)
    out = forward_logic._process_coordinate_encoding(encoder, coords, token_emb)
    assert out.shape[0] == 4

def test_process_style_embedding(dummy_input_feature_dict):
    encoder = DummyEncoder()
    token_emb = torch.rand(4, encoder.c_token)
    style = dummy_input_feature_dict["style"]
    idx = torch.tensor([0, 1, 2, 3])
    out = forward_logic._process_style_embedding(encoder, token_emb, style, idx)
    assert out.shape[0] == 4

def test_aggregate_to_token_level(dummy_input_feature_dict):
    encoder = DummyEncoder()
    atom_emb = torch.rand(4, encoder.c_atom)
    idx = torch.tensor([0, 1, 2, 3])
    out = forward_logic._aggregate_to_token_level(encoder, atom_emb, idx, 4)
    assert out.shape[0] == 4

def test_process_inputs_with_coords(dummy_input_feature_dict):
    encoder = DummyEncoder()
    from rna_predict.pipeline.stageA.input_embedding.current.transformer.encoder_components.forward_logic import ProcessInputsParams
    params = ProcessInputsParams(
        input_feature_dict=dummy_input_feature_dict,
        r_l=None, s=None, z=None, c_l=None, chunk_size=None
    )
    out = forward_logic.process_inputs_with_coords(encoder, params)
    assert isinstance(out, tuple)
    assert out[0].shape[0] == 4
