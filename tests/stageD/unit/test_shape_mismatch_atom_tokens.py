# tests/test_shape_mismatch_atom_tokens.py

import pytest
import torch

# Adjust the import path to match your local project structure:
from rna_predict.pipeline.stageD.run_stageD import run_stageD_diffusion


@pytest.mark.xfail(reason="Shape mismatch bug expected (AssertionError in broadcast_token_to_atom).")
def test_multi_sample_shape_mismatch():
    """
    This test deliberately provides multi-sample trunk embeddings while leaving
    atom_to_token_idx at a smaller batch dimension. By default, this triggers the
    shape mismatch bug in broadcast_token_to_atom, unless the patch is applied.
    """

    # Example partial coords: shape (batch=1, 10 atoms, 3 coords)
    partial_coords = torch.randn(1, 10, 3)

    # Simulate multi-sample usage by giving s_trunk an extra leading dimension:
    # shape -> (N_sample=2, N_token=10, d=384). This means we have two "samples"
    # but haven't updated the rest of the pipeline accordingly yet.
    s_trunk = torch.randn(2, 10, 384)

    # Similarly for pair embeddings
    pair = torch.randn(2, 10, 10, 32)

    trunk_embeddings = {
        "s_trunk": s_trunk,
        "pair": pair,
    }

    # override_input_features has a smaller leading dim of 1 for atom_to_token_idx:
    # shape -> (1, 10). Mismatch: trunk_embeddings have leading dim 2, so an
    # AssertionError is expected without the patch.
    override_input_features = {
        "atom_to_token_idx": torch.arange(10).unsqueeze(0),  # shape (1, 10)
        "ref_pos": partial_coords,                           
        "ref_space_uid": torch.arange(10).unsqueeze(0),
        "ref_charge": torch.zeros(1, 10, 1),
        "ref_mask": torch.ones(1, 10, 1),
        "ref_element": torch.zeros(1, 10, 128),
        "ref_atom_name_chars": torch.zeros(1, 10, 256),
        "restype": torch.zeros(1, 10, 32),
        "profile": torch.zeros(1, 10, 32),
        "deletion_mean": torch.zeros(1, 10, 1),
    }

    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {
            "n_blocks": 4,
            "n_heads": 16
        }
    }

    # We set N_sample=2 to reflect the two-sample shape in trunk_embeddings
    inference_params = {
        "num_steps": 10,
        "sigma_max": 1.0,
        "N_sample": 2,
    }

    # Without the shape fix, this call will fail with AssertionError.
    with pytest.raises(AssertionError, match="Shape mismatch in broadcast_token_to_atom"):
        _ = run_stageD_diffusion(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device="cpu"
        )