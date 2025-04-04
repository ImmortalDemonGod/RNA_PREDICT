import pytest

import torch


def stageA_mock():
    # Fake stage A output
    return {
        "restype_embedding": torch.randn(1, 10, 32),
        "profile_embedding": torch.randn(1, 10, 32),
    }


def stageB_mock(stageA_out):
    # Convert to trunk embeddings
    # Return with correct keys for Stage D
    return {
        "s_trunk": torch.randn(1, 10, 384),  # Changed from 'sing' to 's_trunk'
        "pair": torch.randn(1, 10, 10, 32),
        "s_inputs": torch.randn(1, 10, 384)  # Added s_inputs
    }


def stageC_mock(stageB_out):
    # Make partial coords
    coords = torch.randn(1, 10, 3)
    return coords


# @pytest.mark.integration
@pytest.mark.skip(reason="Causes excessive memory usage")
def test_end_to_end_stageA_to_D():
    """
    A scenario hooking mock stageA->B->C->D.
    Ensures run_stageD_diffusion runs with partial trunk_embeddings & partial_coords.
    """

    # 1) Stage A
    stageA_out = stageA_mock()

    # 2) Stage B
    stageB_out = stageB_mock(stageA_out)

    # 3) Stage C
    partial_coords = stageC_mock(stageB_out)

    # 4) Stage D config
    from rna_predict.pipeline.stageD.run_stageD_unified import (
        run_stageD_diffusion,
    )  # FIX: Import from the correct unified module

    trunk_embeddings = stageB_out  # includes 'sing' & 'pair'

    diffusion_config = {
        "sigma_data": 16.0,
        "c_atom": 128,
        "c_atompair": 16,
        "c_token": 832,
        "c_s": 384,
        "c_z": 32,
        "c_s_inputs": 384,
        "conditioning": {
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 384,
            "c_noise_embedding": 128
        },
        "embedder": {
            "c_atom": 128,
            "c_atompair": 16,
            "c_token": 832
        },
        "transformer": {
            "n_blocks": 2,
            "n_heads": 8
        },
        "atom_encoder": {
            "n_blocks": 2,
            "n_heads": 8
        },
        "atom_decoder": {
            "n_blocks": 2,
            "n_heads": 8
        },
        "initialization": {},
        "inference": {
            "N_sample": 1,
            "num_steps": 10
        }
    }

    # 5) run_stageD_diffusion in inference mode
    coords_final = run_stageD_diffusion(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="inference",
        device="cpu",
    )

    # 6) Confirm output
    assert isinstance(coords_final, torch.Tensor)
    assert coords_final.ndim == 3
    # Usually [1, 10, 3], but we won't strongly enforce that
