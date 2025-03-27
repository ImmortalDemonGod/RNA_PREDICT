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
    # We'll return "sing" & "pair"
    return {
        "sing": torch.randn(1, 10, 384),
        "pair": torch.randn(1, 10, 10, 32),
    }

def stageC_mock(stageB_out):
    # Make partial coords
    coords = torch.randn(1, 10, 3)
    return coords

# @pytest.mark.integration
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
    from rna_predict.pipeline.stageD.run_stageD import run_stageD_diffusion

    trunk_embeddings = stageB_out  # includes 'sing' & 'pair'

    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {"n_blocks": 2, "n_heads": 8},
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