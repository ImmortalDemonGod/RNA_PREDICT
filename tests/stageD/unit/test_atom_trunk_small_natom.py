import pytest
import torch
from rna_predict.pipeline.stageD.run_stageD import run_stageD_diffusion

def test_local_trunk_small_natom():
    """
    Ensures we no longer get a dimension mismatch when N_atom < 32 or 128.
    With the patch applied, the code should now pass without error.
    """
    device = torch.device("cpu")
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "transformer": {
            "n_blocks": 4,
            "n_heads": 16
        }
    }

    # partial_coords => shape [B=1, N_atom=10, 3]
    partial_coords = torch.randn(1, 10, 3, device=device)

    # Minimal trunk embeddings
    trunk_embeddings = {
        "sing": torch.randn(1, 10, 384, device=device),
        "pair": torch.randn(1, 10, 10, 32, device=device)
    }

    # Attempt an inference pass; with the patch, no dimension error should occur
    coords_final = run_stageD_diffusion(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="inference",
        device=device
    )
    assert coords_final.shape[0] == 1
    assert coords_final.shape[1] == 10
    assert coords_final.shape[2] == 3, "Should produce final coords with shape [1, 10, 3]"