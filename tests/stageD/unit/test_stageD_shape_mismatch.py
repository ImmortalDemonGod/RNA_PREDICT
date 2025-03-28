import pytest
import torch

from rna_predict.pipeline.stageD.run_stageD import run_stageD_diffusion

@pytest.mark.xfail(reason="Broadcast shape mismatch before the fix.")
def test_broadcast_token_multisample_fail():
    """
    Reproduces the shape mismatch by giving s_trunk
    an extra dimension for 'samples' while leaving
    atom_to_token_idx at simpler shape [B, N_atom].
    """

    partial_coords = torch.randn(1, 10, 3)  # [B=1, N_atom=10, 3]

    # s_trunk artificially has shape [B=1, sample=1, N_token=10, c_s=384]
    trunk_embeddings = {
        "s_trunk": torch.randn(1, 1, 10, 384),
        "pair": torch.randn(1, 10, 10, 32),
    }

    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {"n_blocks": 4, "n_heads": 16},
    }

    # We attempt inference, expecting the old code to fail
    run_stageD_diffusion(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="inference",
        device="cpu"
    )