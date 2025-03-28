import pytest
import torch
from rna_predict.pipeline.stageD.run_stageD import run_stageD_diffusion
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager

# ------------------------------------------------------------------------------
# Test: Single-sample shape expansion using multi_step_inference

def test_single_sample_shape_expansion():
    """
    Ensures single-sample usage no longer triggers "Shape mismatch" assertion failures.
    We forcibly make s_trunk 4D for single-sample, then rely on the updated logic
    to expand atom_to_token_idx from [B,N_atom] to [B,1,N_atom].
    """
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {"n_blocks": 2, "n_heads": 8},
        "initialization": {}
    }
    manager = ProtenixDiffusionManager(diffusion_config, device="cpu")

    input_feature_dict = {
        "atom_to_token_idx": torch.arange(10).unsqueeze(0),  # [1,10]
        "ref_pos": torch.randn(1, 10, 3),                    # [1,10,3]
        "ref_space_uid": torch.arange(10).unsqueeze(0),      # [1,10]
    }

    # trunk_embeddings forcibly uses [B,1,N_token,c_s]
    trunk_embeddings = {
        "s_trunk": torch.randn(1, 1, 10, 384),
        "pair": torch.randn(1, 10, 10, 32),
    }

    inference_params = {"N_sample": 1, "num_steps": 2}
    coords_init = torch.randn(1, 10, 3)
    coords_final = manager.multi_step_inference(
        coords_init=coords_init,
        trunk_embeddings=trunk_embeddings,
        inference_params=inference_params,
        override_input_features=input_feature_dict,
        debug_logging=True
    )

    assert coords_final.shape == (1, 10, 3), "Final coords should remain [1, 10, 3]"

# ------------------------------------------------------------------------------
# Test: Broadcast token multisample failure (expected failure)

@pytest.mark.xfail(reason="Broadcast shape mismatch before the fix.")
def test_broadcast_token_multisample_fail():
    """
    Reproduces the shape mismatch by giving s_trunk an extra dimension for 'samples'
    while leaving atom_to_token_idx at simpler shape [B, N_atom].
    Expects an assertion failure unless the fix is applied.
    """
    partial_coords = torch.randn(1, 10, 3)  # [B=1, N_atom=10, 3]

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

    with pytest.raises(AssertionError, match="Shape mismatch in broadcast_token_to_atom"):
        _ = run_stageD_diffusion(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device="cpu"
        )

# ------------------------------------------------------------------------------
# Test: Multi-sample shape mismatch with extra sample dimension in s_trunk (expected failure)

@pytest.mark.xfail(reason="Shape mismatch bug expected (AssertionError in broadcast_token_to_atom).")
def test_multi_sample_shape_mismatch():
    """
    Deliberately provides multi-sample trunk embeddings while leaving
    atom_to_token_idx at a smaller batch dimension, expecting an assertion.
    """
    partial_coords = torch.randn(1, 10, 3)
    s_trunk = torch.randn(2, 10, 384)   # extra sample dimension
    pair = torch.randn(2, 10, 10, 32)

    trunk_embeddings = {
        "s_trunk": s_trunk,
        "pair": pair,
    }

    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {"n_blocks": 4, "n_heads": 16},
    }

    with pytest.raises(AssertionError, match="Shape mismatch in broadcast_token_to_atom"):
        _ = run_stageD_diffusion(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device="cpu"
        )

# ------------------------------------------------------------------------------
# Test: Local trunk with small number of atoms should work without shape issues

def test_local_trunk_small_natom():
    """
    Ensures no dimension mismatch when N_atom < certain thresholds.
    With the patch or correct usage, code should now pass without error.
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

    partial_coords = torch.randn(1, 10, 3, device=device)
    trunk_embeddings = {
        "sing": torch.randn(1, 10, 384, device=device),
        "pair": torch.randn(1, 10, 10, 32, device=device)
    }

    coords_final = run_stageD_diffusion(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="inference",
        device=device
    )
    assert coords_final.shape[0] == 1
    assert coords_final.shape[1] == 10
    assert coords_final.shape[2] == 3, "Should produce final coords [1, 10, 3]"

# ------------------------------------------------------------------------------
# Test: Shape mismatch due to c_token normalization mismatch (expected failure)

@pytest.mark.xfail(reason="Shape mismatch bug: c_token=832 vs leftover normalized_shape=833")
def test_shape_mismatch_c_token_832_vs_833():
    """
    Reproduces the 'RuntimeError: Given normalized_shape=[833],
    expected input with shape [*, 833], but got input of size [1, 10, 1664].'
    Usually triggered by mismatch in c_token vs. layernorm shape.
    """
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {"n_blocks": 4, "n_heads": 16},
    }

    partial_coords = torch.randn((1, 10, 3))
    trunk_embeddings = {
        "sing": torch.randn((1, 10, 384)),
        "pair": torch.randn((1, 10, 10, 32)),
    }

    with pytest.raises(RuntimeError, match=r"normalized_shape=\[833\].*got input of size.*1664"):
        run_stageD_diffusion(
            partial_coords,
            trunk_embeddings,
            diffusion_config,
            mode="inference",
            device="cpu"
        )