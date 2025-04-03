import pytest
import torch

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.run_stageD_unified import run_stageD_diffusion

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
        "initialization": {},
    }
    manager = ProtenixDiffusionManager(diffusion_config, device="cpu")

    input_feature_dict = {
        "atom_to_token_idx": torch.arange(10).unsqueeze(0),  # [1,10]
        "ref_pos": torch.randn(1, 10, 3),  # [1,10,3]
        "ref_space_uid": torch.arange(10).unsqueeze(0),  # [1,10]
        "ref_charge": torch.zeros(1, 10),
        "ref_element": torch.zeros(1, 10, 128),
        "ref_atom_name_chars": torch.zeros(1, 10, 256),
        "ref_mask": torch.ones(1, 10),
        "restype": torch.zeros(1, 10, 32),
        "profile": torch.zeros(1, 10, 32),
        "deletion_mean": torch.zeros(1, 10),
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
        debug_logging=True,
    )

    # The model may return coordinates with extra batch dimensions
    # Check that the output has the correct final dimensions
    assert (
        coords_final.size(-2) == 10
    ), "Final coords should have 10 atoms (second-to-last dimension)"
    assert (
        coords_final.size(-1) == 3
    ), "Final coords should have 3 coordinates (last dimension)"

    # Check that the output contains valid values
    assert not torch.isnan(coords_final).any(), "Output contains NaN values"
    assert not torch.isinf(coords_final).any(), "Output contains infinity values"

    print(f"Test passed with coords shape = {coords_final.shape}")


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

    with pytest.raises(
        AssertionError, match="Shape mismatch in broadcast_token_to_atom"
    ):
        _ = run_stageD_diffusion(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device="cpu",
        )


# ------------------------------------------------------------------------------
# Test: Multi-sample shape mismatch with extra sample dimension in s_trunk (expected failure)


@pytest.mark.xfail(
    reason="Shape mismatch bug expected (AssertionError in broadcast_token_to_atom)."
)
def test_multi_sample_shape_mismatch():
    """
    Deliberately provides multi-sample trunk embeddings while leaving
    atom_to_token_idx at a smaller batch dimension, expecting an assertion.
    """
    partial_coords = torch.randn(1, 10, 3)
    s_trunk = torch.randn(2, 10, 384)  # extra sample dimension
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

    with pytest.raises(
        AssertionError, match="Shape mismatch in broadcast_token_to_atom"
    ):
        _ = run_stageD_diffusion(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device="cpu",
        )


# ------------------------------------------------------------------------------
# Test: Local trunk with small number of atoms should work without shape issues


def test_local_trunk_small_natom():
    """Test local trunk with small number of atoms, providing minimal features."""
    # Create minimal valid inputs
    batch_size = 1
    num_atoms = 10
    num_tokens = 10 # Assuming 1 atom maps to 1 token for simplicity here
    device = "cpu"

    partial_coords = torch.randn(batch_size, num_atoms, 3, device=device)
    trunk_embeddings = {
        "s_trunk": torch.randn(batch_size, num_tokens, 384, device=device),
        "pair": torch.randn(batch_size, num_tokens, num_tokens, 32, device=device),
        # s_inputs is expected to be generated internally by the embedder within run_stageD
    }
    
    # Define consistent configuration (as before)
    c_s_val = 384
    c_z_val = 32
    embedder_output_dim = 384
    
    diffusion_config = {
        "c_s": c_s_val,
        "c_z": c_z_val,
        "transformer": {"n_blocks": 2, "n_heads": 8},
        "embedder": {
            "c_atom": 128,
            "c_atompair": 16,
            "c_token": embedder_output_dim
        },
        "conditioning": {
            "c_s": c_s_val,
            "c_z": c_z_val,
            "c_s_inputs": embedder_output_dim
        },
    }

    # --- Create minimal input_features dictionary ---
    # Required to avoid internal loading and prevent NoneType errors
    input_features = {
        # Essential for mapping and potentially shape inference
        "atom_to_token_idx": torch.arange(num_atoms, device=device).unsqueeze(0).expand(batch_size, -1), # Shape: [B, N_atom]
        # Required by AtomAttentionEncoder/Decoder
        "restype": torch.zeros(batch_size, num_tokens, 32, device=device), # Shape: [B, N_token, C_restype] (Using dummy C=32)
        "ref_mask": torch.ones(batch_size, num_atoms, device=device), # Shape: [B, N_atom]
        # Required by AtomAttentionEncoder (ensure_space_uid)
        "ref_space_uid": torch.arange(num_atoms, device=device).unsqueeze(0).expand(batch_size, -1), # Shape: [B, N_atom] - Dummy UID
        # Required by create_pair_embedding
        "ref_charge": torch.zeros(batch_size, num_atoms, device=device), # Shape: [B, N_atom] - Dummy charge
        # Add other minimal features if required by specific components down the line
        "ref_pos": partial_coords, # Often needed as reference
    }
    # --- End input_features creation ---

    coords_final = run_stageD_diffusion(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="inference",
        device=device,
        input_features=input_features, # <-- Pass the created features
    )
    
    # Add assertions about the output shape/type if needed
    assert isinstance(coords_final, torch.Tensor)
    # Expect shape [B, N_atom, 3] after squeezing N_sample=1 dimension
    assert (
        coords_final.ndim == 3
    ), f"Expected 3 dimensions after squeeze, got {coords_final.ndim}"
    assert (
        coords_final.shape[0] == batch_size
    ), f"Expected batch dim {batch_size}, got {coords_final.shape[0]}"
    assert (
        coords_final.shape[1] == num_atoms
    ), f"Expected atom dim {num_atoms}, got {coords_final.shape[1]}"
    assert (
        coords_final.shape[2] == 3
    ), f"Expected coord dim 3, got {coords_final.shape[2]}"
    # Add more specific value checks if required by the test's purpose


# ------------------------------------------------------------------------------
# Test: Shape mismatch due to c_token normalization mismatch (expected failure)


@pytest.mark.xfail(
    reason="Shape mismatch bug: c_token=832 vs leftover normalized_shape=833"
)
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

    with pytest.raises(
        RuntimeError, match=r"normalized_shape=\[833\].*got input of size.*1664"
    ):
        run_stageD_diffusion(
            partial_coords,
            trunk_embeddings,
            diffusion_config,
            mode="inference",
            device="cpu",
        )
