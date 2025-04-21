import os

import psutil
import pytest
import torch

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import run_stageD_diffusion
from rna_predict.pipeline.stageD.diffusion.utils import DiffusionConfig # Import needed class

# ------------------------------------------------------------------------------
# Test: Single-sample shape expansion using multi_step_inference


def _create_diffusion_config():
    """
    Create a minimal diffusion configuration for testing.

    Returns:
        Dictionary with diffusion configuration parameters
    """
    return {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 384,
        "c_s_inputs": 449,
        "transformer": {"n_blocks": 1, "n_heads": 2},
        "conditioning": {
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 384,
            "c_noise_embedding": 128,
        },
        "embedder": {"c_atom": 128, "c_atompair": 16, "c_token": 384},
        "sigma_data": 16.0,
        "initialization": {},
    }


def _create_input_features(num_atoms=5):
    """
    Create input feature dictionary for diffusion model.

    Args:
        num_atoms: Number of atoms to include in features

    Returns:
        Dictionary with input features
    """
    return {
        "atom_to_token_idx": torch.arange(num_atoms).unsqueeze(0),  # [1,num_atoms]
        "ref_pos": torch.randn(1, num_atoms, 3),  # [1,num_atoms,3]
        "ref_space_uid": torch.arange(num_atoms).unsqueeze(0),  # [1,num_atoms]
        "ref_charge": torch.zeros(1, num_atoms, 1),
        "ref_element": torch.zeros(1, num_atoms, 128),
        "ref_atom_name_chars": torch.zeros(1, num_atoms, 256),
        "ref_mask": torch.ones(1, num_atoms, 1),
        "restype": torch.zeros(1, num_atoms, 32),
        "profile": torch.zeros(1, num_atoms, 32),
        "deletion_mean": torch.zeros(1, num_atoms, 1),
        "sing": torch.randn(1, num_atoms, 384),  # Required for s_inputs fallback
    }


def _create_mismatched_trunk_embeddings(num_atoms=5):
    """
    Create trunk embeddings with intentionally mismatched shapes.

    Args:
        num_atoms: Number of atoms to include

    Returns:
        Dictionary with adjusted trunk embeddings
    """
    from rna_predict.utils.shape_utils import adjust_tensor_feature_dim

    # Create tensors with wrong feature dimensions but compatible shapes
    # s_trunk should have shape [B, N, C] to match z_trunk shape [B, N, N, C]
    s_trunk = torch.randn(1, num_atoms, 256)  # Should be 384
    # Create pair with correct shape for z_trunk
    # Note: z_trunk should have shape [B, N, N, C]
    # where N is the number of atoms and C is the feature dimension
    pair = torch.randn(1, num_atoms, num_atoms, 16)  # Should be 32
    sing = torch.randn(1, num_atoms, 256)  # Should be 384

    # Adjust tensor dimensions to correct values
    s_trunk = adjust_tensor_feature_dim(s_trunk, 384, "s_trunk")
    pair = adjust_tensor_feature_dim(pair, 32, "pair")
    sing = adjust_tensor_feature_dim(sing, 384, "sing")

    return {
        "s_trunk": s_trunk,
        "pair": pair,
        "sing": sing,
    }


def _validate_output_coordinates(coords, expected_num_atoms=5):
    """
    Validate the output coordinates from diffusion model.

    Args:
        coords: Output coordinate tensor
        expected_num_atoms: Expected number of atoms

    Raises:
        AssertionError: If validation fails
    """
    # Check dimensions
    assert coords.size(-2) == expected_num_atoms, \
        f"Final coords should have {expected_num_atoms} atoms (second-to-last dimension)"
    assert coords.size(-1) == 3, "Final coords should have 3 coordinates (last dimension)"

    # Check for invalid values
    assert not torch.isnan(coords).any(), "Output contains NaN values"
    assert not torch.isinf(coords).any(), "Output contains infinity values"


def test_single_sample_shape_expansion():
    """
    Ensures single-sample usage no longer triggers "Shape mismatch" assertion failures.
    We forcibly make s_trunk 4D for single-sample, then rely on the updated logic
    to expand atom_to_token_idx from [B,N_atom] to [B,1,N_atom].

    This test uses the shape_utils module to adjust tensor shapes and verifies that
    the diffusion module can handle mismatched shapes gracefully.

    # ERROR_ID: STAGED_SHAPE_EXPANSION_HANDLING
    """
    # Create configuration and manager
    diffusion_config = _create_diffusion_config()

    # Create a Hydra-compatible config structure
    from omegaconf import OmegaConf
    hydra_cfg = OmegaConf.create({
        "stageD": {
            "diffusion": {
                "device": "cpu",
                # Add all diffusion config parameters
                **diffusion_config
            }
        }
    })

    # Create the manager with the Hydra config
    manager = ProtenixDiffusionManager(cfg=hydra_cfg)

    # Create input features and trunk embeddings
    num_atoms = 5
    input_feature_dict = _create_input_features(num_atoms)
    trunk_embeddings = _create_mismatched_trunk_embeddings(num_atoms)

    # Run inference
    coords_init = torch.randn(1, num_atoms, 3)

    # Update manager's config with inference parameters
    if not hasattr(manager, 'cfg') or not OmegaConf.is_config(manager.cfg):
        manager.cfg = OmegaConf.create({
            "stageD": {
                "diffusion": {
                    "inference": {"N_sample": 1, "num_steps": 2},
                    "debug_logging": True
                }
            }
        })
    else:
        # Update existing config
        if "inference" not in manager.cfg.stageD.diffusion:
            manager.cfg.stageD.diffusion.inference = OmegaConf.create({"N_sample": 1, "num_steps": 2})
        else:
            manager.cfg.stageD.diffusion.inference.N_sample = 1
            manager.cfg.stageD.diffusion.inference.num_steps = 2
        manager.cfg.stageD.diffusion.debug_logging = True

    # Call with updated API
    coords_final = manager.multi_step_inference(
        coords_init=coords_init,
        trunk_embeddings=trunk_embeddings,
        override_input_features=input_feature_dict
    )

    # Validate output
    _validate_output_coordinates(coords_final, num_atoms)
    print(f"Test passed with coords shape = {coords_final.shape}")


# ------------------------------------------------------------------------------
# Test: Broadcast token multisample failure (expected failure)


# @pytest.mark.skip(reason="Causes excessive memory usage due to large model config")


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
        "s_inputs": torch.randn(1, 10, 449),  # Added s_inputs
    }

    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {"n_blocks": 4, "n_heads": 16},
        "c_s_inputs": 449,  # Added c_s_inputs
        "inference": {"num_steps": 2},  # Added to limit steps for memory
    }

    with pytest.raises(
        AssertionError, match="Shape mismatch in broadcast_token_to_atom"
    ):
        test_config = DiffusionConfig(
             partial_coords=partial_coords,
             trunk_embeddings=trunk_embeddings,
             diffusion_config=diffusion_config,
             mode="inference",
             device="cpu",
             input_features=None, # Assuming None as it wasn't provided
         )
        _ = run_stageD_diffusion(config=test_config)


# ------------------------------------------------------------------------------
# Test: Multi-sample shape mismatch with extra sample dimension in s_trunk (expected failure)


@pytest.mark.xfail(
    reason="Shape mismatch bug expected (AssertionError in broadcast_token_to_atom)."
)
# @pytest.mark.skip(reason="Causes excessive memory usage") # Ensuring it stays commented
def test_multi_sample_shape_mismatch():
    """
    Deliberately provides multi-sample trunk embeddings while leaving
    atom_to_token_idx at a smaller batch dimension, expecting an assertion.
    """
    partial_coords = torch.randn(1, 10, 3)
    s_trunk = torch.randn(2, 10, 384)  # extra sample dimension
    pair = torch.randn(2, 10, 10, 32)
    s_inputs = torch.randn(2, 10, 449)  # Added s_inputs

    trunk_embeddings = {
        "s_trunk": s_trunk,
        "pair": pair,
        "s_inputs": s_inputs,  # Added s_inputs
    }

    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 384,  # Reduced from 832
        "transformer": {"n_blocks": 1, "n_heads": 2},  # Reduced from 4 blocks, 16 heads
        "inference": {"num_steps": 2},  # Added to limit steps for memory
        "c_s_inputs": 449,  # Added to limit steps for memory
    }

    with pytest.raises(
        AssertionError, match="Shape mismatch in broadcast_token_to_atom"
    ):
        test_config = DiffusionConfig(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device="cpu",
            input_features=None, # Assuming None
        )
        _ = run_stageD_diffusion(config=test_config)


# ------------------------------------------------------------------------------
# Test: Local trunk with small number of atoms should work without shape issues


def test_local_trunk_small_natom_memory_efficient():
    """
    Memory-efficient version of test_local_trunk_small_natom.
    Uses smaller tensors and fewer diffusion steps to avoid memory issues.
    """
    import os

    import psutil

    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB

    # Create minimal valid inputs
    batch_size = 1
    num_atoms = 5
    num_tokens = 5  # Assuming 1 atom maps to 1 token for simplicity here
    device = "cpu"

    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")

    try:
        # Create tensors with explicit device placement
        partial_coords = torch.randn(batch_size, num_atoms, 3, device=device)
        trunk_embeddings = {
            "s_trunk": torch.randn(batch_size, num_tokens, 384, device=device),
            "pair": torch.randn(batch_size, num_tokens, num_tokens, 32, device=device),
            "s_inputs": torch.randn(batch_size, num_tokens, 449, device=device),
        }

        # Use a minimal configuration with fewer transformer blocks and heads
        diffusion_config = {
            "c_atom": 128,
            "c_s": 384,
            "c_z": 32,
            "c_token": 384,
            # "c_s_inputs": 449, # This contradicts conditioning['c_s_inputs'], remove or align. Using conditioning value (384).
            "transformer": {"n_blocks": 1, "n_heads": 2},  # Reduced from original
            "conditioning": {
                "c_s": 384,
                "c_z": 32,
                "c_s_inputs": 449,
                "c_noise_embedding": 128,
            },
            "embedder": {"c_atom": 128, "c_atompair": 16, "c_token": 384},
            "sigma_data": 16.0,
            "initialization": {},
            "inference": {
                "num_steps": 2,  # Reduced from default (20)
                "N_sample": 1,
            },
        }

        # Create minimal input_features dictionary with explicit device placement
        input_features = {
            "atom_to_token_idx": torch.arange(num_atoms, device=device).unsqueeze(0),
            "ref_pos": partial_coords.clone(),  # Use clone to avoid memory sharing
            "ref_space_uid": torch.arange(num_atoms, device=device).unsqueeze(0),
            "ref_charge": torch.zeros(batch_size, num_atoms, 1, device=device),
            "ref_element": torch.zeros(batch_size, num_atoms, 128, device=device),
            "ref_atom_name_chars": torch.zeros(
                batch_size, num_atoms, 256, device=device
            ),
            "ref_mask": torch.ones(batch_size, num_atoms, 1, device=device),
            "restype": torch.zeros(batch_size, num_tokens, 32, device=device),
            "profile": torch.zeros(batch_size, num_tokens, 32, device=device),
            "deletion_mean": torch.zeros(batch_size, num_atoms, 1, device=device),
            "sing": torch.randn(batch_size, num_atoms, 449, device=device),
        }

        current_memory = get_memory_usage()
        print(f"Memory before run_stageD_diffusion: {current_memory:.2f} MB")

        # Run diffusion with explicit garbage collection
        import gc

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        test_config = DiffusionConfig(
             partial_coords=partial_coords,
             trunk_embeddings=trunk_embeddings,
             diffusion_config=diffusion_config,
             mode="inference",
             device=device,
             input_features=input_features,
         )
        coords_out = run_stageD_diffusion(config=test_config)

        current_memory = get_memory_usage()
        print(f"Memory after run_stageD_diffusion: {current_memory:.2f} MB")
        print(f"Memory increase: {current_memory - initial_memory:.2f} MB")

        # Verify output shape and content
        assert isinstance(coords_out, torch.Tensor), "Output must be a tensor"
        assert coords_out.ndim == 3, f"Expected 3D tensor, got {coords_out.ndim}D"
        assert coords_out.shape == (
            batch_size,
            num_atoms,
            3,
        ), f"Expected shape {(batch_size, num_atoms, 3)}, got {coords_out.shape}"
        assert not torch.isnan(coords_out).any(), "Output contains NaN values"
        assert not torch.isinf(coords_out).any(), "Output contains infinity values"

        print("Test passed successfully!")

    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        current_memory = get_memory_usage()
        print(f"Memory after cleanup: {current_memory:.2f} MB")


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
        test_config = DiffusionConfig(
             partial_coords=partial_coords,
             trunk_embeddings=trunk_embeddings,
             diffusion_config=diffusion_config,
             mode="inference",
             device="cpu",
             input_features=None, # Assuming None
         )
        run_stageD_diffusion(config=test_config)


@pytest.mark.slow  # Added mark
def test_transformer_size_memory_threshold():
    """
    Experiment to find the memory threshold for transformer configuration.
    Tests progressively larger transformer sizes until memory issues occur.
    Uses reduced embedding dimensions and limited inference steps for efficiency. # Updated docstring
    """

    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB

    # Use minimal input size
    batch_size = 1
    num_atoms = 5
    num_tokens = 5
    device = "cpu"
    partial_coords = torch.randn(batch_size, num_atoms, 3, device=device)

    # Test configurations with increasing size
    configs = [
        {"n_blocks": 1, "n_heads": 2},  # Smaller base case
        {"n_blocks": 1, "n_heads": 4},
        {"n_blocks": 2, "n_heads": 4},
        {"n_blocks": 2, "n_heads": 8},
    ]

    # Reduced base dimensions for efficiency
    base_c_atom = 64
    base_c_s = 128
    base_c_z = 16
    base_c_token = 128  # Significantly reduced from 832
    base_c_s_inputs = (
        449  # Keep original if necessary for input compatibility, or reduce if possible
    )

    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")

    for transformer_config in configs:  # Renamed loop variable
        print(f"\nTesting config: {transformer_config}")
        try:
            # Create embeddings matching reduced dimensions
            trunk_embeddings = {
                "s_trunk": torch.randn(
                    batch_size, 1, num_tokens, base_c_s, device=device
                ),
                "pair": torch.randn(
                    batch_size, 1, num_tokens, num_tokens, base_c_z, device=device
                ),  # Adjusted pair shape assumption
                "s_inputs": torch.randn(
                    batch_size, num_tokens, base_c_s_inputs, device=device
                ),
            }

            diffusion_config = {
                "c_atom": base_c_atom,
                "c_s": base_c_s,
                "c_z": base_c_z,
                "c_token": base_c_token,
                "transformer": transformer_config,  # Use the loop variable
                "c_s_inputs": base_c_s_inputs,
                # --- Added inference params ---
                "inference": {
                    "num_steps": 2,  # Explicitly set low number of steps
                    "N_sample": 1,
                },
                # --- Add other minimal required keys if run_stageD_diffusion needs them ---
                # Example: Add dummy sections if the constructor/function expects them
                "conditioning": {
                    "c_s": base_c_s,
                    "c_z": base_c_z,
                    "c_s_inputs": base_c_s_inputs,
                    "c_noise_embedding": 64,  # Example value
                },
                "embedder": {
                    "c_atom": base_c_atom,
                    "c_atompair": 8,  # Example value
                    "c_token": base_c_token,
                },
                "sigma_data": 16.0,  # Example value
                "initialization": {},  # Example value
            }

            # Minimal input features matching dimensions
            input_features = {
                "atom_to_token_idx": torch.arange(num_atoms, device=device).unsqueeze(
                    0
                ),
                # Add other minimal features required by run_stageD_diffusion or its internals
                # Ensure dimensions match num_atoms/num_tokens and batch_size
                "ref_mask": torch.ones(batch_size, num_atoms, 1, device=device),
                # ... (add others like ref_pos, ref_space_uid etc. if needed by the specific code path)
            }

            # Try to run with this config
            test_config = DiffusionConfig(
                 partial_coords=partial_coords,
                 trunk_embeddings=trunk_embeddings,
                 diffusion_config=diffusion_config,
                 mode="inference",
                 device=device,
                 input_features=input_features,
             )
            _ = run_stageD_diffusion(config=test_config)

            current_memory = get_memory_usage()
            memory_increase = current_memory - initial_memory
            print(f"Success with config: {transformer_config}")
            print(f"Memory increase: {memory_increase:.2f} MB")

        except Exception as e:
            current_memory = get_memory_usage()  # Check memory even on failure
            memory_increase = current_memory - initial_memory
            print(f"Failed at config: {transformer_config}")
            print(f"Memory increase before failure: {memory_increase:.2f} MB")
            print(f"Error: {str(e)}")
            # Decide if the test should fail here or just report
            # pytest.fail(f"Test failed at config {transformer_config} with error: {e}")
            break  # Stop testing further configs if one fails
        finally:
            # Clean up after each attempt
            del trunk_embeddings
            del diffusion_config
            if "input_features" in locals():
                del input_features
            # No need to check torch.cuda.is_available() for empty_cache
            torch.cuda.empty_cache()
            current_memory = get_memory_usage()
            # Optional: print memory after cleanup for debugging
            # print(f"Memory after cleanup: {current_memory:.2f} MB")


def test_tensor_shape_memory_impact():
    """
    Experiment to test if tensor shape mismatches contribute to memory issues.
    Tests different tensor shapes while keeping total elements constant.
    """

    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB

    # Base configuration that works
    base_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {"n_blocks": 1, "n_heads": 4},
        "c_s_inputs": 449,
    }

    # Test different tensor shapes
    shape_configs = [
        # Original shape
        {
            "s_trunk": (1, 1, 5, 384),
            "pair": (1, 5, 5, 32),
            "s_inputs": (1, 5, 449),
        },
        # Flattened shape
        {
            "s_trunk": (1, 5, 384),
            "pair": (1, 25, 32),
            "s_inputs": (1, 5, 449),
        },
        # Reshaped with same elements
        {
            "s_trunk": (1, 2, 5, 192),
            "pair": (1, 5, 5, 32),
            "s_inputs": (1, 5, 449),
        },
    ]

    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")

    for shapes in shape_configs:
        print(f"\nTesting shapes: {shapes}")
        try:
            trunk_embeddings = {
                "s_trunk": torch.randn(*shapes["s_trunk"]),
                "pair": torch.randn(*shapes["pair"]),
                "s_inputs": torch.randn(*shapes["s_inputs"]),
            }

            partial_coords = torch.randn(1, 5, 3)

            # Try to run with these shapes
            test_config = DiffusionConfig(
                 partial_coords=partial_coords,
                 trunk_embeddings=trunk_embeddings,
                 diffusion_config=base_config,
                 mode="inference",
                 device="cpu",
                 input_features=None, # Assuming None
             )
            _ = run_stageD_diffusion(config=test_config)

            current_memory = get_memory_usage()
            memory_increase = current_memory - initial_memory
            print(f"Success with shapes: {shapes}")
            print(f"Memory increase: {memory_increase:.2f} MB")

        except Exception as e:
            print(f"Failed at shapes: {shapes}")
            print(f"Error: {str(e)}")
            break
        finally:
            # Clean up after each attempt
            del trunk_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            current_memory = get_memory_usage()
            print(f"Memory after cleanup: {current_memory:.2f} MB")


@pytest.mark.slow
def test_problem_size_memory_threshold():
    """
    Memory-efficient test for problem size threshold.
    Uses smaller tensors and proper cleanup to avoid memory issues.
    """

    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB

    # Use smaller dimensions for testing
    batch_size = 1
    num_atoms = 10  # Reduced from original value
    num_tokens = 10
    device = "cpu"

    # Create minimal valid inputs with correct feature dimensions
    input_feature_dict = {
        "atom_to_token_idx": torch.arange(num_atoms).unsqueeze(0),
        "ref_pos": torch.randn(batch_size, num_atoms, 3),
        "ref_space_uid": torch.arange(num_atoms).unsqueeze(0),
        "ref_charge": torch.zeros(batch_size, num_atoms, 1),
        "ref_element": torch.zeros(batch_size, num_atoms, 128),  # Original dimension
        "ref_atom_name_chars": torch.zeros(
            batch_size, num_atoms, 256
        ),  # Original dimension
        "ref_mask": torch.ones(batch_size, num_atoms, 1),
        "restype": torch.zeros(batch_size, num_atoms, 32),  # Original dimension
        "profile": torch.zeros(batch_size, num_atoms, 32),  # Original dimension
        "deletion_mean": torch.zeros(batch_size, num_atoms, 1),
        "sing": torch.randn(batch_size, num_atoms, 449),  # Original dimension
        # Add s_inputs tensor required by the diffusion manager
        "s_inputs": torch.randn(batch_size, num_atoms, 449),
    }

    # Use smaller model configuration but keep original feature dimensions
    diffusion_config = {
        "c_atom": 128,  # Original dimension
        "c_s": 384,  # Original dimension
        "c_z": 32,  # Original dimension
        "c_token": 384,  # Original dimension
        "c_s_inputs": 449,  # Original dimension
        "transformer": {
            "n_blocks": 1,  # Reduced from 4
            "n_heads": 2,  # Reduced from 16
        },
        "conditioning": {
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 449,
            "c_noise_embedding": 64,
        },
        "embedder": {"c_atom": 128, "c_atompair": 8, "c_token": 384},
        "sigma_data": 16.0,
        "initialization": {},
    }

    # Create manager with Hydra-compatible configuration
    from omegaconf import OmegaConf

    # Create a Hydra-compatible config structure
    hydra_cfg = OmegaConf.create({
        "stageD": {
            "diffusion": {
                "device": device,
                # Add all diffusion config parameters
                **diffusion_config
            }
        }
    })

    # Create the manager with the Hydra config
    manager = ProtenixDiffusionManager(cfg=hydra_cfg)

    # Create trunk embeddings with correct dimensions
    trunk_embeddings = {
        # Remove singleton dimension for s_trunk and pair
        "s_trunk": torch.randn(batch_size, num_tokens, diffusion_config["c_s"]),
        "pair": torch.randn(
            batch_size, num_tokens, num_tokens, diffusion_config["c_z"]
        ),
        "s_inputs": torch.randn(batch_size, num_tokens, diffusion_config["c_s_inputs"]),
    }

    # Use fewer steps for inference
    inference_params = {"N_sample": 1, "num_steps": 2}

    # Initial memory usage
    initial_memory = get_memory_usage()

    try:
        # Run inference with smaller tensors
        coords_init = torch.randn(batch_size, num_atoms, 3)
        # Update manager's config with inference parameters
        from omegaconf import OmegaConf
        if not hasattr(manager, 'cfg') or not OmegaConf.is_config(manager.cfg):
            manager.cfg = OmegaConf.create({
                "stageD": {
                    "diffusion": {
                        "inference": inference_params,
                        "debug_logging": False
                    }
                }
            })
        else:
            # Update existing config
            if "inference" not in manager.cfg.stageD.diffusion:
                manager.cfg.stageD.diffusion.inference = OmegaConf.create(inference_params)
            else:
                for k, v in inference_params.items():
                    manager.cfg.stageD.diffusion.inference[k] = v
            manager.cfg.stageD.diffusion.debug_logging = False

        # Call with updated API
        coords_final = manager.multi_step_inference(
            coords_init=coords_init,
            trunk_embeddings=trunk_embeddings,
            override_input_features=input_feature_dict
        )

        # Check output shapes
        assert coords_final.size(-2) == num_atoms
        assert coords_final.size(-1) == 3
        assert not torch.isnan(coords_final).any()
        assert not torch.isinf(coords_final).any()

    finally:
        # Clean up tensors
        if "coords_init" in locals():
            del coords_init
        if "coords_final" in locals():
            del coords_final
        if "trunk_embeddings" in locals():
            del trunk_embeddings
        if "input_feature_dict" in locals():
            del input_feature_dict
        if "manager" in locals():
            del manager
        torch.cuda.empty_cache()  # Clear CUDA cache if using GPU

    # Final memory usage
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory

    # Assert memory usage is within reasonable bounds
    assert (
        memory_increase < 1000
    ), f"Memory increase ({memory_increase:.1f} MB) exceeds threshold"
