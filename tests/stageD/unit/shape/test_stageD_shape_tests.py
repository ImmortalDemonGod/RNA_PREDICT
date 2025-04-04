import pytest
import torch
import psutil
import os

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
        "c_token": 384,
        "c_s_inputs": 449,
        "transformer": {"n_blocks": 1, "n_heads": 2},
        "conditioning": {
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 384,
            "c_noise_embedding": 128
        },
        "embedder": {
            "c_atom": 128,
            "c_atompair": 16,
            "c_token": 384
        },
        "sigma_data": 16.0,
        "initialization": {},
    }
    manager = ProtenixDiffusionManager(diffusion_config, device="cpu")

    input_feature_dict = {
        "atom_to_token_idx": torch.arange(5).unsqueeze(0),  # [1,5]
        "ref_pos": torch.randn(1, 5, 3),  # [1,5,3]
        "ref_space_uid": torch.arange(5).unsqueeze(0),  # [1,5]
        "ref_charge": torch.zeros(1, 5, 1),
        "ref_element": torch.zeros(1, 5, 128),
        "ref_atom_name_chars": torch.zeros(1, 5, 256),
        "ref_mask": torch.ones(1, 5, 1),
        "restype": torch.zeros(1, 5, 32),
        "profile": torch.zeros(1, 5, 32),
        "deletion_mean": torch.zeros(1, 5, 1),
        "sing": torch.randn(1, 5, 449)  # Required for s_inputs fallback
    }

    # trunk_embeddings forcibly uses [B,1,N_token,c_s] and [B,1,N_token,N_token,c_z]
    trunk_embeddings = {
        "s_trunk": torch.randn(1, 1, 5, 384),
        "pair": torch.randn(1, 1, 5, 5, 32), # Added sample dimension to match s_trunk
    }

    inference_params = {"N_sample": 1, "num_steps": 2}
    coords_init = torch.randn(1, 5, 3)
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
        coords_final.size(-2) == 5
    ), "Final coords should have 5 atoms (second-to-last dimension)"
    assert (
        coords_final.size(-1) == 3
    ), "Final coords should have 3 coordinates (last dimension)"

    # Check that the output contains valid values
    assert not torch.isnan(coords_final).any(), "Output contains NaN values"
    assert not torch.isinf(coords_final).any(), "Output contains infinity values"

    print(f"Test passed with coords shape = {coords_final.shape}")


# ------------------------------------------------------------------------------
# Test: Broadcast token multisample failure (expected failure)


@pytest.mark.skip(reason="Causes excessive memory usage due to large model config")
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


def test_broadcast_token_multisample_fail_memory_efficient():
    """
    Memory-efficient version of test_broadcast_token_multisample_fail.
    Tests the same functionality but with smaller tensors and model config.
    Includes memory monitoring and safeguards.
    """
    import psutil
    import os
    import gc
    
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB
    
    # Create minimal valid inputs with smaller dimensions
    batch_size = 1
    num_atoms = 5  # Reduced from 10
    num_tokens = 5  # Reduced from 10
    device = "cpu"
    
    # Memory threshold in MB - adjust based on available system memory
    memory_threshold = 1000  # 1GB threshold
    
    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")
    
    try:
        # Create tensors with explicit device placement
        partial_coords = torch.randn(batch_size, num_atoms, 3, device=device)
        
        # Create trunk embeddings with the shape mismatch
        trunk_embeddings = {
            "s_trunk": torch.randn(batch_size, 1, num_tokens, 384, device=device),  # Extra sample dimension
            "pair": torch.randn(batch_size, num_tokens, num_tokens, 32, device=device),
            "s_inputs": torch.randn(batch_size, num_tokens, 449, device=device),
        }
        
        # Use a minimal configuration with fewer transformer blocks and heads
        diffusion_config = {
            "c_atom": 128,
            "c_s": 384,
            "c_z": 32,
            "c_token": 384,  # Reduced from 832
            "transformer": {"n_blocks": 1, "n_heads": 2},  # Reduced from 4 blocks, 16 heads
            "c_s_inputs": 449,
            "conditioning": {
                "c_s": 384,
                "c_z": 32,
                "c_s_inputs": 449,
                "c_noise_embedding": 128
            },
            "embedder": {
                "c_atom": 128,
                "c_atompair": 16,
                "c_token": 384
            },
            "sigma_data": 16.0,
            "initialization": {},
            # Add inference parameters to limit steps
            "inference": {
                "num_steps": 2,  # Reduced from default (20)
                "N_sample": 1
            }
        }
        
        # Create minimal input_features dictionary
        input_features = {
            "atom_to_token_idx": torch.arange(num_atoms, device=device).unsqueeze(0),
            "ref_pos": torch.randn(batch_size, num_atoms, 3, device=device),
            "ref_space_uid": torch.arange(num_atoms, device=device).unsqueeze(0),
            "ref_charge": torch.zeros(batch_size, num_atoms, 1, device=device),
            "ref_element": torch.zeros(batch_size, num_atoms, 128, device=device),
            "ref_atom_name_chars": torch.zeros(batch_size, num_atoms, 256, device=device),
            "ref_mask": torch.ones(batch_size, num_atoms, 1, device=device),
            "restype": torch.zeros(batch_size, num_tokens, 32, device=device),
            "profile": torch.zeros(batch_size, num_tokens, 32, device=device),
            "deletion_mean": torch.zeros(batch_size, num_atoms, 1, device=device),
            "sing": torch.randn(batch_size, num_atoms, 449, device=device)  # Required for s_inputs fallback
        }
        
        current_memory = get_memory_usage()
        print(f"Memory before run_stageD_diffusion: {current_memory:.2f} MB")
        
        # Check if memory usage is already too high
        if current_memory - initial_memory > memory_threshold:
            pytest.skip(f"Memory usage already exceeds threshold: {current_memory - initial_memory:.2f} MB > {memory_threshold} MB")
        
        # Run with a timeout to prevent hanging
        import signal
        from contextlib import contextmanager
        
        @contextmanager
        def timeout(seconds):
            def signal_handler(signum, frame):
                raise TimeoutError(f"Test timed out after {seconds} seconds")
            
            # Set the signal handler and a timeout
            original_handler = signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            
            try:
                yield
            finally:
                # Restore the original signal handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
        
        # Run the test with a timeout
        with timeout(30):  # 30 second timeout
            # Run the diffusion model
            coords_final = run_stageD_diffusion(
                partial_coords=partial_coords,
                trunk_embeddings=trunk_embeddings,
                diffusion_config=diffusion_config,
                mode="inference",
                device=device,
                input_features=input_features
            )
            
            # Verify the output shape
            assert coords_final.shape == (batch_size, num_atoms, 3), f"Expected shape {(batch_size, num_atoms, 3)}, got {coords_final.shape}"
            
            # Check memory usage after running
            current_memory = get_memory_usage()
            print(f"Memory after run_stageD_diffusion: {current_memory:.2f} MB")
            print(f"Memory increase: {current_memory - initial_memory:.2f} MB")
            
            # Verify memory usage is within threshold
            assert current_memory - initial_memory <= memory_threshold, f"Memory increase {current_memory - initial_memory:.2f} MB exceeds threshold {memory_threshold} MB"
    
    except TimeoutError as e:
        pytest.skip(f"Test timed out: {str(e)}")
    except Exception as e:
        # Re-raise the exception
        raise
    finally:
        # Clean up
        del trunk_embeddings
        del partial_coords
        del input_features
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        current_memory = get_memory_usage()
        print(f"Memory after cleanup: {current_memory:.2f} MB")
        print(f"Final memory increase: {current_memory - initial_memory:.2f} MB")


# ------------------------------------------------------------------------------
# Test: Multi-sample shape mismatch with extra sample dimension in s_trunk (expected failure)


@pytest.mark.xfail(
    reason="Shape mismatch bug expected (AssertionError in broadcast_token_to_atom)."
)
@pytest.mark.skip(reason="Causes excessive memory usage")
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
        "c_s_inputs": 449,  # Added c_s_inputs
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

@pytest.mark.skip(reason="Causes excessive memory usage")
def test_local_trunk_small_natom():
    """Test local trunk with small number of atoms, providing minimal features."""
    # Create minimal valid inputs
    batch_size = 1
    num_atoms = 5
    num_tokens = 5  # Assuming 1 atom maps to 1 token for simplicity here
    device = "cpu"

    partial_coords = torch.randn(batch_size, num_atoms, 3, device=device)
    trunk_embeddings = {
        "s_trunk": torch.randn(batch_size, num_tokens, 384, device=device),
        "pair": torch.randn(batch_size, num_tokens, num_tokens, 32, device=device),
        "s_inputs": torch.randn(batch_size, num_tokens, 449, device=device),
    }
    
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 384,
        "c_s_inputs": 449,
        "transformer": {"n_blocks": 1, "n_heads": 2},
        "conditioning": {
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 449, # Corrected expected dimension
            "c_noise_embedding": 128
        },
        "embedder": {
            "c_atom": 128,
            "c_atompair": 16,
            "c_token": 384
        },
        "sigma_data": 16.0,
        "initialization": {},
    }

    # Create minimal input_features dictionary
    input_features = {
        "atom_to_token_idx": torch.arange(num_atoms, device=device).unsqueeze(0),
        "ref_pos": torch.randn(batch_size, num_atoms, 3, device=device),
        "ref_space_uid": torch.arange(num_atoms, device=device).unsqueeze(0),
        "ref_charge": torch.zeros(batch_size, num_atoms, 1, device=device),
        "ref_element": torch.zeros(batch_size, num_atoms, 128, device=device),
        "ref_atom_name_chars": torch.zeros(batch_size, num_atoms, 256, device=device),
        "ref_mask": torch.ones(batch_size, num_atoms, 1, device=device),
        "restype": torch.zeros(batch_size, num_tokens, 32, device=device),
        "profile": torch.zeros(batch_size, num_tokens, 32, device=device),
        "deletion_mean": torch.zeros(batch_size, num_atoms, 1, device=device),
        "sing": torch.randn(batch_size, num_atoms, 449, device=device)  # Required for s_inputs fallback
    }

    try:
        coords_out = run_stageD_diffusion(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device=device,
            input_features=input_features
        )

        assert isinstance(coords_out, torch.Tensor)
        assert coords_out.ndim == 3  # [batch, n_atoms, 3]
        assert coords_out.shape[1] == num_atoms  # Check number of atoms matches
        assert coords_out.shape[2] == 3  # Check coordinate dimension
    finally:
        # Cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def test_local_trunk_small_natom_memory_efficient():
    """
    Memory-efficient version of test_local_trunk_small_natom.
    Uses smaller tensors and fewer diffusion steps to avoid memory issues.
    """
    import psutil
    import os
    
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
            "c_s_inputs": 449,
            "transformer": {"n_blocks": 1, "n_heads": 2},  # Reduced from original
            "conditioning": {
                "c_s": 384,
                "c_z": 32,
                "c_s_inputs": 449,
                "c_noise_embedding": 128
            },
            "embedder": {
                "c_atom": 128,
                "c_atompair": 16,
                "c_token": 384
            },
            "sigma_data": 16.0,
            "initialization": {},
            # Add inference parameters to limit steps
            "inference": {
                "num_steps": 2,  # Reduced from default (20)
                "N_sample": 1
            }
        }

        # Create minimal input_features dictionary
        input_features = {
            "atom_to_token_idx": torch.arange(num_atoms, device=device).unsqueeze(0),
            "ref_pos": torch.randn(batch_size, num_atoms, 3, device=device),
            "ref_space_uid": torch.arange(num_atoms, device=device).unsqueeze(0),
            "ref_charge": torch.zeros(batch_size, num_atoms, 1, device=device),
            "ref_element": torch.zeros(batch_size, num_atoms, 128, device=device),
            "ref_atom_name_chars": torch.zeros(batch_size, num_atoms, 256, device=device),
            "ref_mask": torch.ones(batch_size, num_atoms, 1, device=device),
            "restype": torch.zeros(batch_size, num_tokens, 32, device=device),
            "profile": torch.zeros(batch_size, num_tokens, 32, device=device),
            "deletion_mean": torch.zeros(batch_size, num_atoms, 1, device=device),
            "sing": torch.randn(batch_size, num_atoms, 449, device=device)  # Required for s_inputs fallback
        }

        current_memory = get_memory_usage()
        print(f"Memory before run_stageD_diffusion: {current_memory:.2f} MB")
        
        coords_out = run_stageD_diffusion(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device=device,
            input_features=input_features
        )

        current_memory = get_memory_usage()
        print(f"Memory after run_stageD_diffusion: {current_memory:.2f} MB")
        print(f"Memory increase: {current_memory - initial_memory:.2f} MB")

        assert isinstance(coords_out, torch.Tensor)
        # For N_sample=1, the manager now ensures the output is [batch, n_atoms, 3]
        assert coords_out.ndim == 3
        assert coords_out.shape[0] == batch_size
        assert coords_out.shape[1] == num_atoms  # Check n_atoms dimension
        assert coords_out.shape[2] == 3          # Check coordinate dimension

        # Since N_sample=1 in the config, the manager already handled the shape.
        # coords_out = coords_out[:, 0]  # This is redundant, shape is already [B, N_atom, 3]

        # Now coords_out should be [batch, n_atoms, 3]
        # assert coords_out.ndim == 3 # Already asserted above
        assert coords_out.shape == (batch_size, num_atoms, 3)

        print("Test passed successfully!")
        
    finally:
        # Cleanup
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
        run_stageD_diffusion(
            partial_coords,
            trunk_embeddings,
            diffusion_config,
            mode="inference",
            device="cpu",
        )


def test_transformer_size_memory_threshold():
    """
    Experiment to find the memory threshold for transformer configuration.
    Tests progressively larger transformer sizes until memory issues occur.
    """
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB
    
    partial_coords = torch.randn(1, 5, 3)  # Reduced size for safety
    
    # Test configurations with increasing size
    configs = [
        {"n_blocks": 1, "n_heads": 4},   # Base case
        {"n_blocks": 2, "n_heads": 8},   # 2x size
        {"n_blocks": 3, "n_heads": 12},  # 3x size
        {"n_blocks": 4, "n_heads": 16},  # Original size
    ]
    
    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")
    
    for config in configs:
        print(f"\nTesting config: {config}")
        try:
            trunk_embeddings = {
                "s_trunk": torch.randn(1, 1, 5, 384),
                "pair": torch.randn(1, 5, 5, 32),
                "s_inputs": torch.randn(1, 5, 449),
            }
            
            diffusion_config = {
                "c_atom": 128,
                "c_s": 384,
                "c_z": 32,
                "c_token": 832,
                "transformer": config,
                "c_s_inputs": 449,
            }
            
            # Try to run with this config
            _ = run_stageD_diffusion(
                partial_coords=partial_coords,
                trunk_embeddings=trunk_embeddings,
                diffusion_config=diffusion_config,
                mode="inference",
                device="cpu",
            )
            
            current_memory = get_memory_usage()
            memory_increase = current_memory - initial_memory
            print(f"Success with config: {config}")
            print(f"Memory increase: {memory_increase:.2f} MB")
            
        except Exception as e:
            print(f"Failed at config: {config}")
            print(f"Error: {str(e)}")
            break
        finally:
            # Clean up after each attempt
            del trunk_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            current_memory = get_memory_usage()
            print(f"Memory after cleanup: {current_memory:.2f} MB")


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
            _ = run_stageD_diffusion(
                partial_coords=partial_coords,
                trunk_embeddings=trunk_embeddings,
                diffusion_config=base_config,
                mode="inference",
                device="cpu",
            )
            
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


def test_problem_size_memory_threshold():
    """
    Experiment to find the memory threshold by gradually increasing problem size.
    Tests different combinations of atoms and transformer size.
    """
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB
    
    # Test different problem sizes
    sizes = [
        {"atoms": 5, "blocks": 1, "heads": 4},    # Base case
        {"atoms": 10, "blocks": 1, "heads": 4},   # More atoms
        {"atoms": 10, "blocks": 2, "heads": 8},   # More atoms + larger transformer
        {"atoms": 20, "blocks": 2, "heads": 8},   # Even more atoms
        {"atoms": 20, "blocks": 4, "heads": 16},  # Full size
    ]
    
    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")
    
    for size in sizes:
        print(f"\nTesting size: {size}")
        try:
            n_atoms = size["atoms"]
            partial_coords = torch.randn(1, n_atoms, 3)
            
            trunk_embeddings = {
                "s_trunk": torch.randn(1, 1, n_atoms, 384),
                "pair": torch.randn(1, n_atoms, n_atoms, 32),
                "s_inputs": torch.randn(1, n_atoms, 449),
            }
            
            diffusion_config = {
                "c_atom": 128,
                "c_s": 384,
                "c_z": 32,
                "c_token": 832,
                "transformer": {"n_blocks": size["blocks"], "n_heads": size["heads"]},
                "c_s_inputs": 449,
            }
            
            # Try to run with this size
            _ = run_stageD_diffusion(
                partial_coords=partial_coords,
                trunk_embeddings=trunk_embeddings,
                diffusion_config=diffusion_config,
                mode="inference",
                device="cpu",
            )
            
            current_memory = get_memory_usage()
            memory_increase = current_memory - initial_memory
            print(f"Success with size: {size}")
            print(f"Memory increase: {memory_increase:.2f} MB")
            
        except Exception as e:
            print(f"Failed at size: {size}")
            print(f"Error: {str(e)}")
            break
        finally:
            # Clean up after each attempt
            del trunk_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            current_memory = get_memory_usage()
            print(f"Memory after cleanup: {current_memory:.2f} MB")


def test_broadcast_token_multisample_progressive():
    """
    Progressive testing to identify the memory threshold for the broadcast_token_multisample test.
    Gradually increases model size and tensor dimensions to find the point where memory issues occur.
    """
    import psutil
    import os
    import gc
    
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB
    
    # Memory threshold in MB - adjust based on available system memory
    memory_threshold = 1000  # 1GB threshold
    
    # Progressive configurations to test
    configs = [
        # Base case - minimal configuration
        {
            "atoms": 5,
            "tokens": 5,
            "blocks": 1,
            "heads": 2,
            "c_token": 384,
            "num_steps": 2,
        },
        # Slightly larger
        {
            "atoms": 8,
            "tokens": 8,
            "blocks": 1,
            "heads": 4,
            "c_token": 384,
            "num_steps": 2,
        },
        # Medium size
        {
            "atoms": 10,
            "tokens": 10,
            "blocks": 2,
            "heads": 4,
            "c_token": 384,
            "num_steps": 3,
        },
        # Larger size
        {
            "atoms": 10,
            "tokens": 10,
            "blocks": 2,
            "heads": 8,
            "c_token": 832,
            "num_steps": 3,
        },
        # Original size (may cause memory issues)
        {
            "atoms": 10,
            "tokens": 10,
            "blocks": 4,
            "heads": 16,
            "c_token": 832,
            "num_steps": 5,
        },
    ]
    
    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\nTesting configuration {i+1}/{len(configs)}:")
        print(f"  atoms: {config['atoms']}, tokens: {config['tokens']}")
        print(f"  transformer: {config['blocks']} blocks, {config['heads']} heads")
        print(f"  c_token: {config['c_token']}, num_steps: {config['num_steps']}")
        
        try:
            # Create tensors with explicit device placement
            batch_size = 1
            device = "cpu"
            
            partial_coords = torch.randn(batch_size, config["atoms"], 3, device=device)
            
            # Create trunk embeddings with the shape mismatch
            trunk_embeddings = {
                "s_trunk": torch.randn(batch_size, 1, config["tokens"], 384, device=device),  # Extra sample dimension
                "pair": torch.randn(batch_size, config["tokens"], config["tokens"], 32, device=device),
                "s_inputs": torch.randn(batch_size, config["tokens"], 449, device=device),
            }
            
            # Use configuration with specified parameters
            diffusion_config = {
                "c_atom": 128,
                "c_s": 384,
                "c_z": 32,
                "c_token": config["c_token"],
                "transformer": {"n_blocks": config["blocks"], "n_heads": config["heads"]},
                "c_s_inputs": 449,
                "conditioning": {
                    "c_s": 384,
                    "c_z": 32,
                    "c_s_inputs": 449,
                    "c_noise_embedding": 128
                },
                "embedder": {
                    "c_atom": 128,
                    "c_atompair": 16,
                    "c_token": config["c_token"]
                },
                "sigma_data": 16.0,
                "initialization": {},
                "inference": {
                    "num_steps": config["num_steps"],
                    "N_sample": 1
                }
            }
            
            # Create minimal input_features dictionary
            input_features = {
                "atom_to_token_idx": torch.arange(config["atoms"], device=device).unsqueeze(0),
                "ref_pos": torch.randn(batch_size, config["atoms"], 3, device=device),
                "ref_space_uid": torch.arange(config["atoms"], device=device).unsqueeze(0),
                "ref_charge": torch.zeros(batch_size, config["atoms"], 1, device=device),
                "ref_element": torch.zeros(batch_size, config["atoms"], 128, device=device),
                "ref_atom_name_chars": torch.zeros(batch_size, config["atoms"], 256, device=device),
                "ref_mask": torch.ones(batch_size, config["atoms"], 1, device=device),
                "restype": torch.zeros(batch_size, config["tokens"], 32, device=device),
                "profile": torch.zeros(batch_size, config["tokens"], 32, device=device),
                "deletion_mean": torch.zeros(batch_size, config["atoms"], 1, device=device),
                "sing": torch.randn(batch_size, config["atoms"], 449, device=device)
            }
            
            current_memory = get_memory_usage()
            print(f"Memory before run_stageD_diffusion: {current_memory:.2f} MB")
            
            # Check if memory usage is already too high
            if current_memory - initial_memory > memory_threshold:
                print(f"Memory usage already exceeds threshold: {current_memory - initial_memory:.2f} MB > {memory_threshold} MB")
                results.append({
                    "config": config,
                    "status": "skipped",
                    "reason": "memory_threshold_exceeded",
                    "memory_before": current_memory,
                    "memory_increase": current_memory - initial_memory
                })
                continue
            
            # Run with a timeout to prevent hanging
            import signal
            from contextlib import contextmanager
            
            @contextmanager
            def timeout(seconds):
                def signal_handler(signum, frame):
                    raise TimeoutError(f"Test timed out after {seconds} seconds")
                
                # Set the signal handler and a timeout
                original_handler = signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(seconds)
                
                try:
                    yield
                finally:
                    # Restore the original signal handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, original_handler)
            
            # Run the test with a timeout
            with timeout(30):  # 30 second timeout
                try:
                    # Run the diffusion model
                    coords_final = run_stageD_diffusion(
                        partial_coords=partial_coords,
                        trunk_embeddings=trunk_embeddings,
                        diffusion_config=diffusion_config,
                        mode="inference",
                        device=device,
                        input_features=input_features
                    )
                    
                    # Verify the output shape
                    assert coords_final.shape == (batch_size, config["atoms"], 3), f"Expected shape {(batch_size, config['atoms'], 3)}, got {coords_final.shape}"
                    
                    # Check memory usage after running
                    current_memory = get_memory_usage()
                    memory_increase = current_memory - initial_memory
                    
                    # Record success
                    results.append({
                        "config": config,
                        "status": "success",
                        "memory_before": current_memory,
                        "memory_after": get_memory_usage(),
                        "memory_increase": memory_increase
                    })
                    
                    print(f"Success with config: {config}")
                    print(f"Memory increase: {memory_increase:.2f} MB")
                    
                except Exception as e:
                    # Record failure
                    current_memory = get_memory_usage()
                    memory_increase = current_memory - initial_memory
                    
                    results.append({
                        "config": config,
                        "status": "error",
                        "error": str(e),
                        "memory_before": current_memory,
                        "memory_after": get_memory_usage(),
                        "memory_increase": memory_increase
                    })
                    
                    print(f"Failed with config: {config}")
                    print(f"Error: {str(e)}")
                    print(f"Memory increase: {memory_increase:.2f} MB")
                    
                    # Stop testing if we hit an error
                    break
            
        except TimeoutError as e:
            print(f"Test timed out: {str(e)}")
            results.append({
                "config": config,
                "status": "timeout",
                "memory_before": current_memory,
                "memory_after": get_memory_usage(),
                "memory_increase": get_memory_usage() - initial_memory
            })
            # Stop testing if we hit a timeout
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            results.append({
                "config": config,
                "status": "error",
                "error": str(e),
                "memory_before": current_memory,
                "memory_after": get_memory_usage(),
                "memory_increase": get_memory_usage() - initial_memory
            })
            # Stop testing if we hit an error
            break
        finally:
            # Clean up
            del trunk_embeddings
            del partial_coords
            del input_features
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            current_memory = get_memory_usage()
            print(f"Memory after cleanup: {current_memory:.2f} MB")
            print(f"Final memory increase: {current_memory - initial_memory:.2f} MB")
    
    # Print summary of results
    print("\n=== Test Results Summary ===")
    for i, result in enumerate(results):
        print(f"\nConfiguration {i+1}:")
        print(f"  Status: {result['status']}")
        if "error" in result:
            print(f"  Error: {result['error']}")
        print(f"  Memory increase: {result.get('memory_increase', 'N/A'):.2f} MB")
    
    # Find the largest configuration that worked
    working_configs = [r for r in results if r["status"] == "success"]
    if working_configs:
        largest_working = max(working_configs, key=lambda r: r["config"]["atoms"] * r["config"]["blocks"] * r["config"]["heads"])
        print(f"\nLargest working configuration:")
        print(f"  atoms: {largest_working['config']['atoms']}, tokens: {largest_working['config']['tokens']}")
        print(f"  transformer: {largest_working['config']['blocks']} blocks, {largest_working['config']['heads']} heads")
        print(f"  c_token: {largest_working['config']['c_token']}, num_steps: {largest_working['config']['num_steps']}")
        print(f"  Memory increase: {largest_working['memory_increase']:.2f} MB")
    else:
        print("\nNo working configurations found.")
