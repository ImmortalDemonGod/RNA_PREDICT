"""
Tests for the ProtenixIntegration class.

This module provides tests for the ProtenixIntegration class, which integrates
Protenix input embedding components for Stage B/C synergy by building single-token and
pair embeddings from raw features.
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, HealthCheck

from rna_predict.pipeline.stageB.pairwise.protenix_integration import (
    ProtenixIntegration,
)


@pytest.mark.skip(reason="Causes excessive memory usage")
def test_residue_index_squeeze_fix():
    """
    Ensures build_embeddings() works when 'residue_index'
    starts as shape [N_token,1]. Should no longer raise a RuntimeError.
    """
    integrator = ProtenixIntegration(device=torch.device("cpu"))
    N_token = 5
    N_atom = 4 * N_token
    atoms_per_token = 4
    torch.arange(N_atom).unsqueeze(-1)

    # Create minimal input features with 2D shapes [batch, feat_dim]
    input_features = {
        "residue_index": torch.arange(N_token).reshape(1, N_token),  # [1, N_token]
        "ref_pos": torch.randn(N_atom, 3).reshape(1, -1),  # [1, N_atom * 3]
        "ref_charge": torch.randn(N_atom, 1).reshape(1, -1),  # [1, N_atom]
        "ref_element": torch.randn(N_atom, 128).reshape(1, -1),  # [1, N_atom * 128]
        "ref_atom_name_chars": torch.zeros(1, N_atom, 256),  # [1, N_atom, 256]
        "atom_to_token": torch.repeat_interleave(
            torch.arange(N_token), atoms_per_token
        ).reshape(1, -1),  # [1, N_atom]
        "atom_to_token_idx": torch.repeat_interleave(
            torch.arange(N_token), atoms_per_token
        ).reshape(1, -1),  # [1, N_atom]
        "restype": torch.zeros(N_token, 32).reshape(1, -1),  # [1, N_token * 32]
        "profile": torch.zeros(N_token, 32).reshape(1, -1),  # [1, N_token * 32]
        "deletion_mean": torch.zeros(N_token).reshape(1, -1),  # [1, N_token]
        "ref_mask": torch.ones(N_atom, dtype=torch.bool).reshape(1, -1),  # [1, N_atom]
        "ref_space_uid": torch.zeros(N_atom, dtype=torch.long).reshape(
            1, -1
        ),  # [1, N_atom]
    }

    # Special handling for ref_atom_name_chars to ensure it has exactly 256 dimensions
    input_features["ref_atom_name_chars"] = input_features[
        "ref_atom_name_chars"
    ].reshape(1, N_atom, 256)

    # Verify no dimension error
    embeddings = integrator.build_embeddings(input_features)

    assert "s_inputs" in embeddings, "Missing single-token embedding"
    assert "z_init" in embeddings, "Missing pair embedding"

    s_inputs = embeddings["s_inputs"]
    z_init = embeddings["z_init"]

    # Confirm shapes
    assert (
        s_inputs.shape[0] == N_token
    ), f"Expected s_inputs shape (N_token, _), got {s_inputs.shape}"
    assert z_init.dim() == 3, f"Expected z_init dimension=3, got {z_init.dim()}"
    assert (
        z_init.shape[0] == N_token and z_init.shape[1] == N_token
    ), f"Expected z_init shape (N_token, N_token, c_z), got {z_init.shape}"

    print(
        "test_residue_index_squeeze_fix passed: no expand() error with (N_token,1) residue_index!"
    )


@settings(
    deadline=None,  # Disable deadline checks since model loading can be slow
    max_examples=5,  # Limit number of examples to keep test runtime reasonable
    suppress_health_check=[HealthCheck.too_slow]
)
@given(
    n_token=st.integers(min_value=2, max_value=5),  # Small number of tokens for memory efficiency
    atoms_per_token=st.integers(min_value=1, max_value=3),  # Vary atoms per token
    residue_index_shape=st.sampled_from(["1d", "2d"])  # Test both 1D and 2D residue_index shapes
)
def test_residue_index_squeeze_fix_memory_efficient(n_token, atoms_per_token, residue_index_shape):
    """
    Property-based test for ProtenixIntegration.build_embeddings().
    Tests that the function handles different residue_index shapes correctly.

    Args:
        n_token: Number of tokens to use in the test
        atoms_per_token: Number of atoms per token
        residue_index_shape: Shape of the residue_index tensor ("1d" or "2d")
    """
    import gc
    import os
    import psutil
    from contextlib import contextmanager
    from omegaconf import OmegaConf

    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @contextmanager
    def timeout(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Test timed out after {seconds} seconds")

        import signal

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    # Set memory threshold (1GB)
    MEMORY_THRESHOLD = 1000  # MB
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    try:
        # Create a proper configuration object for ProtenixIntegration
        cfg = OmegaConf.create({
            "model": {
                "stageB": {
                    "debug_logging": True,
                    "pairformer": {
                        "protenix_integration": {
                            "device": "cpu",
                            "c_token": 384,
                            "restype_dim": 32,
                            "profile_dim": 32,
                            "c_atom": 128,
                            "c_pair": 32,
                            "r_max": 32,  # Required parameter
                            "s_max": 32,  # Required parameter
                            "use_optimized": False
                        }
                    }
                }
            }
        })

        # Initialize with the configuration object
        integrator = ProtenixIntegration(cfg)

        # Calculate number of atoms
        N_token = n_token
        N_atom = N_token * atoms_per_token

        # Create residue_index tensor with the specified shape
        if residue_index_shape == "1d":
            residue_index = torch.arange(N_token)  # [N_token]
        else:  # "2d"
            residue_index = torch.arange(N_token).unsqueeze(-1)  # [N_token, 1]

        # Create minimal input features
        input_features = {
            "residue_index": residue_index,
            "ref_pos": torch.randn(N_atom, 3),  # [N_atom, 3]
            "ref_charge": torch.randn(N_atom, 1),  # [N_atom, 1]
            "ref_element": torch.randn(N_atom, 128),  # [N_atom, 128]
            "ref_atom_name_chars": torch.zeros(N_atom, 256),
            "atom_to_token": torch.repeat_interleave(
                torch.arange(N_token), atoms_per_token
            ),  # [N_atom]
            "atom_to_token_idx": torch.repeat_interleave(
                torch.arange(N_token), atoms_per_token
            ),  # [N_atom]
            "restype": torch.zeros(N_token, 32),  # [N_token, 32]
            "profile": torch.zeros(N_token, 32),  # [N_token, 32]
            "deletion_mean": torch.zeros(N_token),  # [N_token]
            "ref_mask": torch.ones(N_atom, dtype=torch.bool),  # [N_atom]
            "ref_space_uid": torch.zeros(N_atom, dtype=torch.long),  # [N_atom]
        }

        # Monitor memory before building embeddings
        pre_build_memory = get_memory_usage()
        memory_increase = pre_build_memory - initial_memory
        assert (
            memory_increase < MEMORY_THRESHOLD
        ), f"Memory increase ({memory_increase:.2f} MB) exceeds threshold"

        # Set timeout to prevent hanging
        with timeout(seconds=60):
            # Build embeddings
            embeddings = integrator.build_embeddings(input_features)

            # Basic assertions
            assert "s_inputs" in embeddings, "Missing single-token embedding"
            assert "z_init" in embeddings, "Missing pair embedding"

            s_inputs = embeddings["s_inputs"]
            z_init = embeddings["z_init"]

            # Verify shapes
            assert (
                s_inputs.shape[0] == N_token
            ), f"Expected s_inputs shape (N_token, _), got {s_inputs.shape}"
            assert z_init.dim() == 3, f"Expected z_init dimension=3, got {z_init.dim()}"
            assert (
                z_init.shape[0] == N_token and z_init.shape[1] == N_token
            ), f"Expected z_init shape (N_token, N_token, c_z), got {z_init.shape}"

            # Verify no NaN or Inf values
            assert not torch.isnan(s_inputs).any(), "s_inputs contains NaN values"
            assert not torch.isnan(z_init).any(), "z_init contains NaN values"
            assert not torch.isinf(s_inputs).any(), "s_inputs contains infinity values"
            assert not torch.isinf(z_init).any(), "z_init contains infinity values"

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise

    finally:
        # Cleanup
        if "embeddings" in locals():
            del embeddings
        gc.collect()
        torch.cuda.empty_cache()

        # Report final memory usage
        final_memory = get_memory_usage()
        print(f"Final memory increase: {final_memory - initial_memory:.2f} MB")
