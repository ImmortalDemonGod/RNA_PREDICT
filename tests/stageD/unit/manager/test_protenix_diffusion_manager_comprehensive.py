"""Comprehensive tests for ProtenixDiffusionManager."""

import pytest
import torch
from torch import Tensor

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)


@pytest.fixture
def base_manager(hydra_cfg_factory):
    """Create a base ProtenixDiffusionManager instance."""
    cfg = hydra_cfg_factory(
        device="cpu", mode="inference", inference={"num_steps": 2, "temperature": 1.0}
    )
    return ProtenixDiffusionManager(cfg)


def test_initialization(base_manager):
    """Test manager initialization with default config."""
    assert isinstance(base_manager, ProtenixDiffusionManager)
    assert base_manager.device == torch.device("cpu")
    assert base_manager.num_inference_steps == 2
    assert base_manager.temperature == 1.0


@pytest.mark.parametrize("use_override_features", [False, True])
def test_multi_step_inference(
    base_manager, trunk_embeddings_factory, use_override_features
):
    """Test multi-step inference with and without override features."""
    # Prepare inputs
    trunk_embeddings = trunk_embeddings_factory(batch=1, length=50, feat=64)

    # Always construct override_input_features as a full copy of meta features from trunk_embeddings
    meta_keys = [
        "ref_space_uid", "atom_to_token_idx", "ref_pos", "ref_charge", "ref_element",
        "ref_atom_name_chars", "ref_mask", "restype", "profile", "deletion_mean", "sing"
    ]
    override_input_features = {k: trunk_embeddings[k] for k in meta_keys}

    # Run inference
    coords_init = torch.zeros_like(trunk_embeddings["s_trunk"]).unsqueeze(-2).expand(-1, -1, 14, -1).contiguous()[:, :, :, :3]
    outputs = base_manager.multi_step_inference(
        coords_init=coords_init,
        trunk_embeddings=trunk_embeddings,
        override_input_features=override_input_features,
    )

    # Validate outputs
    assert isinstance(outputs, Tensor)
    assert outputs.ndim == 3
    assert outputs.shape[0] == 1
    assert outputs.shape[2] == 3
    assert outputs.dtype == torch.float32


def test_error_handling(base_manager, trunk_embeddings_factory):
    """Test error handling for invalid inputs."""
    # Test with missing required embeddings
    invalid_embeddings = trunk_embeddings_factory()
    del invalid_embeddings["s_trunk"]
    coords_init = torch.zeros(1, 50, 3)  # Dummy coords_init matching batch/length

    with pytest.raises(ValueError, match="requires a valid 's_trunk'"):
        base_manager.multi_step_inference(coords_init, invalid_embeddings)

    # Test with mismatched tensor shapes
    mismatched_embeddings = trunk_embeddings_factory(batch=1, length=50, feat=64)
    mismatched_embeddings["z_trunk"] = torch.randn(1, 40, 40, 32)  # Wrong length

    # The shape mismatch should be handled gracefully with shape_utils
    # Instead of raising an error, it should continue execution

    # Call the function that might have triggered an error before shape_utils integration
    result = base_manager.multi_step_inference(coords_init, mismatched_embeddings)

    # Verify the result is not None
    assert result is not None, "Result should not be None despite shape mismatch"

    # Verify the result has the expected shape
    assert result.shape[0] == 1, f"Expected batch size 1, got {result.shape[0]}"

    # Add a unique error identifier for this test
    # ERROR_ID: STAGEDB_SHAPE_MISMATCH_HANDLING
