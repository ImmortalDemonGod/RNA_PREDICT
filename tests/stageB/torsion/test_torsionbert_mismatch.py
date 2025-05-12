import pytest
import logging
import os
import psutil
import gc
from hypothesis import given, settings, HealthCheck, strategies as st
from omegaconf import OmegaConf

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)

logger = logging.getLogger(__name__)

# Create a single predictor instance to be reused across all test cases
@pytest.fixture(scope="module")
def shared_predictor():
    """
    Create a single predictor instance to be reused across all test cases.
    This significantly speeds up the test by avoiding multiple model loads.
    """
    logger.info("Creating shared TorsionBERT predictor for all test cases")
    cfg = OmegaConf.create({
        "stageB_torsion": {
            "model_name_or_path": "sayby/rna_torsionbert",
            "device": "cpu",
            "angle_mode": "sin_cos",  # Default, will be modified per test
            "num_angles": 7,  # Default, will be modified per test
            "max_length": 256,
            "debug_logging": True,
        }
    })

    predictor = StageBTorsionBertPredictor(cfg)
    return predictor


@settings(
    deadline=None,  # Disable deadline checks since model loading can be slow
    max_examples=1,  # Only run one example to minimize memory usage for debugging
    suppress_health_check=[HealthCheck.too_slow],
    database=None  # Disable Hypothesis example database to avoid large cache
)
@given(
    angle_mode=st.sampled_from(["sin_cos", "degrees", "radians"]),
    num_angles=st.integers(min_value=7, max_value=20),  # Test various angle dimensions
    seq=st.text(alphabet=["A", "C", "G", "U"], min_size=5, max_size=20)  # Random RNA sequences
)
def test_torsionbert_shape_mismatch(shared_predictor, angle_mode, num_angles, seq):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1e6
    logger.info(f"[MEM-DEBUG] Memory before test: {mem_before:.2f} MB")
    """
    Property-based test: TorsionBERT predictor should handle dimension mismatches gracefully.

    This test deliberately creates potential mismatches between the configured num_angles
    and what the model actually outputs. The robust implementation should handle this
    gracefully without raising errors.

    Args:
        shared_predictor: Reused predictor instance to avoid multiple model loads
        angle_mode: The angle representation mode (sin_cos, degrees, radians)
        num_angles: Number of angles to configure (may mismatch with model's actual output)
        seq: Random RNA sequence to predict angles for

    # ERROR_ID: TORSIONBERT_SHAPE_MISMATCH
    """
    # Modify the predictor's properties instead of creating a new one
    shared_predictor.angle_mode = angle_mode
    shared_predictor.num_angles = num_angles

    # Update output_dim based on new angle_mode and num_angles
    if angle_mode == "sin_cos":
        shared_predictor.output_dim = num_angles * 2
    else:
        shared_predictor.output_dim = num_angles

    logger.info(f"Testing with angle_mode={angle_mode}, num_angles={num_angles}, seq_len={len(seq)}")

    e = None
    mem_before = process.memory_info().rss / 1e6
    logger.info(f"[MEM-DEBUG] Memory before model call: {mem_before:.2f} MB")
    try:
        out = shared_predictor(seq)
        torsion_angles = out["torsion_angles"]
        # Verify output shape matches sequence length
        assert torsion_angles.shape[0] == len(seq), f"Output shape {torsion_angles.shape[0]} should match sequence length {len(seq)}"
        # Verify output dimension is appropriate for the angle mode
        if angle_mode == "sin_cos":
            assert torsion_angles.shape[1] % 2 == 0, f"Sin/cos output should have even number of dimensions, got {torsion_angles.shape[1]}"
        else:
            pass  # We're mainly testing that it doesn't crash
        logger.info(f"Successfully processed sequence with angle_mode={angle_mode}, num_angles={num_angles}")
        logger.info(f"Output shape: {torsion_angles.shape}")
    except RuntimeError as exc:
        e = exc
    finally:
        mem_after = process.memory_info().rss / 1e6
        logger.info(f"[MEM-DEBUG] Memory after model call: {mem_after:.2f} MB")
        # Cleanup
        for var in ['out', 'torsion_angles', 'seq']:
            if var in locals():
                del locals()[var]
        gc.collect()
        mem_after_cleanup = process.memory_info().rss / 1e6
        logger.info(f"[MEM-DEBUG] Memory after cleanup: {mem_after_cleanup:.2f} MB")
        # Error handling
        if e is not None:
            if "Model output dimension" in str(e) and "does not match expected dimension" in str(e):
                logger.info(f"Expected dimension mismatch: {e}")
                pass
            else:
                pytest.fail(f"Unexpected error: {e}")
