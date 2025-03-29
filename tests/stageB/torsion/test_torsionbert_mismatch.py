import pytest
import torch

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor

@pytest.mark.parametrize("angle_mode", ["sin_cos", "degrees"])
def test_torsionbert_shape_mismatch(angle_mode):
    """
    This test deliberately sets num_angles=16 while the loaded model
    outputs dimension 14 (2*7). Without the patch or angle fix,
    it should fail with a RuntimeError about dimension mismatch.
    After the patch, the code is robust enough to handle the actual dimension
    or forcibly detect it from the model, thus no error.
    """
    predictor = StageBTorsionBertPredictor(
        model_name_or_path="sayby/rna_torsionbert",
        device="cpu",
        angle_mode=angle_mode,
        num_angles=16,  # triggers mismatch if unpatched
        max_length=256,
    )

    sample_seq = "ACGUACGU"  # length=8
    try:
        out = predictor(sample_seq)
        # Should no longer raise an error if the fix is in place
        torsion_angles = out["torsion_angles"]
        assert torsion_angles.shape[0] == 8, "Should match 8 residues"
        # shape[-1] might be 14 if sin_cos => (N, 14) or 7 if degrees => (N, 7)
    except RuntimeError as e:
        pytest.fail(f"Unpatched dimension mismatch triggered: {e}")