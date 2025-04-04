import pytest

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)


@pytest.fixture
def stageB_predictor():
    """
    Updated fixture to match real pipeline usage:
    We want degrees mode with 16 angles, so final shape is [N, 16].
    """
    return StageBTorsionBertPredictor(
        model_name_or_path="sayby/rna_torsionbert",
        device="cpu",
        angle_mode="degrees",
        num_angles=16,
    )

# @pytest.mark.skip(reason="Execution time is too long") # Temporarily unskipped for investigation
def test_short_seq(stageB_predictor):
    """
    Test short sequence with 2 residues. Expect [2, 16] in degrees mode.
    """
    sequence = "AC"
    output = stageB_predictor(sequence)
    angles = output["torsion_angles"]
    assert angles.shape[0] == 2
    # 16 angles in degrees mode
    assert angles.shape[1] == 16

# @pytest.mark.skip(reason="Execution time is too long") # Temporarily unskipped for investigation
def test_normal_seq(stageB_predictor):
    """
    Test normal 4-letter sequence "ACGU". Expect [4,16] with 16 angles in degrees mode.
    """
    sequence = "ACGU"
    output = stageB_predictor(sequence)
    angles = output["torsion_angles"]
    assert angles.shape == (4, 16)
    print("Degrees angles for 4 residues:", angles)
