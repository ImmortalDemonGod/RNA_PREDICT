import pytest
import torch
from rna_predict.pipeline.stageB.torsion_bert_predictor import StageBTorsionBertPredictor

@pytest.fixture
def stageB_predictor():
    return StageBTorsionBertPredictor(
        model_name_or_path="sayby/rna_torsionbert",
        device="cpu",
        angle_mode="sin_cos",
        num_angles=7
    )

def test_short_seq(stageB_predictor):
    sequence = "AC"
    output = stageB_predictor(sequence)
    angles = output["torsion_angles"]
    # With a 2-letter sequence, expect 2 rows; sincos output has 14 columns (if 7 angles)
    assert angles.shape[0] == 2
    assert angles.shape[1] == 14

def test_normal_seq(stageB_predictor):
    sequence = "ACGU"
    output = stageB_predictor(sequence)
    angles = output["torsion_angles"]
    assert angles.shape == (4, 14)
    print("Sin/Cos angles for 4 residues:", angles)
