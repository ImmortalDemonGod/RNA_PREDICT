import torch

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.pipeline.stageC.stage_c_reconstruction import StageCReconstruction


def run_pipeline(sequence: str):
    # Stage A: Obtain sequence and dummy adjacency matrix
    stageA = StageARFoldPredictor(config={})
    adjacency = stageA.predict_adjacency(sequence)
    adjacency = torch.from_numpy(adjacency).float()
    seq = sequence
    print(f"[Stage A] sequence = {seq}, adjacency shape = {adjacency.shape}")

    # Stage B: Predict torsion angles; choose desired output mode ("sin_cos", "radians", "degrees")
    stageB = StageBTorsionBertPredictor(
        model_name_or_path="sayby/rna_torsionbert",
        device="cpu",
        angle_mode="degrees",
        num_angles=16,
        max_length=512,
    )
    outB = stageB(seq, adjacency)
    torsion_angles = outB["torsion_angles"]
    print(f"[Stage B] angles shape = {torsion_angles.shape}")

    # Stage C: Generate dummy 3D coordinates from angles
    stageC = StageCReconstruction()
    outC = stageC(torsion_angles)
    coords = outC["coords"]
    print(f"[Stage C] coords shape = {coords.shape}, #atoms = {outC['atom_count']}")


def main():
    sample_seq = "ACGUAACGU"
    run_pipeline(sample_seq)


if __name__ == "__main__":
    main()
