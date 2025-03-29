import pandas as pd
import torch

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC


class RNAPredictor:
    """
    High-level interface for the RNA_PREDICT pipeline.

    This class encapsulates the process of converting an RNA sequence to its 3D structure.
    It wraps the torsion angle prediction (Stage B) and 3D reconstruction (Stage C).
    The user can then generate a submission-friendly DataFrame, repeating one
    3D prediction multiple times to satisfy Kaggle's requirement of 5 predicted
    structures per residue.
    """

    def __init__(
        self,
        model_name_or_path="sayby/rna_torsionbert",
        device=None,
        angle_mode="degrees",
        num_angles=7,
        max_length=512,
        stageC_method="mp_nerf",
    ):
        """
        Args:
            model_name_or_path (str): Hugging Face or local path for TorsionBERT model.
            device (str or torch.device): "cpu" or "cuda". If None, auto-detect.
            angle_mode (str): "sin_cos" or "radians" or "degrees".
            num_angles (int): e.g. 7 for alpha..zeta + chi.
            max_length (int): for tokenizer max length.
            stageC_method (str): "mp_nerf" or fallback; used in run_stageC.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Stage B predictor
        self.torsion_predictor = StageBTorsionBertPredictor(
            model_name_or_path=model_name_or_path,
            device=str(self.device),
            angle_mode=angle_mode,
            num_angles=num_angles,
            max_length=max_length,
        )
        self.stageC_method = stageC_method

    def predict_3d_structure(self, sequence: str) -> dict:
        """
        Runs the Stage B predictor -> Stage C reconstruction pipeline on a single RNA sequence.

        Args:
            sequence (str): e.g. "ACGUACGU"

        Returns:
            dict: e.g. {
                "coords": Tensor of shape (N * #atoms_per_res, 3) or [N, #atoms, 3],
                "atom_count": int
            }
        """
        if not sequence:
            # If empty, produce zero-sized coords but correct structure
            return {"coords": torch.empty((0, 3)), "atom_count": 0}

        # Stage B: Torsion angles
        torsion_output = self.torsion_predictor(sequence)
        torsion_angles = torsion_output[
            "torsion_angles"
        ]  # shape [N, ...] either sin/cos or direct angles

        # Stage C: 3D coords
        stageC_result = run_stageC(
            sequence=sequence,
            torsion_angles=torsion_angles,
            method=self.stageC_method,
            device=str(self.device),
            do_ring_closure=False,
            place_bases=True,
            sugar_pucker="C3'-endo",
        )
        return stageC_result

    def predict_submission(
        self, sequence: str, prediction_repeats: int = 5, residue_atom_choice: int = 0
    ) -> pd.DataFrame:
        """
        Generates a submission-style DataFrame for a single RNA sequence.
        We replicate the same predicted 3D coordinate across 'prediction_repeats' structures.

        Columns in the returned DataFrame:
        - ID: residue index (1-based)
        - resname: actual character from 'sequence'
        - resid: numeric residue index (1..N)
        - For each structure i in [1..prediction_repeats], columns: x_i, y_i, z_i

        If the Stage C coords have shape [N, #atoms, 3], we pick the 'residue_atom_choice'
        (like 0 for the first atom or specifically the "C1'" index) per residue.

        If sequence is empty, return an empty DataFrame but with the correct columns.
        If there's a shape mismatch when reshaping coords, raise ValueError.
        If residue_atom_choice is invalid, raise IndexError.
        """
        # Prepare columns
        cols = ["ID", "resname", "resid"]
        for i in range(1, prediction_repeats + 1):
            cols += [f"x_{i}", f"y_{i}", f"z_{i}"]

        # Handle empty sequence => zero rows but correct columns
        if not sequence:
            return pd.DataFrame(columns=cols)

        # 1) Run the pipeline to get coords
        result_dict = self.predict_3d_structure(sequence)
        coords = result_dict[
            "coords"
        ]  # shape could be [N, #atoms, 3], [N, 3], or [N*#atoms, 3]

        N = len(sequence)
        # 2) Ensure shape is [N, #atoms, 3]
        if coords.dim() == 2 and coords.shape[0] == N:
            # shape is [N, 3] -> single atom per residue
            coords_per_res = coords.unsqueeze(1)
        elif coords.dim() == 2 and coords.shape[0] == N * 3:
            # Possibly the "legacy fallback" with 3 atoms per residue
            coords_per_res = coords.view(N, 3, 3)
        elif coords.dim() == 2:
            # E.g. coords.shape[0] = N * someNumber
            atoms_per_res = coords.shape[0] // N
            try:
                coords_per_res = coords.view(N, atoms_per_res, 3)
            except RuntimeError as e:
                raise ValueError(
                    f"Shape mismatch: cannot reshape coords {coords.shape} into "
                    f"[N={N}, {atoms_per_res}, 3]."
                ) from e
        elif coords.dim() == 3:
            coords_per_res = coords
        else:
            raise ValueError(f"Unexpected coords shape: {coords.shape}")

        # 3) We'll pick the coordinate of interest: residue_atom_choice
        try:
            final_coords = coords_per_res[:, residue_atom_choice, :]
        except IndexError:
            raise IndexError(
                f"Invalid residue_atom_choice {residue_atom_choice} "
                f"for coords shape {coords_per_res.shape}."
            )

        # 4) Build the DataFrame with repeated coordinates
        rows = []
        for i, nt in enumerate(sequence):
            row = {"ID": i + 1, "resname": nt, "resid": i + 1}
            x_val = float(final_coords[i, 0])
            y_val = float(final_coords[i, 1])
            z_val = float(final_coords[i, 2])

            # replicate the single predicted coordinate across multiple predictions
            for rep_i in range(1, prediction_repeats + 1):
                row[f"x_{rep_i}"] = x_val
                row[f"y_{rep_i}"] = y_val
                row[f"z_{rep_i}"] = z_val

            rows.append(row)

        submission_df = pd.DataFrame(rows, columns=cols)
        return submission_df


if __name__ == "__main__":
    from rna_predict.interface import RNAPredictor

    predictor = RNAPredictor(
        model_name_or_path="sayby/rna_torsionbert",
        device=None,  # Will auto-detect GPU if present
        angle_mode="degrees",
        num_angles=7,
        max_length=512,
        stageC_method="mp_nerf",
    )

    # Example usage for a short test sequence:
    submission_df = predictor.predict_submission("ACGUACGU", prediction_repeats=5)
    print(submission_df.head())
    submission_df.to_csv("submission.csv", index=False)
