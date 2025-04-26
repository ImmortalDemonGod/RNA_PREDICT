"""High-level interface for the RNA_PREDICT pipeline."""

from __future__ import annotations

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict, Any

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC
from rna_predict.utils.submission import coords_to_df, extract_atom, reshape_coords
from rna_predict.conf.config_schema import register_configs

# Register all configurations with Hydra
register_configs()

class RNAPredictor:
    """High-level interface for the RNA_PREDICT pipeline.

    This class encapsulates the process of converting an RNA sequence to its 3D structure.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the RNA predictor using a Hydra configuration object.

        Args:
            cfg: Hydra configuration object following RNAConfig schema.
        """
        # Determine device from global config or default to CUDA if available
        self.device = getattr(cfg, "device", None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)

        # Get Stage C configuration from model.stageC
        if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageC"):
            raise ValueError("Configuration must contain model.stageC section")
        self.stageC_config = cfg.model.stageC

        # Store prediction defaults from config
        self.prediction_config = getattr(cfg, "prediction", {})
        self.default_repeats = getattr(self.prediction_config, "repeats", 5)
        self.default_atom_choice = getattr(self.prediction_config, "residue_atom_choice", 1)

        # Initialize Stage B Torsion Predictor
        # Pass the correct subconfig to StageBTorsionBertPredictor
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageB') and hasattr(cfg.model.stageB, 'torsion_bert'):
            torsion_bert_cfg = cfg.model.stageB.torsion_bert
            # Ensure required fields are present
            if not hasattr(torsion_bert_cfg, 'model_name_or_path'):
                torsion_bert_cfg.model_name_or_path = "dummy-path"
            if not hasattr(torsion_bert_cfg, 'device'):
                torsion_bert_cfg.device = self.device
        else:
            # Use the provided config directly
            torsion_bert_cfg = cfg
            # Ensure required fields are present
            if not hasattr(torsion_bert_cfg, 'model_name_or_path'):
                torsion_bert_cfg.model_name_or_path = "dummy-path"
            if not hasattr(torsion_bert_cfg, 'device'):
                torsion_bert_cfg.device = self.device

        self.torsion_predictor = StageBTorsionBertPredictor(torsion_bert_cfg)

    def predict_3d_structure(self, sequence: str) -> Dict[str, Any]:
        """Run Stage B predictor -> Stage C reconstruction pipeline on a single RNA sequence.

        Args:
            sequence: RNA sequence string

        Returns:
            Dictionary containing:
                - coords: Tensor of shape [N, atoms, 3] with atomic coordinates
                - atom_count: Total number of atoms
        """
        if not sequence:
            return {
                "coords": torch.empty((0, 3), device=self.device),
                "coords_3d": torch.empty((0, 0, 3), device=self.device),
                "atom_count": 0,
            }

        # Stage B: Torsion angles
        torsion_output = self.torsion_predictor(sequence)
        torsion_angles = torsion_output["torsion_angles"]

        # Stage C: 3D coords - Use configuration from Hydra
        # Create a new config to avoid modifying the original and ensure required parameters
        base_stagec = OmegaConf.to_container(self.stageC_config, resolve=True)
        # Handle different types that might be returned by OmegaConf.to_container
        if isinstance(base_stagec, dict):
            if "debug_logging" not in base_stagec:
                base_stagec["debug_logging"] = False
            base_stagec["device"] = str(self.device)
        else:
            # If not a dict, create a new dict with the required fields
            base_stagec = {
                "debug_logging": False,
                "device": str(self.device)
            }
        stageC_config = OmegaConf.create({"model": {"stageC": base_stagec}})

        return run_stageC(cfg=stageC_config, sequence=sequence, torsion_angles=torsion_angles)

    def predict_submission(
        self,
        sequence: str,
        prediction_repeats: Optional[int] = None,
        residue_atom_choice: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a submission-style DataFrame for a single RNA sequence.

        Args:
            sequence: RNA sequence string
            prediction_repeats: Optional override for number of prediction repeats
            residue_atom_choice: Optional override for atom choice index

        Returns:
            DataFrame with columns:
                - ID: 1-based residue index
                - resname: nucleotide character
                - resid: 1-based residue index
                - x_1..x_n, y_1..y_n, z_1..z_n: Repeated coordinates

        Raises:
            ValueError: If coordinate shapes are incompatible
            IndexError: If residue_atom_choice is invalid
        """
        if not sequence:
            repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats
            return coords_to_df("", torch.empty(0, 3, device=self.device), repeats)

        # Get coordinates and reshape to standard format
        result = self.predict_3d_structure(sequence)
        coords = result["coords"]
        # If variable atom counts, coords is [total_atoms, 3], else [N, atoms, 3]
        if coords.dim() == 2 and coords.shape[0] != len(sequence):
            # Flat array, cannot extract a single atom per residue by index
            # Instead, just return the flat coords to the DataFrame, one row per atom
            # Map atom indices to residues using residue_atom_map if available
            repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats

            # Create a DataFrame with the expected columns for submission format
            base_data = {
                "ID": range(1, coords.shape[0] + 1),
                "resname": ["X"] * coords.shape[0],  # Placeholder
                "resid": range(1, coords.shape[0] + 1)
            }

            # Add coordinate columns
            # Detach the tensor before converting to numpy to avoid gradient issues
            coords_np = coords.detach().cpu().numpy()
            for i in range(1, repeats + 1):
                base_data[f"x_{i}"] = coords_np[:, 0]
                base_data[f"y_{i}"] = coords_np[:, 1]
                base_data[f"z_{i}"] = coords_np[:, 2]

            df = pd.DataFrame(base_data)
            return df
        coords = reshape_coords(coords, len(sequence))

        # Use provided values or defaults from config
        repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats
        atom_choice = residue_atom_choice if residue_atom_choice is not None else self.default_atom_choice

        # Explicit bounds check for atom_choice
        if coords.dim() != 3 or atom_choice < 0 or atom_choice >= coords.shape[1]:
            raise IndexError(f"Invalid residue_atom_choice {atom_choice} for coords shape {coords.shape} (expected shape [N, atoms, 3])")

        # Extract specific atom coordinates
        atom_coords = extract_atom(coords, atom_choice)

        # Convert to submission DataFrame
        return coords_to_df(sequence, atom_coords, repeats)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main function to demonstrate RNAPredictor using Hydra configuration."""
    print("Configuration loaded by Hydra:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 30)

    # Get sequence from config, with fallbacks
    sequence = None

    # First try direct sequence attribute
    if hasattr(cfg, "sequence"):
        sequence = cfg.sequence
    # Then try test_data.sequence
    elif hasattr(cfg, "test_data"):
        sequence = cfg.test_data.sequence

    if sequence is None:
        # Use default sequence if none found
        sequence = "ACGUACGU"
        print("Warning: No sequence found in config, using default sequence:", sequence)

    # Add sequence to top level config for consistency
    cfg.sequence = sequence

    # Instantiate the predictor with the loaded configuration
    try:
        predictor = RNAPredictor(cfg)
    except Exception as e:
        print(f"Error initializing RNAPredictor: {e}")
        # Print relevant config sections for debugging
        if hasattr(cfg, "stageB_torsion"):
             print("Relevant config (stageB_torsion):")
             print(OmegaConf.to_yaml(cfg.stageB_torsion))
        if hasattr(cfg, "stageB_pairformer"):
             print("Relevant config (stageB_pairformer):")
             print(OmegaConf.to_yaml(cfg.stageB_pairformer))
        raise

    # Use test sequence from config
    test_sequence = cfg.sequence
    print(f"Running prediction for sequence: {test_sequence}")

    try:
        submission_df = predictor.predict_submission(test_sequence)
        print("\nSubmission DataFrame Head:")
        print(submission_df.head())
        output_path = "submission.csv"
        submission_df.to_csv(output_path, index=False)
        print(f"\nSubmission saved to {output_path}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()
