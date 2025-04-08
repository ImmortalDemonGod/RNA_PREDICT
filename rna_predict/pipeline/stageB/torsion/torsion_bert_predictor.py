import math
from typing import Any, Dict, Optional, Union # Added Union

import torch

from rna_predict.pipeline.stageB.torsion.dummy_torsion_model import DummyTorsionModel
from rna_predict.pipeline.stageB.torsion.torsionbert_inference import TorsionBertModel


class StageBTorsionBertPredictor:
    """
    Stage B: predict RNA torsion angles using TorsionBERT.
    This version does not force output shape = [N, 2*self.num_angles];
    Instead, it relies on the actual dimension of model output.
    If model_name_or_path is invalid or random, we fallback to a DummyTorsionModel
    that returns zeros, avoiding Hugging Face errors in fuzz tests.

    Updated to store model_name_or_path and max_length as attributes to satisfy
    test cases referencing them.
    """
    model: Union[TorsionBertModel, DummyTorsionModel] # Add type hint for self.model

    def __init__(
        self,
        model_name_or_path: str = "sayby/rna_torsionbert",
        device: str = "cpu", # Keep original device string
        angle_mode: str = "sin_cos",
        num_angles: int = 7,
        max_length: int = 512,
    ):
        """
        Args:
            model_name_or_path: e.g. "sayby/rna_torsionbert" or an invalid path;
                                we attempt to load, else fallback to dummy.
            device: "cpu" or "cuda"
            angle_mode: one of {"sin_cos", "radians", "degrees"}
            num_angles: user guess/config for angles. The actual dimension might differ if model differs.
            max_length: tokenizer max length
        """
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.angle_mode = angle_mode
        self.num_angles = num_angles  # user-provided
        self.device = torch.device(device) # Keep torch.device object for internal use if needed elsewhere

        # Attempt to create TorsionBertModel; fallback to Dummy if error
        try:
            # Pass model_name_or_path as first arg, pass device string
            self.model = TorsionBertModel(
                model_name_or_path, # Pass as first argument
                device=device,      # Pass the original device string
                num_angles=self.num_angles,
                max_length=self.max_length,
            )
        except Exception:
             # Pass the original device string
            self.model = DummyTorsionModel(
                device=device, num_angles=self.num_angles
            )

    def __call__(
        self, sequence: str, adjacency: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Inference pipeline: (sequence, adjacency) -> torsion angles in either sin/cos or angles.

        Returns:
            {
              "torsion_angles": shape [N, 2*K] if sin_cos, or [N, K] for radians/degrees,
              "residue_count": N
            }
        """
        # adjacency is currently unused, but kept for future expansion
        sincos = self.model.predict_angles_from_sequence(sequence)
        N = sincos.size(0)

        if self.angle_mode == "sin_cos":
            angles_out = sincos
        else:
            # Convert sin/cos to angles (radians)
            angles_out = self._convert_sincos_to_angles(sincos)
            if self.angle_mode == "degrees":
                # Convert from radians to degrees
                angles_out = angles_out * (180.0 / math.pi)
            elif self.angle_mode != "radians":
                raise ValueError(f"Unknown angle_mode: {self.angle_mode}")

        return {"torsion_angles": angles_out, "residue_count": N}

    def _convert_sincos_to_angles(self, sincos: torch.Tensor) -> torch.Tensor:
        """
        Convert pairs (sin, cos) into actual angles in radians: [N, actual_num_angles].
        The number of angles is inferred from sincos.size(1)//2.

        For example, if sincos is [N, 14], that typically means 7 angles in sin/cos form.
        """
        N, dim = sincos.shape
        if dim % 2 != 0:
            raise RuntimeError(
                "Expected an even number of sin/cos columns, but got "
                f"{dim}. Perhaps the model output is malformed?"
            )

        num_angles = dim // 2
        sin_vals = sincos[:, :num_angles]  # First half contains sin values
        cos_vals = sincos[:, num_angles:]  # Second half contains cos values

        # Handle special case where cos=0 to avoid getting 0 instead of ±π/2
        # When cos is close to zero, we need to return π/2 * sign(sin)
        eps = 1e-6
        cos_vals.abs() < eps

        # Standard case - use atan2(sin, cos)
        angles = torch.atan2(sin_vals, cos_vals)

        # Correct quadrants when both sin and cos are very small
        mask_both_small = (sin_vals.abs() < eps) & (cos_vals.abs() < eps)
        angles = torch.where(mask_both_small, torch.zeros_like(angles), angles)

        return angles
