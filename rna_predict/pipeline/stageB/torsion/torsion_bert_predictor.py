import math
from typing import Any, Dict, Optional

import torch

from rna_predict.pipeline.stageB.torsion.torsionbert_inference import TorsionBertModel


class StageBTorsionBertPredictor:
    """
    Stage B: predict RNA torsion angles using TorsionBERT.
    This version no longer forces output shape = [N, 2*self.num_angles];
    Instead, it relies on the actual dimension of model output.
    """

    def __init__(
        self,
        model_name_or_path: str = "sayby/rna_torsionbert",
        device: str = "cpu",
        angle_mode: str = "sin_cos",
        num_angles: int = 7,
        max_length: int = 512,
    ):
        """
        Args:
            model_name_or_path: e.g. "sayby/rna_torsionbert"
            device: "cpu" or "cuda"
            angle_mode: one of {"sin_cos", "radians", "degrees"}
            num_angles: user guess or config for angles. The actual dimension might differ if model differs.
            max_length: tokenizer max length
        """
        self.angle_mode = angle_mode
        self.num_angles = num_angles  # user-provided
        self.device = torch.device(device)

        # Create underlying TorsionBertModel (which also uses a user_requested_num_angles)
        self.model = TorsionBertModel(
            model_name_or_path=model_name_or_path,
            device=self.device,
            num_angles=self.num_angles,
            max_length=max_length,
        )

    def __call__(
        self,
        sequence: str,
        adjacency: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Inference pipeline: (sequence, adjacency) -> torsion angles in either sin/cos or angles.
        Returns:
            {
              "torsion_angles": shape [N, 2*K] if sin_cos, or [N, K] otherwise,
              "residue_count": N
            }
        """
        sincos = self.model.predict_angles_from_sequence(sequence)
        N = sincos.size(0)
        # adjacency is currently unused
        _ = adjacency

        if self.angle_mode == "sin_cos":
            angles_out = sincos
        else:
            angles_out = self._convert_sincos_to_angles(sincos)
            if self.angle_mode == "degrees":
                angles_out = angles_out * (180.0 / math.pi)
            elif self.angle_mode != "radians":
                raise ValueError(f"Unknown angle_mode: {self.angle_mode}")

        return {
            "torsion_angles": angles_out,
            "residue_count": N
        }

    def _convert_sincos_to_angles(self, sincos: torch.Tensor) -> torch.Tensor:
        """
        Convert pairs (sin, cos) into actual angles in radians: [N, actual_num_angles].
        The number of angles is inferred from sincos.size(1)//2
        """
        N, dim = sincos.shape
        if dim % 2 != 0:
            raise RuntimeError(
                "Expected an even number of sin/cos columns, but got "
                f"{dim}. Perhaps the model output is malformed?"
            )

        # actual_num_angles is half of the sincos dimension
        actual_num_angles = dim // 2
        sin_vals = sincos[:, 0::2]
        cos_vals = sincos[:, 1::2]
        angles = torch.atan2(sin_vals, cos_vals)
        return angles