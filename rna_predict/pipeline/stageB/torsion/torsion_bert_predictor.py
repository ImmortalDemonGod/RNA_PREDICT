import math
from typing import Any, Dict, Optional

import torch

from rna_predict.pipeline.stageB.torsion.torsionbert_inference import TorsionBertModel

class StageBTorsionBertPredictor:
    """
    Stage B: predict RNA torsion angles using TorsionBERT.
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
            num_angles: number of angles predicted
            max_length: tokenizer max length
        """
        self.angle_mode = angle_mode
        self.num_angles = num_angles
        self.device = torch.device(device)

        self.model = TorsionBertModel(
            model_name_or_path=model_name_or_path,
            device=self.device,
            num_angles=self.num_angles,
            max_length=max_length,
        )

    def __call__(
        self, sequence: str, adjacency: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Pipeline interface: (sequence, adjacency) -> torsion angles.
        Returns:
            {
              "torsion_angles": [N, 2*num_angles] (if sin_cos) or [N, num_angles],
              "residue_count": N
            }
        """
        # 1) Forward pass to get sin/cos predictions
        sincos = self.model.predict_angles_from_sequence(sequence)
        N = sincos.size(0)
        _ = adjacency  # currently unused

        # 2) Convert if mode is not sin_cos
        if self.angle_mode == "sin_cos":
            angles_out = sincos
        else:
            angles_out = self._convert_sincos_to_angles(sincos)
            if self.angle_mode == "degrees":
                angles_out = angles_out * (180.0 / math.pi)
            elif self.angle_mode != "radians":
                raise ValueError(f"Unknown angle_mode: {self.angle_mode}")
        return {"torsion_angles": angles_out, "residue_count": N}

    def _convert_sincos_to_angles(self, sincos: torch.Tensor) -> torch.Tensor:
        """
        Convert sin/cos pairs into angles in radians, shape [N, num_angles].
        """
        N, dim = sincos.shape
        expected = 2 * self.num_angles
        if dim != expected:
            raise RuntimeError(f"Expected {expected} columns, got {dim}")
        sin_vals = sincos[:, 0::2]
        cos_vals = sincos[:, 1::2]
        return torch.atan2(sin_vals, cos_vals)