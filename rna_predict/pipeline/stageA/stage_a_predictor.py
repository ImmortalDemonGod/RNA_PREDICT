import torch
from typing import Dict

class StageAPredictor:
    """
    Demonstration Stage A: produces a zero adjacency for each residue.
    """
    def __call__(self, sequence: str) -> Dict[str, torch.Tensor]:
        N = len(sequence)
        # Create an N x N zero adjacency matrix
        adjacency = torch.zeros((N, N), dtype=torch.float32)
        return {
            "sequence": sequence,
            "adjacency": adjacency
        }
