"""
Context dataclass for Stage D orchestration.
Bundles all tensor/config arguments for clarity and code quality.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch

@dataclass
class StageDContext:
    cfg: Any
    coords: torch.Tensor
    s_trunk: torch.Tensor
    z_trunk: torch.Tensor
    s_inputs: torch.Tensor
    input_feature_dict: Dict[str, Any]
    atom_metadata: Optional[Dict[str, Any]] = None
    trunk_embeddings: Optional[Dict[str, torch.Tensor]] = None
    features: Optional[Dict[str, Any]] = None
    stage_cfg: Optional[Any] = None
    diffusion_cfg: Optional[Any] = None
    c_s: Optional[int] = None
    c_s_inputs: Optional[int] = None
    c_z: Optional[int] = None
    device: Optional[str] = None
    mode: Optional[str] = None
    debug_logging: Optional[bool] = None
    residue_indices: Optional[Any] = None
    num_residues: Optional[int] = None
