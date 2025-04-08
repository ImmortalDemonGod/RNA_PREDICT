"""
Configuration dataclasses for AtomAttentionEncoder components.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
)


@dataclass
class AtomAttentionConfig:
    """Configuration parameters for atom attention modules."""

    has_coords: bool
    c_token: int  # token embedding dim (384 or 768)
    c_atom: int = 128  # atom embedding dim
    c_atompair: int = 16  # atom pair embedding dim
    c_s: int = 384  # single embedding dim
    c_z: int = 128  # pair embedding dim
    n_blocks: int = 3
    n_heads: int = 4
    n_queries: int = 32
    n_keys: int = 128
    blocks_per_ckpt: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate config parameters."""
        if self.c_atom <= 0:
            raise ValueError(f"c_atom must be positive, got {self.c_atom}")
        if self.c_token <= 0:
            raise ValueError(f"c_token must be positive, got {self.c_token}")
        if self.n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {self.n_blocks}")


@dataclass
class EncoderForwardParams:
    """Parameters for AtomAttentionEncoder.forward method."""

    input_feature_dict: InputFeatureDict
    r_l: Optional[torch.Tensor] = None
    s: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None
    inplace_safe: bool = False
    chunk_size: Optional[int] = None


@dataclass
class ProcessInputsParams:
    """Parameters for input processing methods."""

    input_feature_dict: InputFeatureDict
    r_l: Optional[torch.Tensor]
    s: Optional[torch.Tensor]
    z: Optional[torch.Tensor]
    c_l: torch.Tensor
    chunk_size: Optional[int] = None
