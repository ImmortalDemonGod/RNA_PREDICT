"""
Data classes for dense trunk attention configurations and intermediate results.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from ..core_transforms import TrunkInfo


@dataclass
class DenseTrunkConfig:
    """Configuration for dense trunk operations."""

    n_queries: int
    n_keys: int
    attn_bias: Optional[torch.Tensor] = None
    inf: float = 1e10
    compute_mask: bool = True


@dataclass
class AttentionBiasConfig:
    """Configuration for attention bias processing."""

    n_q_trunks: int
    n_queries: int
    n_q_pad: int
    original_length: int
    inf: float = 1e10


@dataclass
class QueryTrunkInfo:
    """Information about query trunk processing."""

    trunked_tensor: torch.Tensor
    padding_length: int
    total_length: int
    num_trunks: int


@dataclass
class ChunkInfo:
    """Information about a tensor chunk."""

    start: int
    end: int
    length: int


@dataclass
class RearrangementConfig:
    """Configuration for tensor rearrangement operations."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    n_queries: int
    n_keys: int
    attn_bias: Optional[torch.Tensor] = None
    inf: float = 1e10


@dataclass
class MaskConfigParams:
    """Configuration parameters for creating a mask configuration."""

    n_queries: int
    n_keys: int
    q_list: List[torch.Tensor]
    k_list: List[torch.Tensor]
    dim_q_list: List[int]
    dim_k_list: List[int]
    n_q_trunks: int
    n_k_trunks: int
    total_q: int


@dataclass
class TrunkDimensionsParams:
    """Parameters for calculating trunk dimensions."""

    q_list: List[torch.Tensor]
    k_list: List[torch.Tensor]
    dim_q_list: List[int]
    dim_k_list: List[int]
    n_queries: int
    n_keys: int
    trunk_info: Optional[TrunkInfo] = None


@dataclass
class PaddingInfoParams:
    """Parameters for preparing padding information."""

    q_list: List[torch.Tensor]
    k_list: List[torch.Tensor]
    dim_q_list: List[int]
    dim_k_list: List[int]
    n_queries: int
    n_keys: int
    compute_mask: bool
    trunk_info: Optional[TrunkInfo] = None


@dataclass
class TensorChunkParams:
    """Parameters for processing a tensor chunk."""

    tensor: torch.Tensor
    chunk_info: ChunkInfo
    chunk_size: int
    padding_value: Union[float, int]
    device: torch.device
    fill_value_func: Callable[[Any], torch.Tensor]
    dim: int = -2


@dataclass
class KeysValuesChunkParams:
    """Parameters for processing keys and values into chunks."""

    k: torch.Tensor
    v: torch.Tensor
    n_keys: int
    n_k_trunks: int
    attn_bias: Optional[torch.Tensor] = None
    inf: float = 1e10