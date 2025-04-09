"""
Type definitions for attention operations.

This module contains data structures and type definitions used in attention operations.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TypedDict, Union

import torch


@dataclass
class AttentionChunkConfig:
    """Configuration for chunked attention processing."""

    chunk_size: int
    n_chunks_q: int
    n_chunks_k: int


class PaddingInfo(TypedDict, total=False):
    """Type for padding information in attention operations."""

    q_mask: Union[List[torch.Tensor], None]
    k_mask: Union[List[torch.Tensor], None]
    mask_trunked: Union[torch.Tensor, None]


@dataclass
class LocalAttentionInputs:
    """Inputs for local attention."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    n_queries: int
    n_keys: int
    attn_bias: Optional[torch.Tensor] = None
    trunked_attn_bias: Optional[torch.Tensor] = None
    inf: float = 1e10
    use_efficient_implementation: bool = False
    attn_weight_dropout_p: float = 0.0
    inplace_safe: bool = False
    chunk_size: Optional[int] = None


@dataclass
class ChunkProcessingInputs:
    """Inputs for processing an attention chunk."""

    q_chunk: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    bias_slice: Optional[torch.Tensor]
    use_efficient_implementation: bool
    attn_weight_dropout_p: float
    inplace_safe: bool


@dataclass
class BiasCreationInputs:
    """Inputs for creating local attention bias."""

    n: int
    n_queries: int
    n_keys: int
    inf: float = 1e10
    device: Optional[torch.device] = None
