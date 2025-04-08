"""
Transformer modules for RNA structure prediction.

This file re-exports transformer components from the modular implementation.
"""

from typing import TypeVar

from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention_decoder import (
    AtomAttentionDecoder,
    DecoderForwardParams,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention_encoder import (
    AtomAttentionConfig,
    AtomAttentionEncoder,
    EncoderForwardParams,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_transformer import (
    AtomTransformer,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.attention import (
    AttentionPairBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
    safe_tensor_access,
    validate_tensor_shape,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.diffusion import (
    DiffusionTransformer,
    DiffusionTransformerBlock,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.transition import (
    ConditionedTransitionBlock,
)

# For backward compatibility, re-export types
T = TypeVar("T")

__all__ = [
    # Main components
    "AttentionPairBias",
    "ConditionedTransitionBlock",
    "DiffusionTransformerBlock",
    "DiffusionTransformer",
    "AtomTransformer",
    "AtomAttentionEncoder",
    "AtomAttentionDecoder",
    # Configuration
    "AtomAttentionConfig",
    "EncoderForwardParams",
    "DecoderForwardParams",
    # Utilities
    "InputFeatureDict",
    "safe_tensor_access",
    "validate_tensor_shape",
]
