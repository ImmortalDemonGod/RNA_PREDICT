"""
Transformer modules for RNA structure prediction.
"""
from rna_predict.pipeline.stageA.input_embedding.current.transformer.attention import (
    AttentionPairBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.transition import (
    ConditionedTransitionBlock,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.diffusion import (
    DiffusionTransformerBlock,
    DiffusionTransformer,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_transformer import (
    AtomTransformer,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention import (
    AtomAttentionEncoder,
    AtomAttentionDecoder,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
    safe_tensor_access,
    validate_tensor_shape,
)

__all__ = [
    # Attention modules
    "AttentionPairBias",
    
    # Transition and diffusion modules
    "ConditionedTransitionBlock",
    "DiffusionTransformerBlock",
    "DiffusionTransformer",
    
    # Atom transformer modules
    "AtomTransformer",
    "AtomAttentionEncoder",
    "AtomAttentionDecoder",
    
    # Common utilities
    "InputFeatureDict",
    "safe_tensor_access",
    "validate_tensor_shape",
] 