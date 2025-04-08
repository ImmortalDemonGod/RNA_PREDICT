"""
Components for atom attention module.
"""

from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components.attention_components import (
    AttentionComponents,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components.coordinate_processing import (
    CoordinateProcessor,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components.feature_processing import (
    FeatureProcessor,
)

__all__ = [
    "AttentionComponents",
    "CoordinateProcessor",
    "FeatureProcessor",
]
