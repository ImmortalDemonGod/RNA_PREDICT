"""
Atom attention decoder module.
"""

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components import (
    AttentionComponents,
    CoordinateProcessor,
    FeatureProcessor,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.config import (
    AtomAttentionConfig,
    DecoderForwardParams,
)


class AtomAttentionDecoder(nn.Module):
    """
    Decoder that processes token-level features and produces atom-level embeddings.
    """

    def __init__(self, config: AtomAttentionConfig) -> None:
        """
        Initialize the AtomAttentionDecoder with a configuration object.

        Args:
            config: Configuration parameters for the decoder
        """
        super(AtomAttentionDecoder, self).__init__()
        self.c_atom = config.c_atom
        self.c_atompair = config.c_atompair
        self.c_token = config.c_token
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys

        # Initialize components
        self.feature_processor = FeatureProcessor(
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_s=config.c_s,
            c_z=config.c_z,
        )

        self.coordinate_processor = CoordinateProcessor(
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_s=config.c_s,
            c_z=config.c_z,
        )

        self.attention_components = AttentionComponents(
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            n_blocks=config.n_blocks,
            n_heads=config.n_heads,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            blocks_per_ckpt=config.blocks_per_ckpt or 0,  # Default to 0 if None
        )

        # Input projection from token dimension
        self.linear_no_bias_a = LinearNoBias(
            in_features=self.c_token, out_features=self.c_atom
        )

        # Output projection to atom dimension
        self.linear_no_bias_out = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atom
        )

    def forward(
        self,
        params: DecoderForwardParams,
    ) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            params: Forward pass parameters

        Returns:
            Atom-level embeddings
        """
        # Project input to atom dimension
        a = self.linear_no_bias_a(params.a)

        # Broadcast to atom level if needed
        if params.atom_to_token_idx is not None:
            a = self.feature_processor.broadcast_to_atom_level(
                a, params.atom_to_token_idx
            )

        # Process coordinate encoding
        if params.extra_feats is not None:
            a = self.coordinate_processor.process_coordinate_encoding(
                a, params.r_l, params.extra_feats
            )

        # Create pair embedding
        if params.extra_feats is not None:
            p_l = self.feature_processor.create_pair_embedding(
                {"ref_pos": params.extra_feats}
            )
        else:
            p_l = torch.zeros(
                (a.shape[0], self.n_queries * self.n_keys, self.c_atompair),
                device=a.device,
            )

        # Process pair features
        p_l = self.attention_components.process_pair_features(a, a)

        # Create attention mask if not provided
        mask = torch.ones_like(a[..., 0], dtype=torch.bool)
        if params.mask is not None:
            mask = params.mask

        # Apply transformer
        a = self.attention_components.apply_transformer(
            a, p_l, mask=mask, chunk_size=params.chunk_size or 0
        )

        # Project to output dimension
        return self.linear_no_bias_out(a)

    @classmethod
    def from_args(
        cls,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: int | None = None,
    ) -> "AtomAttentionDecoder":
        """
        Create an AtomAttentionDecoder instance from arguments.

        Args:
            n_blocks: Number of transformer blocks
            n_heads: Number of attention heads
            c_token: Token embedding dimension
            c_atom: Atom embedding dimension
            c_atompair: Atom pair embedding dimension
            n_queries: Number of queries
            n_keys: Number of keys
            blocks_per_ckpt: Number of blocks per checkpoint

        Returns:
            Configured AtomAttentionDecoder instance
        """
        config = AtomAttentionConfig(
            has_coords=True,  # Decoder always uses coordinates
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return cls(config)
