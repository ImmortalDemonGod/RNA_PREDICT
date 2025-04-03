"""
Initialization logic for the AtomAttentionEncoder.
"""
from typing import Optional, Any

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_transformer import (
    AtomTransformer,
)
from .config import AtomAttentionConfig


def setup_feature_dimensions(encoder: Any) -> None: # Changed nn.Module to Any
    """Define expected feature dimensions."""
    encoder.input_feature = { # type: ignore
        "ref_pos": 3,
        "ref_charge": 1,
        "ref_mask": 1,
        "ref_element": 128,
        "ref_atom_name_chars": 4 * 64,
    }


def setup_atom_encoders(encoder: Any, config: AtomAttentionConfig) -> None: # Changed nn.Module to Any
    """Set up encoders for atom features."""
    encoder.linear_no_bias_f = LinearNoBias(
        in_features=sum(encoder.input_feature.values()), out_features=config.c_atom
    )


def setup_distance_encoders(encoder: Any, config: AtomAttentionConfig) -> None: # Changed nn.Module to Any
    """Set up encoders for distance-related features."""
    encoder.linear_no_bias_d = LinearNoBias(
        in_features=3, out_features=config.c_atompair
    )
    encoder.linear_no_bias_invd = LinearNoBias(
        in_features=1, out_features=config.c_atompair
    )
    encoder.linear_no_bias_v = LinearNoBias(
        in_features=1, out_features=config.c_atompair
    )


def setup_coordinate_components(encoder: Any, config: AtomAttentionConfig) -> None: # Changed nn.Module to Any
    """Set up components used when coordinates are available."""
    # Style normalization and projection
    encoder.layernorm_s = LayerNorm(config.c_s)
    encoder.linear_no_bias_s = LinearNoBias(
        in_features=config.c_s, out_features=config.c_atom
    )

    # Pair embedding normalization and projection
    encoder.layernorm_z = LayerNorm(config.c_z)  # memory bottleneck
    encoder.linear_no_bias_z = LinearNoBias(
        in_features=config.c_z, out_features=config.c_atompair
    )

    # Position encoder
    encoder.linear_no_bias_r = LinearNoBias(in_features=3, out_features=config.c_atom)


def setup_pair_projections(encoder: Any, config: AtomAttentionConfig) -> None: # Changed nn.Module to Any
    """Set up linear projections for atom features to pair dimension."""
    encoder.linear_no_bias_cl = LinearNoBias(
        in_features=config.c_atom, out_features=config.c_atompair
    )
    encoder.linear_no_bias_cm = LinearNoBias(
        in_features=config.c_atom, out_features=config.c_atompair
    )


def setup_small_mlp(encoder: Any, config: AtomAttentionConfig) -> None: # Changed nn.Module to Any
    """Set up small MLP for pair feature processing."""
    encoder.small_mlp = nn.Sequential(
        nn.ReLU(),
        LinearNoBias(in_features=config.c_atompair, out_features=config.c_atompair),
        nn.ReLU(),
        LinearNoBias(in_features=config.c_atompair, out_features=config.c_atompair),
        nn.ReLU(),
        LinearNoBias(in_features=config.c_atompair, out_features=config.c_atompair),
    )


def create_atom_transformer(config: AtomAttentionConfig) -> AtomTransformer:
    """
    Create the AtomTransformer instance.

    Args:
        config: Configuration object containing transformer parameters

    Returns:
        Configured AtomTransformer instance
    """
    return AtomTransformer(
        n_blocks=config.n_blocks,
        n_heads=config.n_heads,
        c_atom=config.c_atom,
        c_s=config.c_s,  # Pass the correct style dimension
        c_atompair=config.c_atompair,
        n_queries=config.n_queries,
        n_keys=config.n_keys,
        blocks_per_ckpt=config.blocks_per_ckpt,
    )


def init_residual_layers(encoder: Any, zero_init: bool) -> None: # Changed nn.Module to Any
    """
    Initialize residual connection layers.

    Args:
        encoder: The encoder module instance
        zero_init: Whether to zero-initialize the weights
    """
    if not zero_init:
        return

    # Always initialize these layers
    nn.init.zeros_(encoder.linear_no_bias_invd.weight)
    nn.init.zeros_(encoder.linear_no_bias_v.weight)
    nn.init.zeros_(encoder.linear_no_bias_cl.weight)
    nn.init.zeros_(encoder.linear_no_bias_cm.weight)

    # Initialize coordinate-dependent layers if needed
    if encoder.has_coords:
        nn.init.zeros_(encoder.linear_no_bias_s.weight)
        nn.init.zeros_(encoder.linear_no_bias_z.weight)
        nn.init.zeros_(encoder.linear_no_bias_r.weight)


def init_mlp_layers(encoder: Any, use_he_normal: bool) -> None: # Changed nn.Module to Any
    """
    Initialize MLP layers with He normal initialization.

    Args:
        encoder: The encoder module instance
        use_he_normal: Whether to use He normal initialization
    """
    if not use_he_normal:
        return

    for layer in encoder.small_mlp:
        if not isinstance(layer, torch.nn.modules.activation.ReLU):
            nn.init.kaiming_normal_(
                layer.weight,
                a=0,
                mode="fan_in",
                nonlinearity="relu",
            )


def init_output_layer(encoder: Any, use_he_normal: bool) -> None: # Changed nn.Module to Any
    """
    Initialize output layer with He normal initialization.

    Args:
        encoder: The encoder module instance
        use_he_normal: Whether to use He normal initialization
    """
    if not use_he_normal:
        return

    nn.init.kaiming_normal_(
        encoder.linear_no_bias_q.weight, a=0, mode="fan_in", nonlinearity="relu"
    )


def linear_init(
    encoder: Any, # Changed nn.Module to Any
    zero_init_atom_encoder_residual_linear: bool = False,
    he_normal_init_atom_encoder_small_mlp: bool = False,
    he_normal_init_atom_encoder_output: bool = False,
) -> None:
    """
    Initialize the parameters of the module.

    Args:
        encoder: The encoder module instance
        zero_init_atom_encoder_residual_linear: Whether to zero-initialize residual linear layers
        he_normal_init_atom_encoder_small_mlp: Whether to initialize MLP with He normal initialization
        he_normal_init_atom_encoder_output: Whether to initialize output with He normal initialization
    """
    init_residual_layers(encoder, zero_init_atom_encoder_residual_linear)
    init_mlp_layers(encoder, he_normal_init_atom_encoder_small_mlp)
    init_output_layer(encoder, he_normal_init_atom_encoder_output)