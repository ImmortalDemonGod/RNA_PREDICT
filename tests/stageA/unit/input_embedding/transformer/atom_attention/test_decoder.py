# tests/stageA/unit/input_embedding/transformer/atom_attention/test_decoder.py
import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.config import (
    AtomAttentionConfig,
    DecoderForwardParams,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.decoder import (
    AtomAttentionDecoder,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components import (
    AttentionComponents,
    CoordinateProcessor,
    FeatureProcessor,
)


# --- Fixtures ---

@pytest.fixture(scope="module")
def default_decoder_config() -> AtomAttentionConfig:
    """Provides a default AtomAttentionConfig for the decoder."""
    # Using smaller dimensions for faster testing
    return AtomAttentionConfig(
        has_coords=True,  # Decoder always uses coordinates
        c_token=64,
        c_atom=32,
        c_atompair=8,
        c_s=64,  # Needed by FeatureProcessor/CoordinateProcessor
        c_z=32,  # Needed by FeatureProcessor/CoordinateProcessor
        n_blocks=2,
        n_heads=2,
        n_queries=16,
        n_keys=32,
        blocks_per_ckpt=None,
    )


@pytest.fixture(scope="module")
def decoder(default_decoder_config: AtomAttentionConfig) -> AtomAttentionDecoder:
    """Provides an AtomAttentionDecoder instance initialized with default config."""
    return AtomAttentionDecoder(default_decoder_config)


@pytest.fixture(scope="module")
def decoder_ckpt(default_decoder_config: AtomAttentionConfig) -> AtomAttentionDecoder:
    """Provides an AtomAttentionDecoder instance with blocks_per_ckpt set."""
    # Create a mutable copy or re-instantiate to avoid modifying the shared fixture state
    config_dict = default_decoder_config.__dict__.copy()
    config_dict['blocks_per_ckpt'] = 1
    config_with_ckpt = AtomAttentionConfig(**config_dict)
    return AtomAttentionDecoder(config_with_ckpt)


@pytest.fixture(scope="module")
def forward_params_minimal(default_decoder_config: AtomAttentionConfig) -> DecoderForwardParams:
    """Provides minimal DecoderForwardParams for testing."""
    batch_size = 2
    n_tokens = 10
    n_atoms = 25 # Must be >= n_tokens if atom_to_token_idx is used later
    c_token = default_decoder_config.c_token
    c_atom = default_decoder_config.c_atom

    # Input tensor 'a' has token dimension initially
    a = torch.randn(batch_size, n_tokens, c_token)
    # Coordinates tensor 'r_l' has atom dimension
    r_l = torch.randn(batch_size, n_atoms, 3)

    return DecoderForwardParams(a=a, r_l=r_l)


@pytest.fixture(scope="function") # Use function scope if modifying params inside tests
def forward_params_full(default_decoder_config: AtomAttentionConfig) -> DecoderForwardParams:
    """Provides DecoderForwardParams with all optional fields populated."""
    batch_size = 2
    n_tokens = 10
    n_atoms = 25
    c_token = default_decoder_config.c_token
    c_atom = default_decoder_config.c_atom
    c_extra = 5 # Dimension for extra_feats

    a = torch.randn(batch_size, n_tokens, c_token)
    r_l = torch.randn(batch_size, n_atoms, 3)
    extra_feats = torch.randn(batch_size, n_atoms, c_extra)
    mask = torch.randint(0, 2, (batch_size, n_atoms), dtype=torch.bool)
    # Ensure indices are within the range of n_tokens
    atom_to_token_idx = torch.randint(0, n_tokens, (batch_size, n_atoms), dtype=torch.long)
    chunk_size = 1 # Small chunk size for testing

    return DecoderForwardParams(
        a=a,
        r_l=r_l,
        extra_feats=extra_feats,
        mask=mask,
        atom_to_token_idx=atom_to_token_idx,
        chunk_size=chunk_size,
    )


# --- Test Cases ---

def test_decoder_initialization(decoder: AtomAttentionDecoder, default_decoder_config: AtomAttentionConfig) -> None:
    """
    Tests the initialization of the AtomAttentionDecoder (Lines 32-72).
    Verifies attributes and sub-module instantiation.
    """
    cfg = default_decoder_config
    assert decoder.c_atom == cfg.c_atom
    assert decoder.c_atompair == cfg.c_atompair
    assert decoder.c_token == cfg.c_token
    assert decoder.n_queries == cfg.n_queries
    assert decoder.n_keys == cfg.n_keys

    # Check sub-module types
    assert isinstance(decoder.feature_processor, FeatureProcessor)
    assert isinstance(decoder.coordinate_processor, CoordinateProcessor)
    assert isinstance(decoder.attention_components, AttentionComponents)
    assert isinstance(decoder.linear_no_bias_a, nn.Linear)
    assert isinstance(decoder.linear_no_bias_out, nn.Linear)

    # Check dimensions of linear layers
    assert decoder.linear_no_bias_a.in_features == cfg.c_token
    assert decoder.linear_no_bias_a.out_features == cfg.c_atom
    assert decoder.linear_no_bias_out.in_features == cfg.c_atom
    assert decoder.linear_no_bias_out.out_features == cfg.c_atom

    # Check config propagation to sub-modules
    assert decoder.feature_processor.c_atom == cfg.c_atom
    assert decoder.coordinate_processor.c_atom == cfg.c_atom
    assert decoder.attention_components.c_atom == cfg.c_atom
    assert decoder.attention_components.atom_transformer.n_blocks == cfg.n_blocks
    assert decoder.attention_components.atom_transformer.n_heads == cfg.n_heads
    # Check default blocks_per_ckpt handling (line 61)
    assert decoder.attention_components.atom_transformer.diffusion_transformer.blocks_per_ckpt == (cfg.blocks_per_ckpt or 0)


def test_decoder_initialization_with_ckpt(decoder_ckpt: AtomAttentionDecoder) -> None:
    """
    Tests initialization when blocks_per_ckpt is explicitly set (Line 61).
    """
    assert decoder_ckpt.attention_components.atom_transformer.diffusion_transformer.blocks_per_ckpt == 1


def test_forward_minimal(decoder: AtomAttentionDecoder, forward_params_minimal: DecoderForwardParams, default_decoder_config: AtomAttentionConfig) -> None:
    """
    Tests the forward pass with minimal inputs (Lines 88, 108-111, 114, 117, 122-123, 127).
    Covers paths where optional parameters are None.
    """
    params = forward_params_minimal
    batch_size, n_tokens, _ = params.a.shape
    _, n_atoms, _ = params.r_l.shape # Use n_atoms from r_l as 'a' is not broadcasted here

    output = decoder(params)

    assert isinstance(output, torch.Tensor)
    # Output shape should be (batch_size, n_tokens, c_atom) because atom_to_token_idx is None
    assert output.shape == (batch_size, n_tokens, default_decoder_config.c_atom)


def test_forward_with_atom_to_token_idx(decoder: AtomAttentionDecoder, forward_params_full: DecoderForwardParams, default_decoder_config: AtomAttentionConfig) -> None:
    """
    Tests the forward pass when atom_to_token_idx is provided (Lines 91-94).
    """
    params = forward_params_full
    # Remove other optional params to isolate atom_to_token_idx effect
    params.extra_feats = None
    params.mask = None
    params.chunk_size = None

    batch_size, _, _ = params.a.shape
    _, n_atoms = params.atom_to_token_idx.shape

    output = decoder(params)

    assert isinstance(output, torch.Tensor)
    # Output shape should now be (batch_size, n_atoms, c_atom) due to broadcasting
    assert output.shape == (batch_size, n_atoms, default_decoder_config.c_atom)


def test_forward_with_extra_feats(decoder: AtomAttentionDecoder, forward_params_full: DecoderForwardParams, default_decoder_config: AtomAttentionConfig) -> None:
    """
    Tests the forward pass when extra_feats is provided (Lines 97-100, 103-106).
    """
    params = forward_params_full
    # Remove other optional params to isolate extra_feats effect
    params.mask = None
    params.atom_to_token_idx = None # Keep 'a' at token level for this specific test focus
    params.chunk_size = None

    batch_size, n_tokens, _ = params.a.shape
    _, n_atoms, _ = params.r_l.shape

    output = decoder(params)

    assert isinstance(output, torch.Tensor)
    # Output shape remains (batch_size, n_tokens, c_atom) as atom_to_token_idx is None
    assert output.shape == (batch_size, n_tokens, default_decoder_config.c_atom)


def test_forward_with_mask(decoder: AtomAttentionDecoder, forward_params_full: DecoderForwardParams, default_decoder_config: AtomAttentionConfig) -> None:
    """
    Tests the forward pass when a mask is provided (Line 119).
    Requires atom_to_token_idx to ensure 'a' has atom dimension for mask compatibility.
    """
    params = forward_params_full
    # Ensure atom_to_token_idx is present so 'a' gets broadcasted to atom dimension
    assert params.atom_to_token_idx is not None
    # Remove other optional params
    params.extra_feats = None
    params.chunk_size = None

    batch_size, n_atoms = params.mask.shape

    output = decoder(params)

    assert isinstance(output, torch.Tensor)
    # Output shape should be (batch_size, n_atoms, c_atom)
    assert output.shape == (batch_size, n_atoms, default_decoder_config.c_atom)


def test_forward_with_chunk_size(decoder: AtomAttentionDecoder, forward_params_full: DecoderForwardParams, default_decoder_config: AtomAttentionConfig) -> None:
    """
    Tests the forward pass when chunk_size is provided (Line 123).
    Requires atom_to_token_idx to ensure 'a' has atom dimension.
    """
    params = forward_params_full
    # Ensure atom_to_token_idx is present
    assert params.atom_to_token_idx is not None
    assert params.chunk_size is not None
    # Remove other optional params
    params.extra_feats = None
    params.mask = None # Let it default to ones

    batch_size, n_atoms = params.atom_to_token_idx.shape

    output = decoder(params)

    assert isinstance(output, torch.Tensor)
    # Output shape should be (batch_size, n_atoms, c_atom)
    assert output.shape == (batch_size, n_atoms, default_decoder_config.c_atom)


def test_forward_full(decoder: AtomAttentionDecoder, forward_params_full: DecoderForwardParams, default_decoder_config: AtomAttentionConfig) -> None:
    """
    Tests the forward pass with all optional parameters provided (Lines 88-127).
    """
    params = forward_params_full
    assert params.atom_to_token_idx is not None
    assert params.extra_feats is not None
    assert params.mask is not None
    assert params.chunk_size is not None

    batch_size, n_atoms = params.atom_to_token_idx.shape

    output = decoder(params)

    assert isinstance(output, torch.Tensor)
    # Output shape should be (batch_size, n_atoms, c_atom)
    assert output.shape == (batch_size, n_atoms, default_decoder_config.c_atom)


def test_from_args_defaults() -> None:
    """
    Tests creating an AtomAttentionDecoder using from_args with default values (Lines 157-168).
    """
    decoder = AtomAttentionDecoder.from_args()
    assert isinstance(decoder, AtomAttentionDecoder)
    # Check a few default config values
    assert decoder.c_token == 384 # Default from signature
    assert decoder.c_atom == 128  # Default from signature
    assert decoder.n_blocks == 3   # Default from signature
    assert decoder.n_heads == 4    # Default from signature
    assert decoder.blocks_per_ckpt == 0 # Default handling


@pytest.mark.parametrize(
    "custom_args",
    [
        {"n_blocks": 5, "n_heads": 8, "c_token": 512, "c_atom": 64, "c_atompair": 12, "n_queries": 24, "n_keys": 48, "blocks_per_ckpt": 2},
        {"n_blocks": 1, "n_heads": 1, "c_token": 128, "c_atom": 16, "blocks_per_ckpt": None},
        # Add case matching default signature values explicitly
        {"n_blocks": 3, "n_heads": 4, "c_token": 384, "c_atom": 128, "c_atompair": 16, "n_queries": 32, "n_keys": 128, "blocks_per_ckpt": None},
    ]
)
def test_from_args_custom(custom_args: Dict[str, Any]) -> None:
    """
    Tests creating an AtomAttentionDecoder using from_args with custom values (Lines 157-168).
    """
    # Define defaults for args not always present in parametrize
    full_args = {
        "n_blocks": 3,
        "n_heads": 4,
        "c_token": 384,
        "c_atom": 128,
        "c_atompair": 16,
        "n_queries": 32,
        "n_keys": 128,
        "blocks_per_ckpt": None,
        **custom_args # Overwrite defaults with test case specifics
    }

    decoder = AtomAttentionDecoder.from_args(**full_args)
    assert isinstance(decoder, AtomAttentionDecoder)

    # Verify config attributes match provided args
    assert decoder.n_blocks == full_args["n_blocks"]
    assert decoder.n_heads == full_args["n_heads"]
    assert decoder.c_token == full_args["c_token"]
    assert decoder.c_atom == full_args["c_atom"]
    assert decoder.c_atompair == full_args["c_atompair"]
    assert decoder.n_queries == full_args["n_queries"]
    assert decoder.n_keys == full_args["n_keys"]

    # Verify blocks_per_ckpt handling
    expected_ckpt = full_args.get("blocks_per_ckpt", None)
    assert decoder.blocks_per_ckpt == (expected_ckpt or 0)