# tests/stageA/unit/input_embedding/current/transformer/atom_attention/test_encoder.py
import pytest
import torch
from unittest.mock import patch

from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.encoder import (
    AtomAttentionEncoder,
    AtomAttentionConfig,
    EncoderForwardParams,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components import (
    FeatureProcessor,
    CoordinateProcessor,
    AttentionComponents,
)


@pytest.fixture
def device():
    """Fixture to provide the device (CPU for consistency in tests)."""
    return torch.device("cpu")


@pytest.fixture
def basic_config():
    """
    Returns a basic configuration for the AtomAttentionEncoder.
    
    This fixture constructs an AtomAttentionConfig instance with predefined
    parameters for testing, including enabled coordinate support and set dimensions for
    tokens, atoms, atom pairs, and related hyperparameters.
    """
    return AtomAttentionConfig(
        has_coords=True,
        c_token=128,
        c_atom=64,
        c_atompair=16,
        c_s=32,
        c_z=16,
        n_blocks=2,
        n_heads=4,
        n_queries=8,
        n_keys=8,
        blocks_per_ckpt=None,
    )


@pytest.fixture
def encoder(basic_config):
    """
    Provides an AtomAttentionEncoder instance for testing.
    
    This fixture initializes and returns an AtomAttentionEncoder using the provided
    basic configuration, which contains the necessary parameters for setting up the encoder.
    """
    return AtomAttentionEncoder(basic_config)


@pytest.fixture
def input_feature_dict(device):
    """
    Generates a dictionary of mock input features for testing the encoder.
    
    This fixture constructs a dictionary with predefined tensors that simulate various input
    features required by the AtomAttentionEncoder. It includes an atom-to-token mapping,
    random reference positions, charges, masks, element data, atom name characters, unique
    space identifiers, residue types, profile features, and deletion statistics, all
    allocated on the specified device.
    
    Args:
        device: The torch device on which tensors are created.
    
    Returns:
        A dictionary with the following keys:
          - "atom_to_token_idx": Tensor of shape (16, 1) mapping each atom to its token.
          - "ref_pos": Tensor of shape (16, 3) with random values representing atomic positions.
          - "ref_charge": Tensor of shape (16, 1) filled with zeros for atomic charges.
          - "ref_mask": Boolean tensor of shape (16, 1) indicating valid atoms.
          - "ref_element": Tensor of shape (16, 128) filled with zeros representing element data.
          - "ref_atom_name_chars": Tensor of shape (16, 256) filled with zeros for atom name encoding.
          - "ref_space_uid": Tensor of shape (16, 1) filled with zeros for unique space identifiers.
          - "restype": Tensor of shape (1, 4, 32) filled with zeros representing residue types.
          - "profile": Tensor of shape (1, 4, 32) filled with zeros for profile features.
          - "deletion_mean": Tensor of shape (1, 4, 1) filled with zeros for deletion statistics.
    """
    n_atom = 16
    n_token = 4

    # Create atom-to-token mapping (4 atoms per token)
    atom_to_token_idx = torch.repeat_interleave(
        torch.arange(n_token, device=device), n_atom // n_token
    ).unsqueeze(-1)

    return {
        "atom_to_token_idx": atom_to_token_idx,
        "ref_pos": torch.randn(n_atom, 3, device=device),
        "ref_charge": torch.zeros(n_atom, 1, device=device),
        "ref_mask": torch.ones(n_atom, 1, dtype=torch.bool, device=device),
        "ref_element": torch.zeros(n_atom, 128, device=device),
        "ref_atom_name_chars": torch.zeros(n_atom, 256, device=device),
        "ref_space_uid": torch.zeros(n_atom, 1, device=device),
        "restype": torch.zeros(1, n_token, 32, device=device),
        "profile": torch.zeros(1, n_token, 32, device=device),
        "deletion_mean": torch.zeros(1, n_token, 1, device=device),
    }


def test_encoder_initialization(basic_config):
    """Test that the encoder initializes correctly with the given configuration."""
    encoder = AtomAttentionEncoder(basic_config)

    # Check that the encoder has the correct attributes
    assert encoder.has_coords == basic_config.has_coords
    assert encoder.c_atom == basic_config.c_atom
    assert encoder.c_atompair == basic_config.c_atompair
    assert encoder.c_token == basic_config.c_token
    assert encoder.c_s == basic_config.c_s
    assert encoder.c_z == basic_config.c_z
    assert encoder.n_queries == basic_config.n_queries
    assert encoder.n_keys == basic_config.n_keys

    # Check that the encoder has the correct components
    assert isinstance(encoder.feature_processor, FeatureProcessor)
    assert isinstance(encoder.coordinate_processor, CoordinateProcessor)
    assert isinstance(encoder.attention_components, AttentionComponents)
    assert encoder.linear_no_bias_q.in_features == basic_config.c_atom
    assert encoder.linear_no_bias_q.out_features == basic_config.c_token


def test_encoder_from_args():
    """Test the from_args class method."""
    encoder = AtomAttentionEncoder.from_args(
        has_coords=True,
        c_token=128,
        c_atom=64,
        c_atompair=16,
        c_s=32,
        c_z=16,
        n_blocks=2,
        n_heads=4,
        n_queries=8,
        n_keys=8,
    )

    # Check that the encoder has the correct attributes
    assert encoder.has_coords is True
    assert encoder.c_token == 128
    assert encoder.c_atom == 64
    assert encoder.c_atompair == 16
    assert encoder.c_s == 32
    assert encoder.c_z == 16
    assert encoder.n_queries == 8
    assert encoder.n_keys == 8


def test_encoder_forward_with_coords(encoder, input_feature_dict, device):
    """Test the forward pass of the encoder with coordinate information."""
    # Create mock return values for the feature processor
    c_l = torch.randn(16, 64, device=device)  # [n_atom, c_atom]
    p_l = torch.randn(16, 16, device=device)  # [n_atom, c_atompair]

    # Create mock return values for the attention components
    a_atom = torch.randn(16, 64, device=device)  # [n_atom, c_atom]

    # Create mock return values for the coordinate processor
    r_l = torch.randn(16, 3, device=device)  # [n_atom, 3]
    s = torch.randn(4, 32, device=device)  # [n_token, c_s]
    z = torch.randn(4, 4, 16, device=device)  # [n_token, n_token, c_z]

    # Create the forward parameters
    params = EncoderForwardParams(
        input_feature_dict=input_feature_dict,
        r_l=r_l,
        s=s,
        z=z,
    )

    # Mock the feature processor methods
    with patch.object(
        encoder.feature_processor, "extract_atom_features", return_value=c_l
    ) as mock_extract, patch.object(
        encoder.feature_processor, "create_pair_embedding", return_value=p_l
    ) as mock_create_pair, patch.object(
        encoder.attention_components, "apply_transformer", return_value=a_atom
    ) as mock_apply, patch.object(
        encoder.coordinate_processor, "process_coordinate_encoding", return_value=a_atom
    ) as mock_process_coord, patch.object(
        encoder.coordinate_processor, "process_style_embedding", return_value=a_atom
    ) as mock_process_style, patch.object(
        encoder.coordinate_processor, "process_pair_embedding", return_value=p_l
    ) as mock_process_pair, patch.object(
        encoder.linear_no_bias_q, "forward", return_value=torch.randn(4, 128, device=device)
    ):

        # Call the forward method
        result = encoder.forward(params)

        # Check that the result is a tuple of the expected length
        assert isinstance(result, tuple)
        assert len(result) == 4

        # Check that the methods were called with the correct arguments
        mock_extract.assert_called_once_with(input_feature_dict)
        mock_create_pair.assert_called_once_with(input_feature_dict)
        mock_apply.assert_called_once()
        mock_process_coord.assert_called_once()
        mock_process_style.assert_called_once()
        mock_process_pair.assert_called_once()


def test_encoder_forward_without_coords(basic_config, input_feature_dict, device):
    """
    Test the encoder's forward pass when coordinate data is not provided.
    
    This test verifies that the AtomAttentionEncoder handles input without
    coordinate information by ensuring the forward method returns a tuple of
    four elements, where the final element is None. It also confirms that the
    feature extraction and attention transformation routines are invoked as expected.
    """
    # Create an encoder without coordinate information
    config = basic_config
    config.has_coords = False
    encoder = AtomAttentionEncoder(config)

    # Create mock return values for the feature processor
    c_l = torch.randn(16, 64, device=device)  # [n_atom, c_atom]
    p_l = torch.randn(16, 16, device=device)  # [n_atom, c_atompair]

    # Create mock return values for the attention components
    a_atom = torch.randn(16, 64, device=device)  # [n_atom, c_atom]

    # Create the forward parameters (without r_l, s, z)
    params = EncoderForwardParams(
        input_feature_dict=input_feature_dict,
        r_l=None,
        s=None,
        z=None,
    )

    # Mock the feature processor methods
    with patch.object(
        encoder.feature_processor, "extract_atom_features", return_value=c_l
    ) as mock_extract, patch.object(
        encoder.feature_processor, "create_pair_embedding", return_value=p_l
    ) as mock_create_pair, patch.object(
        encoder.attention_components, "apply_transformer", return_value=a_atom
    ) as mock_apply, patch.object(
        encoder.linear_no_bias_q, "forward", return_value=torch.randn(4, 128, device=device)
    ):

        # Call the forward method
        result = encoder.forward(params)

        # Check that the result is a tuple of the expected length
        assert isinstance(result, tuple)
        assert len(result) == 4

        # Check that the methods were called with the correct arguments
        mock_extract.assert_called_once_with(input_feature_dict)
        mock_create_pair.assert_called_once_with(input_feature_dict)
        mock_apply.assert_called_once()

        # Check that the last element of the result is None (r_l)
        assert result[3] is None


def test_encoder_forward_with_custom_mask(encoder, input_feature_dict, device):
    """Test the forward pass of the encoder with a custom mask."""
    # Create mock return values for the feature processor
    c_l = torch.randn(16, 64, device=device)  # [n_atom, c_atom]
    p_l = torch.randn(16, 16, device=device)  # [n_atom, c_atompair]

    # Create mock return values for the attention components
    a_atom = torch.randn(16, 64, device=device)  # [n_atom, c_atom]

    # Create a custom mask
    custom_mask = torch.ones(16, dtype=torch.bool, device=device)
    custom_mask[0] = False  # Mask out the first atom
    input_feature_dict["ref_mask"] = custom_mask.unsqueeze(-1)

    # Create the forward parameters
    params = EncoderForwardParams(
        input_feature_dict=input_feature_dict,
        r_l=None,
        s=None,
        z=None,
    )

    # Mock the feature processor methods
    with patch.object(
        encoder.feature_processor, "extract_atom_features", return_value=c_l
    ) as mock_extract, patch.object(
        encoder.feature_processor, "create_pair_embedding", return_value=p_l
    ) as mock_create_pair, patch.object(
        encoder.attention_components, "apply_transformer", return_value=a_atom
    ) as mock_apply, patch.object(
        encoder.linear_no_bias_q, "forward", return_value=torch.randn(4, 128, device=device)
    ):

        # Call the forward method
        result = encoder.forward(params)

        # Check that the result is a tuple of the expected length
        assert isinstance(result, tuple)
        assert len(result) == 4

        # Check that the methods were called with the correct arguments
        mock_extract.assert_called_once_with(input_feature_dict)
        mock_create_pair.assert_called_once_with(input_feature_dict)

        # Check that apply_transformer was called
        assert mock_apply.call_count == 1


def test_encoder_forward_without_mask(encoder, input_feature_dict, device):
    """Test the forward pass of the encoder without a mask in the input dictionary."""
    # Create mock return values for the feature processor
    c_l = torch.randn(16, 64, device=device)  # [n_atom, c_atom]
    p_l = torch.randn(16, 16, device=device)  # [n_atom, c_atompair]

    # Create mock return values for the attention components
    a_atom = torch.randn(16, 64, device=device)  # [n_atom, c_atom]

    # Remove the mask from the input dictionary
    del input_feature_dict["ref_mask"]

    # Create the forward parameters
    params = EncoderForwardParams(
        input_feature_dict=input_feature_dict,
        r_l=None,
        s=None,
        z=None,
    )

    # Mock the feature processor methods
    with patch.object(
        encoder.feature_processor, "extract_atom_features", return_value=c_l
    ) as mock_extract, patch.object(
        encoder.feature_processor, "create_pair_embedding", return_value=p_l
    ) as mock_create_pair, patch.object(
        encoder.attention_components, "apply_transformer", return_value=a_atom
    ) as mock_apply, patch.object(
        encoder.linear_no_bias_q, "forward", return_value=torch.randn(4, 128, device=device)
    ):

        # Call the forward method
        result = encoder.forward(params)

        # Check that the result is a tuple of the expected length
        assert isinstance(result, tuple)
        assert len(result) == 4

        # Check that the methods were called with the correct arguments
        mock_extract.assert_called_once_with(input_feature_dict)
        mock_create_pair.assert_called_once_with(input_feature_dict)

        # Check that apply_transformer was called
        assert mock_apply.call_count == 1
