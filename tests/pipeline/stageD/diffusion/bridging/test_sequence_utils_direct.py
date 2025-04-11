"""
Direct tests for sequence_utils.py module.

This module tests the sequence extraction and processing utilities for Stage D diffusion.
"""

import pytest
import torch
from unittest.mock import patch

from rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils import extract_sequence


def test_explicit_sequence():
    """Test when sequence is explicitly provided."""
    sequence = ["A", "U", "G", "C"]
    result = extract_sequence(sequence, None, {})
    assert result == sequence
    assert result is sequence  # Should return the same object


def test_sequence_from_input_features_tensor():
    """Test extracting sequence from input_features as tensor."""
    input_features = {"sequence": torch.tensor([65, 85, 71, 67])}  # ASCII for "AUGC"
    result = extract_sequence(None, input_features, {})
    assert result == ["65", "85", "71", "67"]  # Converted to strings


def test_sequence_from_input_features_list():
    """Test extracting sequence from input_features as list."""
    input_features = {"sequence": ["A", "U", "G", "C"]}
    result = extract_sequence(None, input_features, {})
    assert result == ["A", "U", "G", "C"]


def test_sequence_from_input_features_string():
    """Test extracting sequence from input_features as string."""
    input_features = {"sequence": "AUGC"}
    result = extract_sequence(None, input_features, {})
    assert result == ["A", "U", "G", "C"]


@patch("rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils.logger")
def test_fallback_to_trunk_embeddings_s_trunk(mock_logger):
    """Test fallback to trunk_embeddings with s_trunk key."""
    trunk_embeddings = {"s_trunk": torch.zeros((1, 4, 16))}  # Batch, seq_len, dim
    result = extract_sequence(None, None, trunk_embeddings)
    assert result == ["A", "A", "A", "A"]  # 4 placeholder 'A's
    mock_logger.warning.assert_called_once()
    assert "No sequence provided" in mock_logger.warning.call_args[0][0]


@patch("rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils.logger")
def test_fallback_to_trunk_embeddings_s_inputs(mock_logger):
    """Test fallback to trunk_embeddings with s_inputs key."""
    trunk_embeddings = {"s_inputs": torch.zeros((1, 5, 16))}  # Batch, seq_len, dim
    result = extract_sequence(None, None, trunk_embeddings)
    assert result == ["A", "A", "A", "A", "A"]  # 5 placeholder 'A's
    mock_logger.warning.assert_called_once()
    assert "No sequence provided" in mock_logger.warning.call_args[0][0]


@patch("rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils.logger")
def test_fallback_to_trunk_embeddings_sing(mock_logger):
    """Test fallback to trunk_embeddings with sing key."""
    trunk_embeddings = {"sing": torch.zeros((1, 3, 16))}  # Batch, seq_len, dim
    result = extract_sequence(None, None, trunk_embeddings)
    assert result == ["A", "A", "A"]  # 3 placeholder 'A's
    mock_logger.warning.assert_called_once()
    assert "No sequence provided" in mock_logger.warning.call_args[0][0]


def test_error_when_no_sequence_available():
    """Test error when no sequence can be derived."""
    with pytest.raises(ValueError) as excinfo:
        extract_sequence(None, None, {})
    assert "Cannot derive sequence" in str(excinfo.value)


def test_input_features_without_sequence():
    """Test when input_features is provided but doesn't contain sequence."""
    input_features = {"other_key": "value"}  # No sequence key
    with pytest.raises(ValueError) as excinfo:
        extract_sequence(None, input_features, {})
    assert "Cannot derive sequence" in str(excinfo.value)


def test_trunk_embeddings_with_non_tensor_values():
    """Test when trunk_embeddings contains keys but with non-tensor values."""
    trunk_embeddings = {"s_trunk": "not a tensor"}
    with pytest.raises(ValueError) as excinfo:
        extract_sequence(None, None, trunk_embeddings)
    assert "Cannot derive sequence" in str(excinfo.value)
