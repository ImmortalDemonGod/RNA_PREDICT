"""
Tests for sequence_utils.py module.

This module tests the sequence extraction and processing utilities for Stage D diffusion.
"""

import logging
import os
import pytest
import torch
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

from hypothesis import given, strategies as st

from rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils import extract_sequence


class TestExtractSequence:
    """Tests for the extract_sequence function."""

    def test_explicit_sequence(self):
        """Test when sequence is explicitly provided."""
        # Arrange
        sequence = ["A", "U", "G", "C"]
        input_features = None
        trunk_embeddings = {}

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == sequence
        assert result is sequence  # Should return the same object

    def test_sequence_from_input_features_tensor(self):
        """Test extracting sequence from input_features as tensor."""
        # Arrange
        sequence = None
        input_features = {"sequence": torch.tensor([65, 85, 71, 67])}  # ASCII for "AUGC"
        trunk_embeddings = {}

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == ["65", "85", "71", "67"]  # Converted to strings

    def test_sequence_from_input_features_list(self):
        """Test extracting sequence from input_features as list."""
        # Arrange
        sequence = None
        input_features = {"sequence": ["A", "U", "G", "C"]}
        trunk_embeddings = {}

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == ["A", "U", "G", "C"]

    def test_sequence_from_input_features_string(self):
        """Test extracting sequence from input_features as string."""
        # Arrange
        sequence = None
        input_features = {"sequence": "AUGC"}
        trunk_embeddings = {}

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == ["A", "U", "G", "C"]

    @patch("rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils.logger")
    def test_fallback_to_trunk_embeddings_s_trunk(self, mock_logger):
        """Test fallback to trunk_embeddings with s_trunk key."""
        # Arrange
        sequence = None
        input_features = None
        trunk_embeddings = {"s_trunk": torch.zeros((1, 4, 16))}  # Batch, seq_len, dim

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == ["A", "A", "A", "A"]  # 4 placeholder 'A's
        mock_logger.warning.assert_called_once()
        assert "No sequence provided" in mock_logger.warning.call_args[0][0]

    @patch("rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils.logger")
    def test_fallback_to_trunk_embeddings_s_inputs(self, mock_logger):
        """Test fallback to trunk_embeddings with s_inputs key."""
        # Arrange
        sequence = None
        input_features = None
        trunk_embeddings = {"s_inputs": torch.zeros((1, 5, 16))}  # Batch, seq_len, dim

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == ["A", "A", "A", "A", "A"]  # 5 placeholder 'A's
        mock_logger.warning.assert_called_once()
        assert "No sequence provided" in mock_logger.warning.call_args[0][0]

    @patch("rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils.logger")
    def test_fallback_to_trunk_embeddings_sing(self, mock_logger):
        """Test fallback to trunk_embeddings with sing key."""
        # Arrange
        sequence = None
        input_features = None
        trunk_embeddings = {"sing": torch.zeros((1, 3, 16))}  # Batch, seq_len, dim

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == ["A", "A", "A"]  # 3 placeholder 'A's
        mock_logger.warning.assert_called_once()
        assert "No sequence provided" in mock_logger.warning.call_args[0][0]

    def test_error_when_no_sequence_available(self):
        """Test error when no sequence can be derived."""
        # Arrange
        sequence = None
        input_features = None
        trunk_embeddings = {}  # Empty dict

        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            extract_sequence(sequence, input_features, trunk_embeddings)

        assert "Cannot derive sequence" in str(excinfo.value)

    def test_input_features_without_sequence(self):
        """Test when input_features is provided but doesn't contain sequence."""
        # Arrange
        sequence = None
        input_features = {"other_key": "value"}  # No sequence key
        trunk_embeddings = {}

        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            extract_sequence(sequence, input_features, trunk_embeddings)

        assert "Cannot derive sequence" in str(excinfo.value)

    def test_trunk_embeddings_with_non_tensor_values(self):
        """Test when trunk_embeddings contains keys but with non-tensor values."""
        # Arrange
        sequence = None
        input_features = None
        trunk_embeddings = {"s_trunk": "not a tensor"}

        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            extract_sequence(sequence, input_features, trunk_embeddings)

        assert "Cannot derive sequence" in str(excinfo.value)

    @given(
        seq_list=st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=10)
    )
    def test_property_based_explicit_sequence(self, seq_list):
        """Property-based test for explicitly provided sequence."""
        # Arrange
        input_features = None
        trunk_embeddings = {}

        # Act
        result = extract_sequence(seq_list, input_features, trunk_embeddings)

        # Assert
        assert result == seq_list
        assert result is seq_list  # Should return the same object

    @given(
        seq_tensor=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=10)
    )
    def test_property_based_sequence_from_tensor(self, seq_tensor):
        """Property-based test for sequence from tensor."""
        # Arrange
        sequence = None
        input_features = {"sequence": torch.tensor(seq_tensor)}
        trunk_embeddings = {}

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == [str(x) for x in seq_tensor]

    @given(
        seq_str=st.text(min_size=1, max_size=10, alphabet="AUGC")
    )
    def test_property_based_sequence_from_string(self, seq_str):
        """Property-based test for sequence from string."""
        # Arrange
        sequence = None
        input_features = {"sequence": seq_str}
        trunk_embeddings = {}

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == list(seq_str)

    @given(
        seq_len=st.integers(min_value=1, max_value=20)
    )
    @patch("rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils.logger")
    def test_property_based_fallback_to_trunk_embeddings(self, mock_logger, seq_len):
        """Property-based test for fallback to trunk_embeddings."""
        # Arrange
        sequence = None
        input_features = None
        trunk_embeddings = {"s_trunk": torch.zeros((1, seq_len, 16))}

        # Act
        result = extract_sequence(sequence, input_features, trunk_embeddings)

        # Assert
        assert result == ["A"] * seq_len
        mock_logger.warning.assert_called_once()
        assert "No sequence provided" in mock_logger.warning.call_args[0][0]
        assert str(seq_len) in mock_logger.warning.call_args[0][0]
