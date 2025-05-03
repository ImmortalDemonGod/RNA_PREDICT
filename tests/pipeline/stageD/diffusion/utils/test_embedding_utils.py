"""
Tests for embedding_utils.py module.

This module tests the embedding utility functions for Stage D diffusion.
"""

import pytest
import torch
from unittest.mock import patch

from rna_predict.pipeline.stageD.diffusion.utils.embedding_utils import (
    ensure_s_inputs,
    ensure_z_trunk,
    EmbeddingContext,
)


class TestEnsureSInputs:
    """Tests for the ensure_s_inputs function."""

    def test_s_inputs_already_exists(self):
        """Test when s_inputs already exists in trunk_embeddings_internal."""
        # Arrange
        trunk_embeddings_internal = {
            "s_trunk": torch.zeros((1, 5, 384)),
            "s_inputs": torch.ones((1, 5, 449)),
        }
        original_trunk_embeddings_ref = {}
        input_features = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": 449}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_s_inputs(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            input_features,
            context,
        )

        # Assert
        assert "s_inputs" in trunk_embeddings_internal
        assert torch.all(trunk_embeddings_internal["s_inputs"] == 1.0)
        assert trunk_embeddings_internal["s_inputs"].shape == (1, 5, 449)

    def test_s_inputs_from_input_features_sing(self):
        """Test when s_inputs is obtained from input_features 'sing' key."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        input_features = {"sing": torch.ones((1, 5, 449))}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": 449}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_s_inputs(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            input_features,
            context,
        )

        # Assert
        assert "s_inputs" in trunk_embeddings_internal
        assert torch.all(trunk_embeddings_internal["s_inputs"] == 1.0)
        assert trunk_embeddings_internal["s_inputs"].shape == (1, 5, 449)

    @patch("rna_predict.pipeline.stageD.diffusion.utils.embedding_utils.logger")
    def test_s_inputs_fallback_creation(self, mock_logger):
        """Test fallback creation of s_inputs when not found in trunk_embeddings or input_features."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        input_features = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": 449}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_s_inputs(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            input_features,
            context,
        )

        # Assert
        assert "s_inputs" in trunk_embeddings_internal
        assert trunk_embeddings_internal["s_inputs"].shape == (1, 5, 449)
        assert torch.all(trunk_embeddings_internal["s_inputs"] == 0.0)
        mock_logger.warning.assert_called_once()
        assert "'s_inputs' not found" in mock_logger.warning.call_args[0][0]

    @patch("rna_predict.pipeline.stageD.diffusion.utils.embedding_utils.logger")
    def test_s_inputs_fallback_with_get_embedding_dimension(self, mock_logger):
        """Test fallback creation of s_inputs using get_embedding_dimension."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        input_features = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_s_inputs(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            input_features,
            context,
        )

        # Assert
        assert "s_inputs" in trunk_embeddings_internal
        assert trunk_embeddings_internal["s_inputs"].shape == (1, 5, 449)  # Default value
        assert torch.all(trunk_embeddings_internal["s_inputs"] == 0.0)
        mock_logger.warning.assert_called_once()

    @patch("rna_predict.pipeline.stageD.diffusion.utils.embedding_utils.logger")
    def test_s_inputs_update_original_ref(self, mock_logger):
        """Test that original_trunk_embeddings_ref is updated when s_inputs is created."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        input_features = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": 449}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_s_inputs(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            input_features,
            context,
        )

        # Assert
        assert "s_inputs" in original_trunk_embeddings_ref
        assert original_trunk_embeddings_ref["s_inputs"].shape == (1, 5, 449)
        assert torch.all(original_trunk_embeddings_ref["s_inputs"] == 0.0)
        mock_logger.debug.assert_called_once()
        assert "Copying generated 's_inputs'" in mock_logger.debug.call_args[0][0]

    @patch("rna_predict.pipeline.stageD.diffusion.utils.embedding_utils.logger")
    def test_s_inputs_no_update_original_ref_when_exists(self, mock_logger):
        """Test that original_trunk_embeddings_ref is not updated when s_inputs already exists there."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {"s_inputs": torch.ones((1, 5, 449))}
        input_features = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": 449}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_s_inputs(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            input_features,
            context,
        )

        # Assert
        assert "s_inputs" in trunk_embeddings_internal
        assert torch.all(trunk_embeddings_internal["s_inputs"] == 0.0)  # New tensor created
        assert torch.all(original_trunk_embeddings_ref["s_inputs"] == 1.0)  # Original unchanged
        mock_logger.debug.assert_not_called()

    def test_different_device(self):
        """Test that tensors are created on the specified device."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device test")

        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384), device="cuda")}
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": 449}
        )
        device = "cuda"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_s_inputs(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            {},
            context,
        )

        # Assert
        assert "s_inputs" in trunk_embeddings_internal
        assert trunk_embeddings_internal["s_inputs"].device.type == "cuda"


class TestEnsureZTrunk:
    """Tests for the ensure_z_trunk function."""

    def test_pair_already_exists(self):
        """Test when pair (z_trunk) already exists in trunk_embeddings_internal."""
        # Arrange
        trunk_embeddings_internal = {
            "s_trunk": torch.zeros((1, 5, 384)),
            "pair": torch.ones((1, 5, 5, 32)),
        }
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_z": 32}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_z_trunk(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            context,
        )

        # Assert
        assert "pair" in trunk_embeddings_internal
        assert torch.all(trunk_embeddings_internal["pair"] == 1.0)
        assert trunk_embeddings_internal["pair"].shape == (1, 5, 5, 32)

    @patch("rna_predict.pipeline.stageD.diffusion.utils.embedding_utils.logger")
    def test_pair_fallback_creation(self, mock_logger):
        """Test fallback creation of pair (z_trunk) when not found in trunk_embeddings."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_z": 32}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_z_trunk(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            context,
        )

        # Assert
        assert "pair" in trunk_embeddings_internal, "Fallback: 'pair' key should be created in trunk_embeddings_internal"
        assert trunk_embeddings_internal["pair"].shape == (1, 5, 5, 32), "Fallback: 'pair' tensor shape mismatch"
        assert torch.all(trunk_embeddings_internal["pair"] == 0.0), "Fallback: 'pair' tensor should be all zeros"
        mock_logger.warning.assert_called_once()
        assert "Fallback: Creating dummy 'z_trunk'" in mock_logger.warning.call_args[0][0]

    @patch("rna_predict.pipeline.stageD.diffusion.utils.embedding_utils.logger")
    def test_pair_fallback_with_get_embedding_dimension(self, mock_logger):
        """Test fallback creation of pair using get_embedding_dimension."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_z_trunk(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            context,
        )

        # Assert
        assert "pair" in trunk_embeddings_internal
        assert trunk_embeddings_internal["pair"].shape == (1, 5, 5, 128)  # Default value
        assert torch.all(trunk_embeddings_internal["pair"] == 0.0)
        mock_logger.warning.assert_called_once()

    @patch("rna_predict.pipeline.stageD.diffusion.utils.embedding_utils.logger")
    def test_pair_update_original_ref(self, mock_logger):
        """Test that original_trunk_embeddings_ref is updated when pair is created."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_z": 32}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_z_trunk(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            context,
        )

        # Assert
        assert "pair" in original_trunk_embeddings_ref
        assert original_trunk_embeddings_ref["pair"].shape == (1, 5, 5, 32)
        assert torch.all(original_trunk_embeddings_ref["pair"] == 0.0)
        mock_logger.debug.assert_called_once()
        assert "Copying generated 'pair'" in mock_logger.debug.call_args[0][0]

    @patch("rna_predict.pipeline.stageD.diffusion.utils.embedding_utils.logger")
    def test_pair_no_update_original_ref_when_exists(self, mock_logger):
        """Test that original_trunk_embeddings_ref is not updated when pair already exists there."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {"pair": torch.ones((1, 5, 5, 32))}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_z": 32}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_z_trunk(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            context,
        )

        # Assert
        assert "pair" in trunk_embeddings_internal
        assert torch.all(trunk_embeddings_internal["pair"] == 0.0)  # New tensor created
        assert torch.all(original_trunk_embeddings_ref["pair"] == 1.0)  # Original unchanged
        mock_logger.debug.assert_not_called()

    def test_different_device(self):
        """Test that tensors are created on the specified device."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device test")

        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384), device="cuda")}
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_z": 32}
        )
        device = "cuda"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_z_trunk(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            context,
        )

        # Assert
        assert "pair" in trunk_embeddings_internal
        assert trunk_embeddings_internal["pair"].device.type == "cuda"


class TestPropertyBasedTests:
    """Property-based tests for embedding_utils functions."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,c_s,c_s_inputs",
        [
            (1, 5, 384, 449),
            (2, 10, 256, 512),
            (3, 15, 128, 256),
        ],
    )
    def test_ensure_s_inputs_various_shapes(self, batch_size, seq_len, c_s, c_s_inputs):
        """Test ensure_s_inputs with various tensor shapes."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((batch_size, seq_len, c_s))}
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((batch_size, seq_len, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": c_s_inputs}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_s_inputs(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            {},
            context,
        )

        # Assert
        assert "s_inputs" in trunk_embeddings_internal, f"s_inputs missing for shape ({batch_size}, {seq_len}, {c_s}, {c_s_inputs})"
        # The function always creates a tensor with batch size 1, regardless of input batch size
        # This is the actual behavior of the function, not a bug in our test
        assert trunk_embeddings_internal["s_inputs"].shape == (1, seq_len, c_s_inputs), f"Shape mismatch for ({batch_size}, {seq_len}, {c_s}, {c_s_inputs})"

    @pytest.mark.parametrize(
        "batch_size,seq_len,c_s,c_z",
        [
            (1, 5, 384, 32),
            (2, 10, 256, 64),
            (3, 15, 128, 128),
        ],
    )
    def test_ensure_z_trunk_various_shapes(self, batch_size, seq_len, c_s, c_z):
        """Test ensure_z_trunk with various tensor shapes."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((batch_size, seq_len, c_s))}
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((batch_size, seq_len, 3)),
            trunk_embeddings={},
            diffusion_config={"c_z": c_z}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act
        ensure_z_trunk(
            trunk_embeddings_internal,
            original_trunk_embeddings_ref,
            context,
        )

        # Assert
        assert "pair" in trunk_embeddings_internal, f"pair missing for shape ({batch_size}, {seq_len}, {c_s}, {c_z})"
        # The function always creates a tensor with batch size 1, regardless of input batch size
        # This is the actual behavior of the function, not a bug in our test
        assert trunk_embeddings_internal["pair"].shape == (1, seq_len, seq_len, c_z), f"Shape mismatch for ({batch_size}, {seq_len}, {c_s}, {c_z})"


class TestEdgeCases:
    """Tests for edge cases in embedding_utils functions."""

    def test_empty_trunk_embeddings(self):
        """Test with empty trunk_embeddings_internal."""
        # Arrange
        trunk_embeddings_internal = {}
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": 449}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act & Assert
        with pytest.raises(KeyError):
            ensure_s_inputs(
                trunk_embeddings_internal,
                original_trunk_embeddings_ref,
                {},
                context,
            )

    def test_none_input_features(self):
        """Test with None input_features."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        input_features = None
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": 449}
        )
        device = "cpu"

        # Create embedding context
        context = EmbeddingContext(
            diffusion_config=diffusion_config,
            device=device
        )

        # Act & Assert
        # The function doesn't handle None for input_features, which is a limitation
        # In a real-world scenario, input_features would be an empty dict, not None
        with pytest.raises(AttributeError):
            ensure_s_inputs(
                trunk_embeddings_internal,
                original_trunk_embeddings_ref,
                input_features,
                context,
            )

    def test_none_diffusion_config(self):
        """Test with None diffusion_config."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        input_features = {}
        diffusion_config = None
        device = "cpu"

        # Act & Assert
        # The function doesn't handle None for diffusion_config, which is a limitation
        with pytest.raises(AttributeError):
            # Create embedding context with None diffusion_config
            context = EmbeddingContext(
                diffusion_config=diffusion_config,
                device=device
            )

            ensure_s_inputs(
                trunk_embeddings_internal,
                original_trunk_embeddings_ref,
                input_features,
                context,
            )

    def test_invalid_device(self):
        """Test with invalid device."""
        # Arrange
        trunk_embeddings_internal = {"s_trunk": torch.zeros((1, 5, 384))}
        original_trunk_embeddings_ref = {}
        from rna_predict.pipeline.stageD.diffusion.utils.config_types import DiffusionConfig
        diffusion_config = DiffusionConfig(
            partial_coords=torch.zeros((1, 5, 3)),
            trunk_embeddings={},
            diffusion_config={"c_s_inputs": 449}
        )
        device = "invalid_device"

        # Act & Assert
        with pytest.raises(RuntimeError, match="invalid_device.*|device.*invalid_device.*not available"):
            context = EmbeddingContext(
                diffusion_config=diffusion_config,
                device=device
            )
            ensure_s_inputs(
                trunk_embeddings_internal,
                original_trunk_embeddings_ref,
                {},
                context,
            )
