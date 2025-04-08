import pytest
import torch

from rna_predict.pipeline.stageA.input_embedding.current.primitives.atom_pair_transforms import (
    AtomPairConfig,
    _map_tokens_to_atoms,
    _validate_token_feats_shape,
    broadcast_token_to_local_atom_pair,
    gather_pair_embedding_in_dense_trunk,
)


class TestAtomPairTransforms:
    """Tests for atom_pair_transforms module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size = 2
        n_tokens = 3
        n_atoms = 6
        n_atoms_per_token = 2
        feature_dim = 4

        # Create token features
        token_feats = torch.randn(batch_size, n_tokens, feature_dim)

        # Create atom to token mapping (each token maps to n_atoms_per_token atoms)
        atom_to_token_idx = torch.zeros(batch_size, n_atoms, dtype=torch.long)
        for b in range(batch_size):
            for t in range(n_tokens):
                atom_to_token_idx[
                    b, t * n_atoms_per_token : (t + 1) * n_atoms_per_token
                ] = t

        # Create indices for gather operation
        idx_q = torch.randint(0, n_atoms, (batch_size, 3))
        idx_k = torch.randint(0, n_atoms, (batch_size, 3))

        # Input tensor for gather operation
        x = torch.randn(batch_size, n_atoms, feature_dim)

        return {
            "token_feats": token_feats,
            "atom_to_token_idx": atom_to_token_idx,
            "n_atoms_per_token": n_atoms_per_token,
            "idx_q": idx_q,
            "idx_k": idx_k,
            "x": x,
        }

    def test_validate_token_feats_shape_valid(self, sample_data):
        """Test validation of token features shape with valid input."""
        token_feats = sample_data["token_feats"]
        _validate_token_feats_shape(token_feats, 3)  # Should not raise

    def test_validate_token_feats_shape_invalid(self, sample_data):
        """Test validation of token features shape with invalid input."""
        token_feats = sample_data["token_feats"].unsqueeze(0)  # Make 4D
        with pytest.raises(ValueError):
            _validate_token_feats_shape(token_feats, 3)

    def test_map_tokens_to_atoms(self, sample_data):
        """Test mapping of tokens to atoms."""
        token_feats = sample_data["token_feats"]
        atom_to_token_idx = sample_data["atom_to_token_idx"]
        n_atoms_per_token = sample_data["n_atoms_per_token"]

        config = AtomPairConfig(
            n_batch=token_feats.shape[0],
            n_tokens=token_feats.shape[1],
            n_atoms_per_token=n_atoms_per_token,
            atom_to_token_idx=atom_to_token_idx,
        )

        result = _map_tokens_to_atoms(token_feats, config)

        # Check shape
        expected_atoms = atom_to_token_idx.shape[1]
        assert result.shape == (
            token_feats.shape[0],
            expected_atoms,
            token_feats.shape[2],
        )

    def test_broadcast_token_to_local_atom_pair_with_config(self, sample_data):
        """Test broadcasting tokens to local atom pairs with config object."""
        token_feats = sample_data["token_feats"]
        atom_to_token_idx = sample_data["atom_to_token_idx"]
        n_atoms_per_token = sample_data["n_atoms_per_token"]

        config = AtomPairConfig(
            n_batch=token_feats.shape[0],
            n_tokens=token_feats.shape[1],
            n_atoms_per_token=n_atoms_per_token,
            atom_to_token_idx=atom_to_token_idx,
        )

        result = broadcast_token_to_local_atom_pair(token_feats, config)

        # Check shape
        n_atoms = atom_to_token_idx.shape[1]
        assert result.shape == (
            token_feats.shape[0],
            n_atoms,
            n_atoms,
            token_feats.shape[2] * 2,
        )

    def test_broadcast_token_to_local_atom_pair_with_dict(self, sample_data):
        """Test broadcasting tokens to local atom pairs with dict input."""
        token_feats = sample_data["token_feats"]
        atom_to_token_idx = sample_data["atom_to_token_idx"]
        n_atoms_per_token = sample_data["n_atoms_per_token"]

        atom_config = {
            "atom_to_token_idx": atom_to_token_idx,
            "n_atoms_per_token": n_atoms_per_token,
        }

        result = broadcast_token_to_local_atom_pair(token_feats, atom_config)

        # Check shape
        n_atoms = atom_to_token_idx.shape[1]
        assert result.shape == (
            token_feats.shape[0],
            n_atoms,
            n_atoms,
            token_feats.shape[2] * 2,
        )

    def test_broadcast_token_to_local_atom_pair_missing_params(self, sample_data):
        """Test broadcasting with missing parameters."""
        token_feats = sample_data["token_feats"]

        # Missing required parameters
        atom_config = {}

        with pytest.raises(ValueError):
            broadcast_token_to_local_atom_pair(token_feats, atom_config)

    def test_gather_pair_embedding_in_dense_trunk(self, sample_data):
        """Test gathering pair embeddings in dense trunks."""
        x = sample_data["x"]
        idx_q = sample_data["idx_q"]
        idx_k = sample_data["idx_k"]

        result = gather_pair_embedding_in_dense_trunk(x, (idx_q, idx_k))

        # Check shape
        assert result.shape == (*idx_q.shape, x.shape[2] * 2)

    def test_gather_pair_embedding_invalid_input(self, sample_data):
        """Test gathering with invalid input."""
        x = sample_data["x"].unsqueeze(0)  # Make 4D
        idx_q = sample_data["idx_q"]
        idx_k = sample_data["idx_k"]

        with pytest.raises(ValueError):
            gather_pair_embedding_in_dense_trunk(x, (idx_q, idx_k))

    def test_gather_pair_embedding_mismatched_indices(self, sample_data):
        """Test gathering with mismatched indices."""
        x = sample_data["x"]
        idx_q = sample_data["idx_q"]
        idx_k = sample_data["idx_k"][0:1]  # Different shape

        with pytest.raises(ValueError):
            gather_pair_embedding_in_dense_trunk(x, (idx_q, idx_k))

    def test_atom_pair_config_creation(self, sample_data):
        """Test creation of AtomPairConfig from tensors."""
        token_feats = sample_data["token_feats"]
        atom_to_token_idx = sample_data["atom_to_token_idx"]
        n_atoms_per_token = sample_data["n_atoms_per_token"]

        atom_config = {
            "atom_to_token_idx": atom_to_token_idx,
            "n_atoms_per_token": n_atoms_per_token,
        }

        config = AtomPairConfig.from_tensors(token_feats, atom_config)

        assert config.n_batch == token_feats.shape[0]
        assert config.n_tokens == token_feats.shape[1]
        assert config.n_atoms_per_token == n_atoms_per_token
        assert config.atom_to_token_idx is atom_to_token_idx  # Same object
